import { BufferAttribute, BufferGeometry } from 'three';
import type { BodyPartCategory, BodyPartDef } from '@dt/avatar-core';
import type { SolveEnvelope } from '../../../workers/avatarClient';
import { effectiveEyebrowColor, useAppearanceStore } from '../../../store/appearanceStore';
import type { AvatarAssets } from '../avatarAssets';
import { filterBodyIndex, hiddenVertexMask } from './bodyMask';
import { PartInstance } from './partInstance';
import { findPart, loadPartsIndex, partsBaseUrl, type PartsIndex } from './partsIndex';
import { loadPartRuntime, type PartRuntime } from './partsRuntime';

type Source = BufferAttribute | number[] | null;

/**
 * Mounts the body parts (eyes, eyebrows, eyelashes, hair) on the avatar skeleton and keeps them bound to the body:
 *
 * - `onSolve` (called by the avatar after the body geometry and skeleton were updated, for EVERY solve including the
 *   ones that carry face-shape modifiers): `bindGarment` from the solved render positions + welded normals.
 * - appearance store changes: mount / swap / remove parts, recolour hair, eyebrows and iris.
 * - body triangles listed in the parts' delete lists (eye-socket cavity, tousled hair scalp) are hidden. The
 *   wardrobe also rewrites the body index; both compose through a `setIndex` hook that filters whatever index the
 *   wardrobe (or anyone) assigns, so neither needs to know about the other.
 */
export class PartsRig {
  private readonly instances = new Map<BodyPartCategory, PartInstance>();
  private readonly runtimes = new Map<string, Promise<PartRuntime>>();
  private readonly wanted = new Map<BodyPartCategory, string | null>();
  private readonly unsubscribe: () => void;
  private readonly geometry: BufferGeometry;
  private index: PartsIndex | null = null;
  private body: Float32Array | null = null;
  private disposed = false;

  // body index composition
  private baseIndex: Source;
  private maskKey = '';
  private mask: Uint8Array | null = null;
  private readonly assignIndex: (index: Source) => BufferGeometry;
  private composed: { source: Source; key: string; result: Source } | null = null;

  constructor(private readonly assets: AvatarAssets) {
    this.geometry = assets.mesh.geometry;
    this.baseIndex = this.geometry.getIndex();
    // BufferGeometry.setIndex is a prototype method; shadow it on this instance only while the rig lives.
    this.assignIndex = (index) => BufferGeometry.prototype.setIndex.call(this.geometry, index);
    const hooked = (index: Source): BufferGeometry => {
      this.baseIndex = index;
      return this.assignIndex(this.compose(index));
    };
    this.geometry.setIndex = hooked as BufferGeometry['setIndex'];
    this.unsubscribe = useAppearanceStore.subscribe((state, previous) => {
      if (state.hairId !== previous.hairId || state.eyebrowId !== previous.eyebrowId) this.sync();
      if (
        state.hairColor !== previous.hairColor ||
        state.eyebrowColor !== previous.eyebrowColor ||
        state.eyeColor !== previous.eyeColor
      )
        this.applyColors();
    });
    if (import.meta.env.DEV) {
      // probe for the e2e tests: which parts are mounted, and the recoloured eye texture
      (window as unknown as { __dtParts?: unknown }).__dtParts = {
        ids: (): string[] => [...this.instances.values()].map((i) => i.id),
        eyeCanvas: (): HTMLCanvasElement | null => this.instances.get('eyes')?.irisCanvas ?? null,
        hiddenBodyVertices: (): number => this.mask?.reduce((n, v) => n + v, 0) ?? 0,
      };
    }
    void loadPartsIndex().then(
      (index) => {
        if (this.disposed) return;
        this.index = index;
        this.sync();
      },
      (error: unknown) => console.error('Could not load the body parts index', error),
    );
  }

  /** Called by the avatar after each solve was applied to the body geometry and skeleton. */
  onSolve(envelope: SolveEnvelope): void {
    this.body = envelope.result.positions;
    for (const instance of this.instances.values()) this.fit(instance);
  }

  private fit(instance: PartInstance): void {
    if (!this.body) return;
    instance.syncBind(this.assets.mesh.bindMatrix);
    try {
      instance.update(this.body);
    } catch (error) {
      console.error(`Could not bind part ${instance.id}`, error);
      this.removeInstance(instance.runtime.def.category);
    }
  }

  private runtimeFor(def: BodyPartDef): Promise<PartRuntime> {
    let runtime = this.runtimes.get(def.id);
    if (!runtime) {
      const geometry = this.geometry;
      runtime = loadPartRuntime(
        def,
        {
          renderVertexCount: geometry.getAttribute('position').count,
          skinIndex: geometry.getAttribute('skinIndex').array,
          skinWeight: geometry.getAttribute('skinWeight').array,
        },
        partsBaseUrl(),
      );
      runtime.catch(() => this.runtimes.delete(def.id));
      this.runtimes.set(def.id, runtime);
    }
    return runtime;
  }

  /** Part id wanted for a category: the defaults for eyes and lashes, the store's choice for brows and hair. */
  private desired(category: BodyPartCategory): string | null {
    const index = this.index;
    if (!index) return null;
    const state = useAppearanceStore.getState();
    if (category === 'hair') return state.hairId;
    if (category === 'eyebrows')
      return findPart(index, state.eyebrowId)?.category === 'eyebrows' ? state.eyebrowId : (index.defaults.eyebrows ?? null);
    return index.defaults[category] ?? null;
  }

  /** Reconciles the mounted parts with the store. */
  private sync(): void {
    if (this.disposed || !this.index) return;
    for (const category of ['eyes', 'eyebrows', 'eyelashes', 'hair'] as const) {
      const id = this.desired(category);
      this.wanted.set(category, id);
      const existing = this.instances.get(category);
      if (existing && existing.id === id) continue;
      const def = findPart(this.index, id);
      if (!def || def.category !== category) {
        this.removeInstance(category);
        continue;
      }
      void this.create(category, def);
    }
  }

  private async create(category: BodyPartCategory, def: BodyPartDef): Promise<void> {
    try {
      const runtime = await this.runtimeFor(def);
      if (this.disposed || this.wanted.get(category) !== def.id) return;
      if (this.instances.get(category)?.id === def.id) return;
      this.removeInstance(category);
      const instance = new PartInstance(runtime, this.assets.skeleton, this.assets.mesh.bindMatrix);
      this.instances.set(category, instance);
      this.assets.scene.add(instance.mesh);
      this.applyColor(instance);
      this.fit(instance);
      this.updateMask();
    } catch (error) {
      console.error(`Could not load part ${def.id}`, error);
    }
  }

  private removeInstance(category: BodyPartCategory): void {
    const instance = this.instances.get(category);
    if (!instance) return;
    instance.dispose();
    this.instances.delete(category);
    this.updateMask();
  }

  private applyColor(instance: PartInstance): void {
    const state = useAppearanceStore.getState();
    switch (instance.runtime.def.category) {
      case 'hair':
        instance.setColor(state.hairColor);
        break;
      case 'eyebrows':
        instance.setColor(effectiveEyebrowColor(state));
        break;
      case 'eyes':
        instance.setColor(state.eyeColor);
        break;
      default:
        break;
    }
  }

  private applyColors(): void {
    for (const instance of this.instances.values()) this.applyColor(instance);
  }

  // --- hiding the body triangles behind the parts -------------------------------------------------------------

  private updateMask(): void {
    if (this.disposed) return;
    const lists = [...this.instances.values()].map((i) => i.runtime.deleteVerts).filter((l) => l.length > 0);
    const key = [...this.instances.values()]
      .filter((i) => i.runtime.deleteVerts.length > 0)
      .map((i) => i.id)
      .sort()
      .join('|');
    if (key === this.maskKey) return;
    this.maskKey = key;
    this.mask =
      lists.length > 0 ? hiddenVertexMask(lists, this.geometry.getAttribute('position').count, this.assets.weld) : null;
    this.composed = null;
    if (this.baseIndex) this.assignIndex(this.compose(this.baseIndex));
  }

  /** `index` without the triangles of the hidden vertices (cached per index object and mask). */
  private compose(index: Source): Source {
    if (!index || !this.mask) return index;
    if (this.composed && this.composed.source === index && this.composed.key === this.maskKey) return this.composed.result;
    const values: ArrayLike<number> = Array.isArray(index) ? index : (index as BufferAttribute).array;
    const filtered = filterBodyIndex(values, this.mask);
    const result = filtered === values ? index : new BufferAttribute(Uint32Array.from(filtered), 1);
    this.composed = { source: index, key: this.maskKey, result };
    return result;
  }

  dispose(): void {
    this.disposed = true;
    this.unsubscribe();
    for (const category of [...this.instances.keys()]) {
      const instance = this.instances.get(category);
      instance?.dispose();
      this.instances.delete(category);
    }
    Reflect.deleteProperty(this.geometry, 'setIndex'); // back to the prototype method
    this.mask = null;
    if (this.baseIndex) this.assignIndex(this.baseIndex);
  }
}
