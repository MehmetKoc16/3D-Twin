import { BufferAttribute } from 'three';
import {
  createGarmentSections,
  type GarmentCategory,
  type GarmentSections,
  type GarmentTemplateDef,
  type MeasuresDef,
  type StoreItemDef,
} from '@dt/avatar-core';
import { useWardrobeStore } from '../../store/wardrobeStore';
import type { AvatarAssets } from '../avatar/avatarAssets';
import type { SolveEnvelope } from '../../workers/avatarClient';
import { filterBodyIndex, hiddenVertexMask } from './bodyHide';
import type { Surface } from './garmentCollide';
import { GarmentInstance, type GarmentUpdateContext } from './garmentInstance';
import { isTuckedTop } from './garmentStyle';
import { bodyPlanes, type BodyPlanes } from './garmentGrading';
import { loadTemplateRuntime, type TemplateRuntime } from './templateLoader';

const CATEGORIES: readonly GarmentCategory[] = ['top', 'bottom', 'shoes'];
/** Inner layers first: shoes, trousers over the shoes, then the top over the trousers. */
const FIT_ORDER: readonly GarmentCategory[] = ['shoes', 'bottom', 'top'];
/** A tucked top is inner to the trousers: it is fitted before them and they are pushed out of it. */
const FIT_ORDER_TUCKED: readonly GarmentCategory[] = ['shoes', 'top', 'bottom'];

interface LastSolve {
  body: Float32Array;
  joints: Float32Array;
  sections: GarmentSections;
  achievedCm: SolveEnvelope['result']['achievedCm'];
}

/**
 * Mounts the worn garments into the avatar scene and keeps them fitted. Lives next to the avatar (features/avatar
 * creates it and forwards every solve result after the body geometry and skeleton were updated):
 *
 * - `onSolve`: keeps the solved render positions, computes the measure planes and re-fits every worn garment
 *   (`bindGarment` from the solved body, `gradeGarment` towards the chart size, welded normals, optional heatmap).
 * - store changes (worn set, chart / size / colour, heatmap) create, remove or refresh instances and rebuild the
 *   body index that hides triangles under the garments.
 */
export class WardrobeRig {
  private readonly instances = new Map<GarmentCategory, GarmentInstance>();
  private readonly runtimes = new Map<string, Promise<TemplateRuntime>>();
  private readonly pending = new Map<GarmentCategory, string>();
  private readonly originalIndex: BufferAttribute;
  private readonly unsubscribe: () => void;
  private last: LastSolve | null = null;
  private planes: BodyPlanes | undefined;
  private measures: MeasuresDef | null = null;
  private hiddenKey = '';
  private heatmap: boolean;
  private disposed = false;

  constructor(private readonly assets: AvatarAssets) {
    const index = assets.mesh.geometry.getIndex();
    if (!index) throw new Error('avatar body has no index buffer');
    this.originalIndex = index as BufferAttribute;
    this.heatmap = useWardrobeStore.getState().heatmap;
    void this.loadMeasures();
    this.unsubscribe = useWardrobeStore.subscribe((state, previous) => {
      if (
        state.worn !== previous.worn ||
        state.items !== previous.items ||
        state.templates !== previous.templates ||
        state.heatmap !== previous.heatmap
      ) {
        this.sync();
      }
    });
    this.sync();
  }

  private async loadMeasures(): Promise<void> {
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}assets/body/measures.json`);
      if (!response.ok) return;
      this.measures = (await response.json()) as MeasuresDef;
      if (this.last) {
        this.planes = bodyPlanes(this.measures, this.last.body);
        this.refreshAll();
      }
    } catch {
      /* without planes garments are bound but not graded */
    }
  }

  /** Called by the avatar after each solve was applied to the body geometry and skeleton. */
  onSolve(envelope: SolveEnvelope): void {
    this.last = {
      body: envelope.result.positions,
      joints: envelope.result.joints,
      sections: createGarmentSections(envelope.result.positions),
      achievedCm: envelope.result.achievedCm,
    };
    this.planes = this.measures ? bodyPlanes(this.measures, this.last.body) : undefined;
    this.refreshAll();
  }

  private baseUrl(): string {
    return `${import.meta.env.BASE_URL}assets/garments/`;
  }

  private runtimeFor(def: GarmentTemplateDef): Promise<TemplateRuntime> {
    let runtime = this.runtimes.get(def.id);
    if (!runtime) {
      const geometry = this.assets.mesh.geometry;
      runtime = loadTemplateRuntime(
        def,
        {
          renderVertexCount: geometry.getAttribute('position').count,
          skinIndex: geometry.getAttribute('skinIndex').array,
          skinWeight: geometry.getAttribute('skinWeight').array,
        },
        this.baseUrl(),
      );
      runtime.catch(() => this.runtimes.delete(def.id));
      this.runtimes.set(def.id, runtime);
    }
    return runtime;
  }

  /** Reconciles the mounted instances with the store's worn set. */
  private sync(): void {
    if (this.disposed) return;
    const state = useWardrobeStore.getState();
    this.heatmap = state.heatmap;
    for (const category of CATEGORIES) {
      const itemId = state.worn[category];
      const item = itemId ? state.items.find((i) => i.id === itemId) : undefined;
      const template = item ? state.templates.find((t) => t.id === item.templateId) : undefined;
      const existing = this.instances.get(category);
      if (!item || !template || template.category !== category) {
        this.pending.delete(category);
        if (existing) this.remove(category);
        continue;
      }
      if (existing && existing.item.id === item.id && existing.template.id === template.id) {
        existing.setItem(item);
        continue;
      }
      if (existing) this.remove(category);
      if (this.pending.get(category) !== item.id) {
        this.pending.set(category, item.id);
        void this.create(category, item, template);
      }
    }
    this.updateBodyIndex();
    this.refreshAll();
  }

  private async create(
    category: GarmentCategory,
    item: StoreItemDef,
    template: GarmentTemplateDef,
  ): Promise<void> {
    try {
      const runtime = await this.runtimeFor(template);
      if (this.disposed || this.pending.get(category) !== item.id) return;
      this.pending.delete(category);
      const current = useWardrobeStore.getState();
      const latest = current.items.find((i) => i.id === item.id);
      if (!latest || current.worn[category] !== item.id) return;
      this.instances.get(category)?.dispose();
      const instance = new GarmentInstance(
        runtime,
        latest,
        this.assets.skeleton,
        this.assets.mesh.bindMatrix,
      );
      this.assets.scene.add(instance.mesh);
      this.instances.set(category, instance);
      this.updateBodyIndex();
      this.refreshAll();
    } catch (error) {
      if (this.pending.get(category) === item.id) this.pending.delete(category);
      console.error(`Could not load garment ${template.id}`, error);
      useWardrobeStore.getState().takeOff(category);
    }
  }

  private remove(category: GarmentCategory): void {
    this.instances.get(category)?.dispose();
    this.instances.delete(category);
    this.applySoleLift();
  }

  /** A tucked top (e.g. the tucked t-shirt) is worn together with a bottom: the bottom's waist is the outer layer. */
  private tuckedTop(): GarmentInstance | undefined {
    const top = this.instances.get('top');
    return top && this.instances.has('bottom') && isTuckedTop(top.template) ? top : undefined;
  }

  /**
   * Inner garments an item is layered over: trousers over shoes and over a tucked top; an untucked top over trousers
   * of a lower layer number.
   */
  private lowersOf(category: GarmentCategory, instance: GarmentInstance): Surface[] {
    if (category === 'bottom') {
      const shoes = this.instances.get('shoes');
      const tucked = this.tuckedTop();
      return [...(shoes ? [shoes.surface()] : []), ...(tucked ? [tucked.surface()] : [])];
    }
    if (category === 'top') {
      if (this.tuckedTop()) return [];
      const bottom = this.instances.get('bottom');
      return bottom && bottom.template.layer < instance.template.layer ? [bottom.surface()] : [];
    }
    return [];
  }

  private refreshAll(): void {
    const last = this.last;
    if (!last) return;
    let coverageReady = false;
    const tucked = this.tuckedTop();
    // inner layers first: shoes, then trousers (over the shoes), then the top (over the trousers if it is a higher
    // layer); a tucked top goes before the trousers, which then lie over it
    const contextFor = (
      category: GarmentCategory,
      instance: GarmentInstance,
    ): GarmentUpdateContext => ({
      body: last.body,
      joints: last.joints,
      sections: last.sections,
      achievedCm: last.achievedCm,
      planes: this.planes,
      heatmap: this.heatmap,
      lowers: this.lowersOf(category, instance),
      tuckedHem: tucked !== undefined && category === 'top',
    });
    const guarded = (
      category: GarmentCategory,
      instance: GarmentInstance,
      run: () => void,
    ): void => {
      try {
        run();
      } catch (error) {
        console.error(`Could not fit garment ${instance.template.id}`, error);
        this.remove(category);
        useWardrobeStore.getState().takeOff(category);
      }
    };
    for (const category of tucked ? FIT_ORDER_TUCKED : FIT_ORDER) {
      const instance = this.instances.get(category);
      if (!instance) continue;
      guarded(category, instance, () => {
        const epoch = instance.coverageEpoch;
        instance.syncBind(this.assets.mesh.bindMatrix);
        const context = contextFor(category, instance);
        // a tucked top is only fitted here; it is uploaded once the trousers over it are in place
        if (tucked && category === 'top') instance.fit(context);
        else instance.update(context);
        if (instance.coverageEpoch !== epoch) coverageReady = true;
      });
    }
    if (tucked && this.instances.get('top') === tucked) {
      guarded('top', tucked, () =>
        tucked.commit(contextFor('top', tucked), this.instances.get('bottom')?.surface()),
      );
    }
    const top = this.instances.get('top');
    const bottom = this.instances.get('bottom');
    if (tucked && top && bottom) {
      // tucked: the trousers' waistband is the outer layer, the top's triangles behind it are dropped
      top.hideUnder(bottom.surface());
      bottom.hideUnder(null);
    } else {
      top?.hideUnder(null);
      bottom?.hideUnder(top && top.template.layer > bottom.template.layer ? top.surface() : null);
    }
    if (coverageReady) this.updateBodyIndex();
    this.applySoleLift();
  }

  /**
   * Shoe soles can sit below the floor on the neutral body: the whole avatar (body, garments, skeleton) is raised by
   * that depth so the soles stand at y = 0. The skinning math is unaffected (bind and bone matrices share the offset).
   */
  private applySoleLift(): void {
    const lift = this.instances.get('shoes')?.soleLift ?? 0;
    if (this.assets.scene.position.y !== lift) {
      this.assets.scene.position.y = lift;
      this.assets.scene.updateMatrixWorld(true);
    }
  }

  /**
   * Hides the body triangles under the worn garments: the pipeline's delete lists plus the triangles under each
   * garment's footprint (islands the lists leave behind, e.g. the navel, would poke through the fabric). Restores the
   * full index when nothing is worn.
   */
  private updateBodyIndex(): void {
    const parts = [...this.instances.values()].filter(
      (i) => i.runtime.deleteVerts.length > 0 || i.coverage !== null,
    );
    const key = parts
      .map((i) => `${i.template.id}${i.coverage ? `+${i.coverageEpoch}` : ''}`)
      .sort()
      .join('|');
    if (key === this.hiddenKey) return;
    this.hiddenKey = key;
    const geometry = this.assets.mesh.geometry;
    if (parts.length === 0) {
      geometry.setIndex(this.originalIndex);
      return;
    }
    const count = geometry.getAttribute('position').count;
    const mask = hiddenVertexMask(
      parts.map((i) => i.runtime.deleteVerts),
      count,
      this.assets.weld,
    );
    for (const part of parts) part.coverage?.forEach((v, i) => (mask[i] = mask[i]! | v));
    const filtered = filterBodyIndex(this.assets.indices, mask);
    geometry.setIndex(
      filtered === this.assets.indices ? this.originalIndex : new BufferAttribute(filtered, 1),
    );
  }

  dispose(): void {
    this.disposed = true;
    this.unsubscribe();
    for (const category of [...this.instances.keys()]) this.remove(category);
    this.assets.mesh.geometry.setIndex(this.originalIndex);
    this.assets.scene.position.y = 0;
    this.pending.clear();
  }
}
