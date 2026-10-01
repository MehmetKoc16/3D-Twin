import {
  BufferAttribute,
  BufferGeometry,
  DynamicDrawUsage,
  Float32BufferAttribute,
  Sphere,
  SkinnedMesh,
  Vector3,
} from 'three';
import type { SolveEnvelope } from '../../workers/avatarClient';
import type { TwinRuntimeInfo } from '../../store/twinStore';
import type { AvatarAssets } from '../avatar/avatarAssets';
import { headsFromJoints } from '../avatar/meshMath';
import { useWardrobeStore } from '../../store/wardrobeStore';
import { handVertexMask, SCAN_WRIST_M, TwinHands } from './twinHands';
import { pushInPositions, pushInWeights } from './twinPushIn';
import { coveredBodyVertices } from '../wardrobe/bodyHide';
import { restAlignment, translatePositions, type RestAlignment } from './twinBinding';
import type { TwinModel } from './twinModel';
import {
  compactTwinIndex,
  hiddenTwinVertices,
  unreferencedVertices,
  validateMapping,
} from './twinMapping';

type Source = BufferAttribute | number[] | null;

function sameValues(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Twin vertices this close to a worn garment's surface (either side) and under its footprint are hidden as well. */
export const FOOTPRINT_BAND_M = 0.03;

/** Largest rest-head difference (m) between the twin's skeleton and the avatar's before the two files are declared foreign. */
export const MAX_HEAD_RESIDUAL_M = 0.005;

/**
 * The realistic twin in the avatar scene. A SkinnedMesh with the twin's own vertices and skin weights bound to the
 * avatar's skeleton, exactly like a garment: PoseDriver, the camera focus presets and the worn garments all follow the
 * one skeleton (see twinBinding.ts). The MakeHuman body stays solved (with the twin's fitted shape) but hidden.
 *
 * - `onSolve` (called by the avatar after the body geometry and skeleton were updated): translates the twin's rest
 *   vertices onto the avatar's rest heads and follows the rebuilt bind matrix.
 * - Twin triangles under worn garments: the wardrobe removes the body triangles under each garment through
 *   `geometry.setIndex`; a twin triangle is hidden when its three vertices map (mh2twin.bin) to body vertices left
 *   without any triangle, or when they lie under a worn garment's surface (footprint pass, see `refreshHidden`). The rig
 *   shadows `setIndex` on the body geometry for the first pass (PartsRig does the same in standard mode; the two are
 *   never alive together).
 */
export class TwinRig {
  readonly mesh: SkinnedMesh;
  private readonly geometry = new BufferGeometry();
  private readonly position: BufferAttribute;
  private readonly indexArray: Uint32Array;
  private readonly indexAttribute: BufferAttribute;
  private readonly bodyGeometry: BufferGeometry;
  private readonly bodyVertexCount: number;
  private readonly unreferenced: Uint8Array;
  private readonly hiddenTwin: Uint8Array;
  private readonly bodyWasVisible: boolean;
  private alignment: RestAlignment | null = null;
  private bodyIndex: Source = null;
  private hiddenTriangles = 0;
  private disposed = false;
  private readonly hands: TwinHands;
  private readonly rest: Float32Array;
  private handMask: Uint8Array;
  private readonly offWardrobe: () => void;
  private pushWeights: Float32Array;
  private coverageKey = '';
  private coverageIndex = new Uint32Array();
  private readonly surfaceCache = new Map<
    string,
    {
      version: string;
      epoch: number;
      positions: Float32Array;
      index: Uint32Array;
    }
  >();

  constructor(
    private readonly assets: AvatarAssets,
    private readonly model: TwinModel,
    private readonly mapping: Uint32Array | null,
    private readonly onInfo: (info: TwinRuntimeInfo) => void = () => undefined,
    skinToneHex = `#${model.material.color.getHexString()}`,
  ) {
    this.bodyGeometry = assets.mesh.geometry;
    this.bodyVertexCount = this.bodyGeometry.getAttribute('position').count;
    if (mapping) validateMapping(mapping, model.vertexCount, this.bodyVertexCount);
    this.unreferenced = new Uint8Array(this.bodyVertexCount);
    this.hiddenTwin = new Uint8Array(model.vertexCount);
    this.rest = new Float32Array(model.position);
    this.handMask = new Uint8Array(model.vertexCount);
    this.pushWeights = new Float32Array(model.vertexCount);
    this.hands = new TwinHands(assets, skinToneHex);
    // Wardrobe refreshes in its synchronous subscriber first; this also catches size/colour changes without a new index.
    this.offWardrobe = useWardrobeStore.subscribe((state, previous) => {
      if (state.worn !== previous.worn || state.items !== previous.items)
        queueMicrotask(() => {
          if (!this.disposed) this.refreshHidden();
        });
    });

    this.position = new Float32BufferAttribute(new Float32Array(model.position), 3).setUsage(
      DynamicDrawUsage,
    );
    this.geometry.setAttribute('position', this.position);
    this.geometry.setAttribute('normal', new Float32BufferAttribute(model.normal, 3));
    if (model.uv) this.geometry.setAttribute('uv', new Float32BufferAttribute(model.uv, 2));
    this.geometry.setAttribute('skinIndex', new BufferAttribute(model.skinIndex, 4));
    this.geometry.setAttribute('skinWeight', new BufferAttribute(model.skinWeight, 4));
    this.indexArray = Uint32Array.from(model.index);
    this.indexAttribute = new BufferAttribute(this.indexArray, 1).setUsage(DynamicDrawUsage);
    this.geometry.setIndex(this.indexAttribute);

    this.mesh = new SkinnedMesh(this.geometry, model.material);
    this.mesh.name = 'twin';
    this.mesh.frustumCulled = false; // bounds of a posed skinned mesh are not tracked
    this.mesh.visible = false; // swapped in with the first fixed-shape solve that aligns it (no flash in the wrong place)
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    // a generous fixed bound keeps raycasting (double click to focus) independent of the pose
    this.mesh.boundingSphere = new Sphere(new Vector3(0, 0.95, 0), 2);
    this.mesh.bind(assets.skeleton, assets.mesh.bindMatrix);
    assets.scene.add(this.mesh);

    this.bodyWasVisible = assets.mesh.visible;
    // follow whatever index the wardrobe assigns to the (hidden) body
    const setIndex = BufferGeometry.prototype.setIndex;
    this.bodyGeometry.setIndex = ((index: Source): BufferGeometry => {
      const result = setIndex.call(this.bodyGeometry, index);
      this.followBodyIndex(index);
      return result;
    }) as BufferGeometry['setIndex'];
    this.followBodyIndex(this.bodyGeometry.getIndex());
    if (import.meta.env.DEV && typeof window !== 'undefined') this.installProbe();
  }

  /** Rest translation / residual of the last fixed-shape solve (null before the first one). */
  get lastAlignment(): RestAlignment | null {
    return this.alignment;
  }

  /**
   * Called by the avatar after each solve was applied to the body geometry and skeleton. Only solves of the twin's
   * fixed shape carry the heads the twin was rigged with; anything else (a stale standard solve) is ignored.
   */
  onSolve(envelope: SolveEnvelope): RestAlignment | null {
    if (this.disposed || envelope.result.fixedShape !== true) return null;
    const appHeads = headsFromJoints(envelope.result.joints, this.assets.rigIndexOfBone);
    const alignment = restAlignment(this.model.heads, this.model.remap, appHeads);
    this.alignment = alignment;
    if (alignment.maxResidual > MAX_HEAD_RESIDUAL_M) return alignment; // foreign files: the caller rejects them, nothing is shown
    translatePositions(this.model.position, alignment.offset, this.rest);
    this.handMask = handVertexMask(
      this.rest,
      this.model.skinIndex,
      this.model.skinWeight,
      this.assets.skeleton.bones.map((b) => b.name),
      appHeads,
      SCAN_WRIST_M,
    );
    this.hands.update(appHeads);
    this.position.needsUpdate = true;
    this.mesh.bindMatrix.copy(this.assets.mesh.bindMatrix);
    this.mesh.bindMatrixInverse.copy(this.assets.mesh.bindMatrix).invert();
    this.mesh.visible = true;
    this.assets.mesh.visible = false; // swap with the mannequin body in one go: it stays solved, only hidden
    this.refreshHidden(); // the garments were refitted to this solve before the twin was told
    return alignment;
  }

  private followBodyIndex(index: Source): void {
    if (this.disposed) return;
    this.bodyIndex = index;
    this.refreshHidden();
  }

  /**
   * Twin triangles to hide = those whose three vertices are hidden by (a) the mapping: the nearest body vertices lost
   * all their triangles (delete lists + the wardrobe's footprint pass on the body), or (b) the garment footprint on the
   * twin itself: the vertex lies under a worn garment within `FOOTPRINT_BAND_M` (the scan's own clothes sit a little
   * outside the fitted body, where the body-based pass would leave them poking through the new fabric).
   */
  private refreshHidden(): void {
    const index = this.bodyIndex;
    const bodyValues = index ? (Array.isArray(index) ? index : index.array) : [];
    const surfaces = this.garmentSurfaces();
    const key = `${this.alignment?.offset.join(',') ?? 'pending'}|${surfaces.map((s) => s.key).join('|')}`;
    if (key === this.coverageKey && sameValues(bodyValues, this.coverageIndex)) return;
    this.coverageKey = key;
    this.coverageIndex = Uint32Array.from(bodyValues);
    const hidden = this.hiddenTwin;
    hidden.fill(0);
    let any = false;
    if (this.mapping && index) {
      const values: ArrayLike<number> = Array.isArray(index)
        ? index
        : (index as BufferAttribute).array;
      unreferencedVertices(values, this.bodyVertexCount, this.unreferenced);
      hiddenTwinVertices(this.mapping, this.unreferenced, hidden);
      any = true;
    }
    if (this.alignment) {
      const positions = this.rest;
      for (const garment of surfaces) {
        const covered = coveredBodyVertices(
          positions,
          this.model.vertexCount,
          garment.positions,
          garment.index,
          {
            bandM: FOOTPRINT_BAND_M,
          },
        );
        for (let v = 0; v < hidden.length; v++) hidden[v] = hidden[v]! | covered[v]!;
        any = true;
      }
    }
    this.pushWeights = pushInWeights(this.rest, hidden);
    pushInPositions(
      this.rest,
      this.model.normal,
      this.pushWeights,
      this.position.array as Float32Array,
    );
    this.position.needsUpdate = true;
    // Recompute on the complete surface before compacting, so coverage boundaries retain correct normals.
    this.indexArray.set(this.model.index);
    this.geometry.computeVertexNormals();
    for (let v = 0; v < hidden.length; v++) hidden[v] = hidden[v]! | this.handMask[v]!;
    if (!any && !this.handMask.some(Boolean)) {
      this.showAll();
      return;
    }
    const count = compactTwinIndex(this.model.index, hidden, this.indexArray);
    // A triangle touching the hand region is removed entirely: fused scan fists never remain at the wrist edge.
    let kept = 0;
    for (let t = 0; t < count; t += 3) {
      if (
        this.handMask[this.indexArray[t]!] ||
        this.handMask[this.indexArray[t + 1]!] ||
        this.handMask[this.indexArray[t + 2]!]
      )
        continue;
      for (let j = 0; j < 3; j++) this.indexArray[kept++] = this.indexArray[t + j]!;
    }
    this.setDrawn(kept);
  }

  /**
   * The worn garments' fitted rest surfaces: the wardrobe mounts each one as a SkinnedMesh named `garment:<id>` in the
   * avatar scene (its position attribute holds the graded rest positions in the body's frame, the draw range the
   * triangles that are not hidden under an outer garment).
   */
  private garmentSurfaces(): { positions: Float32Array; index: ArrayLike<number>; key: string }[] {
    const out: { positions: Float32Array; index: ArrayLike<number>; key: string }[] = [];
    for (const child of this.assets.scene.children) {
      if (!child.name.startsWith('garment:') || !(child instanceof SkinnedMesh)) continue;
      const geometry = child.geometry;
      const index = geometry.getIndex();
      if (!index) continue;
      const count = Math.min(index.count, geometry.drawRange.count);
      const positions = geometry.getAttribute('position').array as Float32Array;
      const values = (index.array as Uint32Array).subarray(0, count);
      const version = `${(geometry.getAttribute('position') as BufferAttribute).version}:${index.version}:${count}`;
      let cached = this.surfaceCache.get(child.uuid);
      if (!cached || cached.version !== version) {
        if (
          !cached ||
          !sameValues(positions, cached.positions) ||
          !sameValues(values, cached.index)
        ) {
          cached = {
            version,
            epoch: (cached?.epoch ?? 0) + 1,
            positions: positions.slice(),
            index: Uint32Array.from(values),
          };
          this.surfaceCache.set(child.uuid, cached);
        } else cached.version = version;
      }
      out.push({
        key: `${child.uuid}:${cached.epoch}`,
        positions,
        index: values,
      });
    }
    const live = new Set(out.map((surface) => surface.key.split(':')[0]));
    for (const id of this.surfaceCache.keys()) if (!live.has(id)) this.surfaceCache.delete(id);
    return out;
  }

  private showAll(): void {
    this.indexArray.set(this.model.index);
    this.setDrawn(this.model.index.length);
  }

  private setDrawn(count: number): void {
    this.geometry.setDrawRange(0, count);
    this.indexAttribute.needsUpdate = true;
    this.hiddenTriangles = (this.model.index.length - count) / 3;
    this.report();
  }

  private report(): void {
    const a = this.alignment;
    this.onInfo({
      vertices: this.model.vertexCount,
      triangles: this.model.index.length / 3,
      hiddenTriangles: this.hiddenTriangles,
      offsetCm: a ? [a.offset[0] * 100, a.offset[1] * 100, a.offset[2] * 100] : [0, 0, 0],
      residualMm: a ? a.maxResidual * 1000 : 0,
    });
  }

  /** Dev-only probe for the e2e tests (skinned vertex positions and what is hidden). */
  private installProbe(): void {
    const scratch = new Vector3();
    (window as unknown as { __dtTwin?: unknown }).__dtTwin = {
      vertexCount: this.model.vertexCount,
      vertex: (i: number): [number, number, number] => {
        this.mesh.getVertexPosition(i, scratch);
        this.mesh.localToWorld(scratch);
        return [scratch.x, scratch.y, scratch.z];
      },
      hiddenTriangles: (): number => this.hiddenTriangles,
      handTriangles: (): number => (this.hands.mesh.geometry.getIndex()?.count ?? 0) / 3,
      handsVisible: (): boolean => this.hands.mesh.visible,
      handColor: (): string =>
        `#${(this.hands.mesh.material as import('three').MeshStandardMaterial).color.getHexString()}`,
      handVertices: (): number[][] => {
        const out: number[][] = [];
        for (let i = 0; i < this.hands.mesh.geometry.getAttribute('position').count; i++) {
          this.hands.mesh.getVertexPosition(i, scratch);
          this.hands.mesh.localToWorld(scratch);
          out.push([scratch.x, scratch.y, scratch.z]);
        }
        return out;
      },
      fingerVertices: (): number[][] => {
        const out: number[][] = [];
        const wrist = this.assets.skeleton.bones.find((b) => b.name === 'hand_l');
        const index = this.assets.skeleton.bones.findIndex((b) => b.name === 'index_03_l');
        if (!wrist) return out;
        const geometry = this.hands.mesh.geometry;
        const joints = geometry.getAttribute('skinIndex');
        const weights = geometry.getAttribute('skinWeight');
        if (!joints || !weights) return out;
        for (let v = 0; v < joints.count; v++) {
          if (joints.getX(v) !== index || weights.getX(v) < 0.75) continue;
          this.hands.mesh.getVertexPosition(v, scratch);
          this.hands.mesh.localToWorld(scratch);
          wrist.worldToLocal(scratch); // removes arm and wrist motion: only articulated fingers remain
          out.push([scratch.x, scratch.y, scratch.z]);
        }
        return out;
      },
      pushDistances: (): number[] => Array.from(this.pushWeights, (weight) => weight * 0.008),
      restPositions: (): number[] => Array.from(this.position.array),
      bodyVisible: (): boolean => this.assets.mesh.visible,
      twinVisible: (): boolean => this.mesh.visible && this.mesh.parent !== null,
      alignment: (): RestAlignment | null => this.alignment,
      bone: (name: string): { q: number[]; world: number[] } | null => {
        const bone = this.assets.skeleton.bones.find((b) => b.name === name);
        if (!bone) return null;
        bone.getWorldPosition(scratch);
        return {
          q: [bone.quaternion.x, bone.quaternion.y, bone.quaternion.z, bone.quaternion.w],
          world: [scratch.x, scratch.y, scratch.z],
        };
      },
    };
  }

  dispose(): void {
    this.disposed = true;
    this.offWardrobe();
    this.surfaceCache.clear();
    this.hands.dispose();
    Reflect.deleteProperty(this.bodyGeometry, 'setIndex'); // back to the prototype method
    this.assets.mesh.visible = this.bodyWasVisible;
    this.mesh.removeFromParent();
    this.geometry.dispose();
    this.model.dispose();
    if (import.meta.env.DEV && typeof window !== 'undefined')
      delete (window as unknown as { __dtTwin?: unknown }).__dtTwin;
  }
}
