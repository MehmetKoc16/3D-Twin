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

  constructor(
    private readonly assets: AvatarAssets,
    private readonly model: TwinModel,
    private readonly mapping: Uint32Array | null,
    private readonly onInfo: (info: TwinRuntimeInfo) => void = () => undefined,
  ) {
    this.bodyGeometry = assets.mesh.geometry;
    this.bodyVertexCount = this.bodyGeometry.getAttribute('position').count;
    if (mapping) validateMapping(mapping, model.vertexCount, this.bodyVertexCount);
    this.unreferenced = new Uint8Array(this.bodyVertexCount);
    this.hiddenTwin = new Uint8Array(model.vertexCount);

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
    translatePositions(this.model.position, alignment.offset, this.position.array as Float32Array);
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
      const positions = this.position.array as Float32Array;
      for (const garment of this.garmentSurfaces()) {
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
    if (!any) {
      this.showAll();
      return;
    }
    this.setDrawn(compactTwinIndex(this.model.index, hidden, this.indexArray));
  }

  /**
   * The worn garments' fitted rest surfaces: the wardrobe mounts each one as a SkinnedMesh named `garment:<id>` in the
   * avatar scene (its position attribute holds the graded rest positions in the body's frame, the draw range the
   * triangles that are not hidden under an outer garment).
   */
  private garmentSurfaces(): { positions: Float32Array; index: ArrayLike<number> }[] {
    const out: { positions: Float32Array; index: ArrayLike<number> }[] = [];
    for (const child of this.assets.scene.children) {
      if (!child.name.startsWith('garment:') || !(child instanceof SkinnedMesh)) continue;
      const geometry = child.geometry;
      const index = geometry.getIndex();
      if (!index) continue;
      const count = Math.min(index.count, geometry.drawRange.count);
      out.push({
        positions: geometry.getAttribute('position').array as Float32Array,
        index: (index.array as Uint32Array).subarray(0, count),
      });
    }
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
    Reflect.deleteProperty(this.bodyGeometry, 'setIndex'); // back to the prototype method
    this.assets.mesh.visible = this.bodyWasVisible;
    this.mesh.removeFromParent();
    this.geometry.dispose();
    this.model.dispose();
    if (import.meta.env.DEV && typeof window !== 'undefined')
      delete (window as unknown as { __dtTwin?: unknown }).__dtTwin;
  }
}
