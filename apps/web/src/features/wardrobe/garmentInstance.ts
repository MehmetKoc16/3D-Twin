import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DynamicDrawUsage,
  Float32BufferAttribute,
  MeshStandardMaterial,
  SkinnedMesh,
  type Matrix4,
  type Skeleton,
} from 'three';
import {
  clearanceToColor,
  computeScale,
  fitGarmentRigidAttachments,
  garmentRigidSkinWeights,
  garmentClearance,
  garmentLayerOrder,
  garmentNormalOffset,
  garmentSlot,
  offsetAlongNormals,
  prepareGarmentRigidAttachments,
  type PreparedGarmentRigidAttachments,
  type GarmentSections,
  type Vec3,
  type GarmentTemplateDef,
  type MeasureId,
  type StoreItemDef,
} from '@dt/avatar-core';
import { computeWeldedNormals } from '../avatar/meshMath';
import { clipCoverageToHem, clipDeleteVerts, coveredBodyVertices } from './bodyHide';
import { OUTER_PUSH_MARGIN_M, pushInside, pushOutside, type Surface } from './garmentCollide';
import { flareOverShoes } from './garmentFlare';
import { boundBodyPoints, fitToChart, type BodyPlanes } from './garmentGrading';
import type { TemplateRuntime } from './templateLoader';

/** Largest recolour factor of the (grey) template texture, so near-black textures do not blow up. */
const MAX_RECOLOR_FACTOR = 12;

export interface GarmentUpdateContext {
  /** Grounded solved render positions (3 floats per render vertex). */
  body: Float32Array;
  joints: Float32Array;
  sections: GarmentSections;
  achievedCm: Partial<Record<MeasureId, number>>;
  planes: BodyPlanes | undefined;
  heatmap: boolean;
  /** Inner garments this one is worn over (shoes under trousers, trousers under a jumper). */
  lowers: readonly Surface[];
  /** Optional shoe surface worn under trousers: flared radially over the shoes instead of pushOutside. */
  shoes?: Surface;
  /** Optional inner top surface worn under an outer garment (e.g. polo or t-shirt under a denim jacket). */
  innerTop?: Surface;
  /** A tucked top worn with a bottom: the chart's hem length is not applied (the hem stays inside the trousers). */
  tuckedHem?: boolean;
  /** Absolute normal expansion, ordered above all inner garments by the rig. */
  normalOffsetM?: number;
}

/**
 * Clearance the outer layer (e.g. denim jacket) keeps outside of the inner garments it is worn over (metres).
 * Outer garments need a larger clearance (12 mm vs default 6 mm) over inner garments (such as polo shirts)
 * to prevent inner fabric patches from showing through near welt pockets and side seams, while keeping
 * lighter outfits clean.
 */
export { OUTER_PUSH_MARGIN_M };

/**
 * Multiplier that turns the template's texture into the item colour: the texture carries the fabric detail around
 * the template's `baseColor`, so the material colour is `item / base` per linear channel (1 at the default colour).
 */
export function recolorFactor(itemHex: string, baseHex: string): [number, number, number] {
  const item = new Color(itemHex);
  const base = new Color(baseHex);
  const factor = (i: number, b: number): number =>
    Math.min(MAX_RECOLOR_FACTOR, i / Math.max(b, 0.02));
  return [factor(item.r, base.r), factor(item.g, base.g), factor(item.b, base.b)];
}

/** One worn garment: a SkinnedMesh sharing the avatar skeleton, re-fitted to the body after every solve. */
export class GarmentInstance {
  readonly mesh: SkinnedMesh;
  private readonly geometry = new BufferGeometry();
  private readonly material: MeshStandardMaterial;
  private readonly heatMaterial: MeshStandardMaterial;
  private readonly rest: Float32Array;
  private readonly graded: Float32Array;
  private readonly clearance: Float32Array;
  private readonly colors: Float32Array;
  private readonly normalScratch: Float32Array;
  private readonly rgb = new Float32Array(3);
  private readonly bound: Float32Array;
  private readonly indexArray: Uint32Array;
  private readonly indexAttribute: BufferAttribute;
  private readonly rigid: PreparedGarmentRigidAttachments | null;
  private hiddenByOuter = false;
  private heat = false;
  /** How far the lowest vertex of the graded garment is below the floor (m, >= 0); the avatar is lifted by it. */
  soleLift = 0;
  /** Static delete list clipped to the graded garment's lower edge; defaults to the template's full list. */
  deleteVerts: Uint32Array;
  /** Body vertices under this garment's footprint (one byte per render vertex), computed once at the first fit. */
  coverage: Uint8Array | null = null;
  /** Incremented whenever `coverage` is recomputed, so the body index is rebuilt. */
  coverageEpoch = 0;
  private hemTucked = false;
  private coveredBody: Float32Array | null = null;
  private coveredHemShift = 0;
  item: StoreItemDef;

  constructor(
    readonly runtime: TemplateRuntime,
    item: StoreItemDef,
    skeleton: Skeleton,
    bindMatrix: Matrix4,
  ) {
    const def = runtime.def;
    const n = runtime.vertexCount;
    this.item = item;
    this.deleteVerts = runtime.deleteVerts;
    this.rest = new Float32Array(n * 3);
    this.graded = new Float32Array(n * 3);
    this.clearance = new Float32Array(n);
    this.normalScratch = new Float32Array(runtime.weld.groupCount * 3);
    this.bound = new Float32Array(n * 3);
    this.rigid = runtime.rigidAttachments
      ? prepareGarmentRigidAttachments(
          runtime.rigidAttachments,
          runtime.neutralPositions,
          runtime.index,
          skeleton.bones.map((bone) => bone.name),
        )
      : null;
    const skin = this.rigid
      ? garmentRigidSkinWeights(this.rigid, runtime.skinIndices, runtime.skinWeights)
      : runtime;

    this.geometry.setAttribute(
      'position',
      new Float32BufferAttribute(new Float32Array(n * 3), 3).setUsage(DynamicDrawUsage),
    );
    this.geometry.setAttribute(
      'normal',
      new Float32BufferAttribute(new Float32Array(n * 3), 3).setUsage(DynamicDrawUsage),
    );
    this.geometry.setAttribute('uv', new Float32BufferAttribute(runtime.uv, 2));
    const colorAttribute = new Float32BufferAttribute(new Float32Array(n * 3), 3).setUsage(
      DynamicDrawUsage,
    );
    this.colors = colorAttribute.array as Float32Array; // the attribute copies its input: write into its own array
    this.geometry.setAttribute('color', colorAttribute);
    this.geometry.setAttribute('skinIndex', new BufferAttribute(skin.skinIndices, 4));
    this.geometry.setAttribute('skinWeight', new BufferAttribute(skin.skinWeights, 4));
    this.indexArray = Uint32Array.from(runtime.index);
    this.indexAttribute = new BufferAttribute(this.indexArray, 1).setUsage(DynamicDrawUsage);
    this.geometry.setIndex(this.indexAttribute);

    // Higher layers win depth ties against lower ones and against the body (which has no offset).
    const layer = garmentLayerOrder(def);
    this.material = runtime.material.clone();
    // Keep fabric matte but allow a broad studio reflection to describe folds.
    this.material.roughness = Math.max(0.65, Math.min(0.85, this.material.roughness));
    this.material.envMapIntensity = 0.65;
    this.material.polygonOffset = true;
    this.material.polygonOffsetFactor = -(1 + layer * 0.5);
    this.material.polygonOffsetUnits = -(1 + layer);
    this.heatMaterial = new MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.9,
      metalness: 0,
      side: this.material.side,
      polygonOffset: true,
      polygonOffsetFactor: this.material.polygonOffsetFactor,
      polygonOffsetUnits: this.material.polygonOffsetUnits,
    });
    this.applyColor();

    this.mesh = new SkinnedMesh(this.geometry, this.material);
    this.mesh.name = `garment:${def.id}`;
    this.mesh.frustumCulled = false; // bounds of a posed skinned mesh are not tracked
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.mesh.bind(skeleton, bindMatrix);
  }

  get template(): GarmentTemplateDef {
    return this.runtime.def;
  }

  /** Swaps the item definition (size, chart or colour changed); the caller re-runs `update`. */
  setItem(item: StoreItemDef): void {
    if (item.selectedSize !== this.item.selectedSize || item.chart !== this.item.chart) {
      this.coverage = null;
      this.deleteVerts = this.runtime.deleteVerts;
    }
    this.item = item;
    this.applyColor();
  }

  private applyColor(): void {
    const [r, g, b] = this.runtime.material.map
      ? recolorFactor(this.item.color, this.runtime.def.baseColor)
      : (() => {
          const c = new Color(this.item.color);
          return [c.r, c.g, c.b] as const;
        })();
    this.material.color.setRGB(r, g, b);
  }

  /** Follows the body rest bind matrix after the skeleton was rebuilt. */
  syncBind(bindMatrix: Matrix4): void {
    this.mesh.bindMatrix.copy(bindMatrix);
    this.mesh.bindMatrixInverse.copy(bindMatrix).invert();
  }

  /**
   * Re-fits the garment to the solved body: proxy binding, size grading, layering, normals, optional clearance
   * colours. `tuckInto` is the outer layer a tucked top disappears into (the trousers that were fitted after `fit`).
   */
  update(ctx: GarmentUpdateContext, tuckInto?: Surface): void {
    this.fit(ctx);
    this.commit(ctx, tuckInto);
  }

  /** First half of `update`: the graded and layered vertex positions (nothing is uploaded yet). */
  fit(ctx: GarmentUpdateContext): void {
    const { runtime, item, rest, graded } = this;
    const def = runtime.def;
    const restJoint = (index: number): Vec3 | undefined => {
      if (index < 0 || index * 6 + 2 >= ctx.joints.length) return undefined;
      return [ctx.joints[index * 6]!, ctx.joints[index * 6 + 1]!, ctx.joints[index * 6 + 2]!];
    };
    const joint = (name: string): Vec3 | undefined => {
      const index = this.mesh.skeleton.bones.findIndex((bone) => bone.name === name);
      return restJoint(index);
    };
    const isOuter = garmentSlot(def) === 'outer';
    let innerTop = ctx.innerTop;
    if (!innerTop && isOuter && ctx.lowers.length > 0) {
      // Find the inner top surface from lowers (e.g. polo shirt or t-shirt worn under jacket).
      // A top has vertices in the upper torso / chest (y > jacketMaxY - 0.25).
      let jacketMaxY = -Infinity;
      for (let i = 1; i < runtime.vertexCount * 3; i += 3) {
        const y = runtime.neutralPositions[i]!;
        if (y > jacketMaxY) jacketMaxY = y;
      }
      for (const surface of ctx.lowers) {
        let maxSy = -Infinity;
        for (let i = 1; i < surface.positions.length; i += 3) {
          const sy = surface.positions[i]!;
          if (sy > maxSy) maxSy = sy;
        }
        if (maxSy > jacketMaxY - 0.25) {
          innerTop = surface;
          break;
        }
      }
    }
    fitToChart(
      {
        def,
        item,
        binding: runtime.binding,
        body: ctx.body,
        sections: ctx.sections,
        achievedCm: ctx.achievedCm,
        planes: ctx.planes,
        joint,
        tuckedHem: ctx.tuckedHem ?? false,
        innerSurface: isOuter ? innerTop : undefined,
      },
      rest,
      graded,
    );
    // Grading/clearance may bend a sole. Restore its scaled local shape before uploading to the shared skeleton.
    if (this.rigid)
      fitGarmentRigidAttachments(
        this.rigid,
        computeScale(def.scaleRefs, ctx.body),
        restJoint,
        graded,
      );
    this.updateBound(ctx.body);
    const offset = ctx.normalOffsetM ?? garmentNormalOffset(def);
    if (offset > 0) {
      const normals = this.geometry.getAttribute('normal').array as Float32Array;
      computeWeldedNormals(graded, runtime.index, runtime.weld, normals, this.normalScratch);
      const outward = new Float32Array(runtime.vertexCount * 3);
      for (let i = 0; i < outward.length; i++) outward[i] = graded[i]! - this.bound[i]!;
      offsetAlongNormals(graded, normals, offset, { outward });
    }
    const tucked = ctx.tuckedHem ?? false;
    let restMinY = Infinity;
    let gradedMinY = Infinity;
    for (let i = 1; i < graded.length; i += 3) {
      if (rest[i]! < restMinY) restMinY = rest[i]!;
      if (graded[i]! < gradedMinY) gradedMinY = graded[i]!;
    }
    const hemShift =
      Number.isFinite(gradedMinY) && Number.isFinite(restMinY) ? gradedMinY - restMinY : 0;
    if (
      tucked !== this.hemTucked ||
      this.coveredBody !== ctx.body ||
      Math.abs(hemShift - this.coveredHemShift) > 1e-5
    ) {
      // the hem or body moved (chart length applied, skipped or re-graded): the old footprint no longer matches
      this.hemTucked = tucked;
      this.coverage = null;
    }
    if (this.coverage === null) {
      const rawCoverage = coveredBodyVertices(
        ctx.body,
        ctx.body.length / 3,
        graded,
        runtime.index,
        { slack: garmentSlot(def) === 'outer' ? 0 : undefined },
      );
      this.coverage =
        def.category === 'shoes'
          ? rawCoverage
          : clipCoverageToHem(rawCoverage, ctx.body, graded, { rest });
      this.deleteVerts =
        def.category === 'shoes' || runtime.deleteVerts.length === 0
          ? runtime.deleteVerts
          : clipDeleteVerts(runtime.deleteVerts, ctx.body, graded, { rest });
      this.coveredBody = ctx.body;
      this.coveredHemShift = hemShift;
      this.coverageEpoch++;
    }
    if (garmentSlot(def) === 'bottom' && ctx.shoes) {
      flareOverShoes(graded, runtime.vertexCount, ctx.shoes);
    }
    if (ctx.lowers.length > 0) {
      const isOuter = garmentSlot(def) === 'outer';
      pushOutside(graded, runtime.vertexCount, ctx.lowers, {
        orientation: isOuter ? 'surface' : 'triangle',
        marginM: isOuter ? OUTER_PUSH_MARGIN_M : undefined,
        frontDepthOnly: isOuter,
      });
    }
  }

  /** Second half of `update`: optionally tucks the hem into the outer layer, then uploads positions and normals. */
  commit(ctx: GarmentUpdateContext, tuckInto?: Surface): void {
    const { runtime, graded } = this;
    const def = runtime.def;
    if (tuckInto) pushInside(graded, runtime.vertexCount, [tuckInto]);

    const position = this.geometry.getAttribute('position');
    (position.array as Float32Array).set(graded);
    position.needsUpdate = true;
    if (def.category === 'shoes') {
      let lowest = Infinity;
      for (let v = 1; v < graded.length; v += 3) if (graded[v]! < lowest) lowest = graded[v]!;
      this.soleLift = Math.max(0, -lowest);
    }
    const normal = this.geometry.getAttribute('normal');
    computeWeldedNormals(
      graded,
      runtime.index,
      runtime.weld,
      normal.array as Float32Array,
      this.normalScratch,
    );
    normal.needsUpdate = true;

    this.setHeatmap(ctx.heatmap);
    if (ctx.heatmap) {
      garmentClearance(graded, runtime.binding, ctx.body, this.clearance, ctx.sections);
      const rgb = this.rgb;
      for (let v = 0; v < runtime.vertexCount; v++) {
        clearanceToColor(this.clearance[v]!, rgb);
        this.colors[v * 3] = rgb[0]!;
        this.colors[v * 3 + 1] = rgb[1]!;
        this.colors[v * 3 + 2] = rgb[2]!;
      }
      this.geometry.getAttribute('color').needsUpdate = true;
    }
  }

  /** The body point every garment vertex is bound to (its offset from it points away from the body). */
  private updateBound(body: Float32Array): void {
    boundBodyPoints(this.runtime.binding, body, this.bound);
  }

  /**
   * Drops the triangles of this (inner) garment that lie inside the footprint of the outer garment, so that trousers
   * do not poke through a jumper hem between its sparse vertices, or shoe ankles do not poke through a trouser cuff.
   * `null` restores every triangle.
   */
  hideUnder(
    outer: Surface | readonly Surface[] | null,
    options?: {
      bandM?: number;
      slack?: number;
      extraMask?: ArrayLike<number>;
      extraTriangleMask?: ArrayLike<number>;
    },
  ): void {
    if (garmentSlot(this.runtime.def) === 'outer') {
      if (!this.hiddenByOuter) return;
      this.indexArray.set(this.runtime.index);
      this.geometry.setDrawRange(0, this.runtime.index.length);
      this.indexAttribute.needsUpdate = true;
      this.hiddenByOuter = false;
      return;
    }
    const surfaces: readonly Surface[] =
      outer === null ? [] : 'positions' in outer ? [outer] : outer;
    const extraMask = options?.extraMask;
    const extraTriangleMask = options?.extraTriangleMask;
    if (surfaces.length === 0 && !extraMask && !extraTriangleMask) {
      if (!this.hiddenByOuter) return;
      this.indexArray.set(this.runtime.index);
      this.geometry.setDrawRange(0, this.runtime.index.length);
      this.indexAttribute.needsUpdate = true;
      this.hiddenByOuter = false;
      return;
    }
    const bandM = options?.bandM ?? 0.02;
    const slack = options?.slack ?? 0;
    const covered = new Uint8Array(this.runtime.vertexCount);
    if (extraMask) {
      const len = Math.min(covered.length, extraMask.length);
      for (let v = 0; v < len; v++) {
        if (extraMask[v]) covered[v] = 1;
      }
    }
    for (const surface of surfaces) {
      const mask = coveredBodyVertices(
        this.graded,
        this.runtime.vertexCount,
        surface.positions,
        surface.index,
        { bandM, slack },
      );
      for (let v = 0; v < covered.length; v++) covered[v] = covered[v]! | mask[v]!;
    }
    const source = this.runtime.index;
    let kept = 0;
    for (let t = 0; t < source.length; t += 3) {
      const a = source[t]!;
      const b = source[t + 1]!;
      const c = source[t + 2]!;
      const triIdx = t / 3;
      if (extraTriangleMask && extraTriangleMask[triIdx]) continue;
      if (covered[a] === 1 && covered[b] === 1 && covered[c] === 1) continue;
      this.indexArray[kept++] = a;
      this.indexArray[kept++] = b;
      this.indexArray[kept++] = c;
    }
    this.geometry.setDrawRange(0, kept);
    this.indexAttribute.needsUpdate = true;
    this.hiddenByOuter = true;
  }

  /** This garment's graded surface, for outer garments that must stay outside of it. */
  surface(): Surface {
    return { positions: this.graded, index: this.runtime.index, bound: this.bound };
  }

  private setHeatmap(on: boolean): void {
    if (on === this.heat) return;
    this.heat = on;
    this.mesh.material = on ? this.heatMaterial : this.material;
  }

  dispose(): void {
    this.mesh.removeFromParent();
    this.geometry.dispose();
    this.material.dispose(); // the texture belongs to the template runtime and stays
    this.heatMaterial.dispose();
  }
}
