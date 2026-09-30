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
  garmentClearance,
  type GarmentSections,
  type Vec3,
  type GarmentTemplateDef,
  type MeasureId,
  type StoreItemDef,
} from '@dt/avatar-core';
import { computeWeldedNormals } from '../avatar/meshMath';
import { coveredBodyVertices } from './bodyHide';
import { pushInside, pushOutside, type Surface } from './garmentCollide';
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
  /** A tucked top worn with a bottom: the chart's hem length is not applied (the hem stays inside the trousers). */
  tuckedHem?: boolean;
}

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
  private hiddenByOuter = false;
  private heat = false;
  /** How far the lowest vertex of the graded garment is below the floor (m, >= 0); the avatar is lifted by it. */
  soleLift = 0;
  /** Body vertices under this garment's footprint (one byte per render vertex), computed once at the first fit. */
  coverage: Uint8Array | null = null;
  /** Incremented whenever `coverage` is recomputed, so the body index is rebuilt. */
  coverageEpoch = 0;
  private hemTucked = false;
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
    this.rest = new Float32Array(n * 3);
    this.graded = new Float32Array(n * 3);
    this.clearance = new Float32Array(n);
    this.normalScratch = new Float32Array(runtime.weld.groupCount * 3);
    this.bound = new Float32Array(n * 3);

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
    this.geometry.setAttribute('skinIndex', new BufferAttribute(runtime.skinIndices, 4));
    this.geometry.setAttribute('skinWeight', new BufferAttribute(runtime.skinWeights, 4));
    this.indexArray = Uint32Array.from(runtime.index);
    this.indexAttribute = new BufferAttribute(this.indexArray, 1).setUsage(DynamicDrawUsage);
    this.geometry.setIndex(this.indexAttribute);

    // Higher layers win depth ties against lower ones and against the body (which has no offset).
    const layer = def.layer + (def.category === 'top' ? 0.25 : 0);
    this.material = runtime.material.clone();
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
    if (item.selectedSize !== this.item.selectedSize || item.chart !== this.item.chart)
      this.coverage = null;
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
    const joint = (name: string): Vec3 | undefined => {
      const index = this.mesh.skeleton.bones.findIndex((bone) => bone.name === name);
      if (index < 0 || index * 6 + 2 >= ctx.joints.length) return undefined;
      return [ctx.joints[index * 6]!, ctx.joints[index * 6 + 1]!, ctx.joints[index * 6 + 2]!];
    };
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
      },
      rest,
      graded,
    );
    const tucked = ctx.tuckedHem ?? false;
    if (tucked !== this.hemTucked) {
      // the hem moved (chart length applied or skipped): the body under the old footprint no longer matches
      this.hemTucked = tucked;
      this.coverage = null;
    }
    if (this.coverage === null) {
      this.coverage = coveredBodyVertices(ctx.body, ctx.body.length / 3, graded, runtime.index);
      this.coverageEpoch++;
    }
    this.updateBound(ctx.body);
    if (ctx.lowers.length > 0) pushOutside(graded, runtime.vertexCount, ctx.lowers);
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
   * do not poke through a jumper hem between its sparse vertices. `null` restores every triangle.
   */
  hideUnder(outer: Surface | null): void {
    if (outer === null) {
      if (!this.hiddenByOuter) return;
      this.indexArray.set(this.runtime.index);
      this.geometry.setDrawRange(0, this.runtime.index.length);
      this.indexAttribute.needsUpdate = true;
      this.hiddenByOuter = false;
      return;
    }
    const covered = coveredBodyVertices(
      this.graded,
      this.runtime.vertexCount,
      outer.positions,
      outer.index,
      { bandM: 0.02 },
    );
    const source = this.runtime.index;
    let kept = 0;
    for (let t = 0; t < source.length; t += 3) {
      const a = source[t]!;
      const b = source[t + 1]!;
      const c = source[t + 2]!;
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
