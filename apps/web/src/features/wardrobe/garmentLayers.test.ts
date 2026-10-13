import { describe, expect, it } from 'vitest';
import { Bone, Matrix4, MeshStandardMaterial, Skeleton } from 'three';
import {
  createGarmentSections,
  garmentNormalOffset,
  offsetAlongNormals,
  type GarmentTemplateDef,
  type StoreItemDef,
} from '@dt/avatar-core';
import { buildWeldMap } from '../avatar/meshMath';
import { GarmentInstance, type GarmentUpdateContext } from './garmentInstance';
import type { Surface } from './garmentCollide';
import type { TemplateRuntime } from './templateLoader';

// Three independent patches: covered, partially covered, and visible inside an open jacket.
const points = new Float32Array([
  0.2, 1, 0.1, 0.3, 1, 0.1, 0.2, 1.1, 0.1, 0.2, 0.8, 0.1, 0.3, 0.8, 0.1, 0.05, 0.9, 0.1, -0.05, 1,
  0.1, 0.05, 1, 0.1, 0, 1.1, 0.1,
]);
const index = Uint32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8]);

function fixture(layer = 2) {
  const def: GarmentTemplateDef = {
    id: 'test',
    kind: 'tshirt',
    category: 'top',
    label: { tr: 'Test', en: 'Test' },
    license: 'CC0-1.0',
    mesh: 'test.glb',
    layer,
    baseColor: '#888888',
    nativeMeasures: {},
    defaultEase: {},
  };
  const item: StoreItemDef = {
    id: 'item',
    name: 'Test',
    templateId: def.id,
    color: '#888888',
    sizes: ['M'],
    selectedSize: 'M',
    chart: {},
  };
  const material = new MeshStandardMaterial();
  const weights = new Float32Array(27);
  const indices = new Uint32Array(27);
  const skinWeights = new Float32Array(36);
  for (let v = 0; v < 9; v++) {
    weights[v * 3] = 1;
    indices.fill(v, v * 3, v * 3 + 3);
    skinWeights[v * 4] = 1;
  }
  const runtime: TemplateRuntime = {
    def,
    material,
    vertexCount: 9,
    index,
    uv: new Float32Array(18),
    neutralPositions: points,
    binding: { count: 9, weights, indices, offsets: new Float32Array(27) },
    skinIndices: new Uint16Array(36),
    skinWeights,
    weld: buildWeldMap(points),
    deleteVerts: new Uint32Array(),
    rigidAttachments: null,
  };
  const instance = new GarmentInstance(runtime, item, new Skeleton([new Bone()]), new Matrix4());
  (instance.surface().positions as Float32Array).set(points);
  const ctx: GarmentUpdateContext = {
    body: points,
    joints: new Float32Array(6),
    sections: createGarmentSections(points),
    achievedCm: {},
    planes: undefined,
    heatmap: false,
    lowers: [],
  };
  return { instance, material, ctx };
}

function panel(x0: number, x1: number, y0: number, y1: number, z = 0.11): Surface {
  return {
    positions: new Float32Array([x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z]),
    index: [0, 1, 2, 0, 2, 3],
    bound: new Float32Array([
      x0,
      y0,
      z - 0.01,
      x1,
      y0,
      z - 0.01,
      x1,
      y1,
      z - 0.01,
      x0,
      y1,
      z - 0.01,
    ]),
  };
}

function drawn(instance: GarmentInstance): number[] {
  const geometry = instance.mesh.geometry;
  return Array.from(geometry.getIndex()!.array).slice(0, geometry.drawRange.count);
}

describe('garment-on-garment layers', () => {
  it('hides fully covered triangles, preserves partially covered triangles and the open front, and restores removal', () => {
    const { instance, material } = fixture();
    try {
      instance.hideUnder(panel(0.15, 0.35, 0.75, 1.15));
      expect(drawn(instance)).toEqual([3, 4, 5, 6, 7, 8]);
      instance.hideUnder(null);
      expect(drawn(instance)).toEqual(Array.from(index));
      instance.hideUnder([]);
      expect(drawn(instance)).toEqual(Array.from(index));
    } finally {
      instance.dispose();
      material.dispose();
    }
  });

  it('unions multiple outer surfaces and rebuilds from the full index when an outer layer changes', () => {
    const { instance, material } = fixture();
    try {
      const jacket = panel(0.15, 0.35, 0.95, 1.15);
      const waistband = panel(-0.1, 0.35, 0.75, 0.92);
      instance.hideUnder([jacket, waistband]);
      expect(drawn(instance)).toEqual([6, 7, 8]);
      instance.hideUnder(jacket);
      expect(drawn(instance)).toEqual([3, 4, 5, 6, 7, 8]);
    } finally {
      instance.dispose();
      material.dispose();
    }
  });

  it('orders physical and depth offsets and repeats fits without accumulating expansion', () => {
    const inner = fixture(2);
    const outer = fixture(3);
    try {
      const innerOffset = garmentNormalOffset(inner.instance.template);
      const outerOffset = garmentNormalOffset(outer.instance.template, [innerOffset]);
      inner.instance.update({ ...inner.ctx, normalOffsetM: innerOffset });
      outer.instance.update({
        ...outer.ctx,
        normalOffsetM: outerOffset,
        lowers: [inner.instance.surface()],
      });
      const innerPositions = inner.instance.surface().positions;
      const outerPositions = Float32Array.from(outer.instance.surface().positions);
      for (let v = 0; v < 9; v++)
        expect(outerPositions[v * 3 + 2]! - innerPositions[v * 3 + 2]!).toBeGreaterThan(0.0059);
      outer.instance.update({
        ...outer.ctx,
        normalOffsetM: outerOffset,
        lowers: [inner.instance.surface()],
      });
      expect(Array.from(outer.instance.surface().positions)).toEqual(Array.from(outerPositions));
      expect(
        (outer.instance.mesh.material as MeshStandardMaterial).polygonOffsetUnits,
      ).toBeLessThan((inner.instance.mesh.material as MeshStandardMaterial).polygonOffsetUnits);
    } finally {
      inner.instance.dispose();
      inner.material.dispose();
      outer.instance.dispose();
      outer.material.dispose();
    }
  });

  it('gives a garment worn alone zero offset identical to pre-3e9fcf1 path', () => {
    const { instance, material, ctx } = fixture(3);
    try {
      expect(garmentNormalOffset(instance.template)).toBe(0);
      instance.update(ctx);
      const alonePositions = Array.from(instance.surface().positions);
      // Repeated updates must not accumulate offsets
      instance.update(ctx);
      expect(Array.from(instance.surface().positions)).toEqual(alonePositions);
      instance.update({ ...ctx, normalOffsetM: 0 });
      expect(Array.from(instance.surface().positions)).toEqual(alonePositions);
    } finally {
      instance.dispose();
      material.dispose();
    }
  });

  it('does not split a thin two-sided flap under normal expansion', () => {
    const flapPositions = new Float32Array([
      0.2, 1.0, 0.1,
      0.3, 1.0, 0.1,
      0.2, 1.1, 0.1,
      0.2, 1.0, 0.098,
      0.3, 1.0, 0.098,
      0.2, 1.1, 0.098,
    ]);
    const flapNormals = [
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
      0, 0, -1,
      0, 0, -1,
      0, 0, -1,
    ];
    offsetAlongNormals(flapPositions, flapNormals, 0.003);
    for (let v = 0; v < 3; v++) {
      expect(flapPositions[v * 3 + 2]! - flapPositions[(v + 3) * 3 + 2]!).toBeCloseTo(0.002);
    }
  });

  it('an open-front outer over a top does not hide inner triangles in the opening', () => {
    const { instance, material } = fixture(2);
    try {
      // Left and right panels leaving open front in the center (x = -0.10..0.10)
      const leftPanel = panel(0.12, 0.35, 0.75, 1.15);
      const rightPanel = panel(-0.35, -0.12, 0.75, 1.15);
      instance.hideUnder([leftPanel, rightPanel]);
      // Triangle [6, 7, 8] is centered in the open front (x = -0.05..0.05) and must NOT be hidden
      const visible = drawn(instance);
      expect(visible).toContain(6);
      expect(visible).toContain(7);
      expect(visible).toContain(8);
      // Triangle [0, 1, 2] is fully covered by left panel and must be hidden
      expect(visible).not.toContain(0);
      expect(visible).not.toContain(1);
      expect(visible).not.toContain(2);
    } finally {
      instance.dispose();
      material.dispose();
    }
  });

  it('never hides outerwear garment triangles', () => {
    const { instance, material } = fixture(3);
    try {
      instance.hideUnder(panel(0.15, 0.35, 0.75, 1.15));
      expect(drawn(instance)).toEqual(Array.from(index));
    } finally {
      instance.dispose();
      material.dispose();
    }
  });
});
