import { describe, expect, it } from 'vitest';
import {
  createGarmentSections,
  gradeGarment,
  type GarmentBinding,
  type GarmentTemplateDef,
  type StoreItemDef,
} from '@dt/avatar-core';
import { buildGradeRings, fitToChart, type BodyPlanes } from './garmentGrading';

/** Cylinder body of radius 0.2 m at y = 1, 64 vertices. */
function cylinderBody(): Float32Array {
  const body = new Float32Array(64 * 3);
  for (let i = 0; i < 64; i++) {
    const angle = (2 * Math.PI * i) / 64;
    body[i * 3] = 0.2 * Math.cos(angle);
    body[i * 3 + 1] = 1;
    body[i * 3 + 2] = 0.2 * Math.sin(angle);
  }
  return body;
}

/** Two stacked vertices 3 mm apart in front of the body, bound to neighbouring body vertices 16 and 17. */
function stackedBinding(body: Float32Array): GarmentBinding {
  const targets: [number, number, number][] = [
    [0, 1, 0.203],
    [0, 1, 0.206],
  ];
  const bodyVerts = [16, 17];
  const indices = new Uint32Array(6);
  const weights = new Float32Array(6);
  const offsets = new Float32Array(6);
  targets.forEach((target, v) => {
    const b = bodyVerts[v]!;
    indices.fill(b, v * 3, v * 3 + 3);
    weights[v * 3] = 1;
    for (let k = 0; k < 3; k++) offsets[v * 3 + k] = target[k]! - body[b * 3 + k]!;
  });
  return { count: 2, indices, weights, offsets };
}

const tee: GarmentTemplateDef = {
  id: 'tee',
  kind: 'tshirt',
  category: 'top',
  label: { tr: 'Tişört', en: 'T-shirt' },
  license: 'CC0-1.0',
  mesh: 'tee.glb',
  nativeMeasures: { chest: 100 },
  defaultEase: {},
  layer: 1,
  baseColor: '#838383',
};
const jacket: GarmentTemplateDef = { ...tee, id: 'jacket', slot: 'outer' };

const item: StoreItemDef = {
  id: 'i',
  name: 'Item',
  templateId: 'tee',
  color: '#ff0000',
  sizes: ['M'],
  chart: { chest: [94] },
  selectedSize: 'M',
};

/** An open cylinder "jacket" covering angles from 0.4pi to 1.6pi around cylinderBody. */
function openCylinderJacket(body: Float32Array): {
  binding: GarmentBinding;
  edgeVerts: number[];
} {
  const heights = [0.96, 0.98, 1.0, 1.02, 1.04];
  const angles: number[] = [];
  for (let a = 0.4; a <= 1.601; a += 0.05) {
    angles.push(a * Math.PI);
  }
  const targets: [number, number, number][] = [];
  const edgeVerts: number[] = [];

  for (const y of heights) {
    for (let ai = 0; ai < angles.length; ai++) {
      const angle = angles[ai]!;
      const x = 0.205 * Math.cos(angle);
      const z = 0.205 * Math.sin(angle);
      const vIdx = targets.length;
      targets.push([x, y, z]);
      if (ai === 0) {
        edgeVerts.push(vIdx);
      }
    }
  }

  const count = targets.length;
  const indices = new Uint32Array(count * 3);
  const weights = new Float32Array(count * 3);
  const offsets = new Float32Array(count * 3);

  for (let v = 0; v < count; v++) {
    const target = targets[v]!;
    let bestB = 0;
    let bestDist = Infinity;
    for (let b = 0; b < 64; b++) {
      const bx = body[b * 3]!;
      const by = body[b * 3 + 1]!;
      const bz = body[b * 3 + 2]!;
      const d = (target[0] - bx) ** 2 + (target[1] - by) ** 2 + (target[2] - bz) ** 2;
      if (d < bestDist) {
        bestDist = d;
        bestB = b;
      }
    }
    indices.fill(bestB, v * 3, v * 3 + 3);
    weights[v * 3] = 1;
    for (let k = 0; k < 3; k++) {
      offsets[v * 3 + k] = target[k]! - body[bestB * 3 + k]!;
    }
  }

  return {
    binding: { count, indices, weights, offsets },
    edgeVerts,
  };
}

describe('fitToChart outer slot grading', () => {
  const body = cylinderBody();
  const sections = createGarmentSections(body);
  const binding = stackedBinding(body);
  const planes: BodyPlanes = { planeY: { chest: 1 } };
  const none = (): undefined => undefined;

  function fit(
    def: GarmentTemplateDef,
    fitItem: StoreItemDef = item,
    fitPlanes: BodyPlanes = planes,
  ): { rest: Float32Array; graded: Float32Array } {
    const rest = new Float32Array(6);
    const graded = new Float32Array(6);
    fitToChart(
      {
        def,
        item: fitItem,
        binding,
        body,
        sections,
        achievedCm: {},
        planes: fitPlanes,
        joint: none,
      },
      rest,
      graded,
    );
    return { rest, graded };
  }

  it('(1) preserves order and spacing within ±1 mm for stacked parts 3 mm apart bound to different body vertices', () => {
    const rings = buildGradeRings(item, jacket, {}, planes);
    expect(rings).toHaveLength(1); // chart 94 vs native 100 -> delta -6 cm

    const outer = fit(jacket);
    // Girth grading was applied
    expect(Array.from(outer.graded)).not.toEqual(Array.from(outer.rest));

    // Both shifted inward
    expect(outer.graded[2]).toBeLessThan(outer.rest[2]!);
    expect(outer.graded[5]).toBeLessThan(outer.rest[5]!);

    // Order preserved: target 1 is still in front of target 0 (greater Z)
    expect(outer.graded[5]!).toBeGreaterThan(outer.graded[2]!);

    // Spacing preserved within ±1 mm of initial 3 mm (2 mm .. 4 mm)
    const dist = Math.hypot(
      outer.graded[3]! - outer.graded[0]!,
      outer.graded[4]! - outer.graded[1]!,
      outer.graded[5]! - outer.graded[2]!,
    );
    expect(dist).toBeGreaterThanOrEqual(0.002);
    expect(dist).toBeLessThanOrEqual(0.004);
    expect(Math.abs(dist - 0.003)).toBeLessThanOrEqual(0.001);
  });

  it('(2) keeps displacement difference between neighboring edge vertices < 1 mm when narrowing an open cylinder jacket', () => {
    const openJacket = openCylinderJacket(body);
    const rest = new Float32Array(openJacket.binding.count * 3);
    const graded = new Float32Array(openJacket.binding.count * 3);
    fitToChart(
      {
        def: jacket,
        item,
        binding: openJacket.binding,
        body,
        sections,
        achievedCm: {},
        planes,
        joint: none,
      },
      rest,
      graded,
    );

    // Verify displacement difference between consecutive vertices along the opening edge is < 1 mm
    const edge = openJacket.edgeVerts;
    expect(edge.length).toBeGreaterThan(1);
    for (let i = 0; i < edge.length - 1; i++) {
      const v0 = edge[i]!;
      const v1 = edge[i + 1]!;
      const u0x = graded[v0 * 3]! - rest[v0 * 3]!;
      const u0y = graded[v0 * 3 + 1]! - rest[v0 * 3 + 1]!;
      const u0z = graded[v0 * 3 + 2]! - rest[v0 * 3 + 2]!;
      const u1x = graded[v1 * 3]! - rest[v1 * 3]!;
      const u1y = graded[v1 * 3 + 1]! - rest[v1 * 3 + 1]!;
      const u1z = graded[v1 * 3 + 2]! - rest[v1 * 3 + 2]!;
      const diff = Math.hypot(u1x - u0x, u1y - u0y, u1z - u0z);
      expect(diff).toBeLessThan(0.001); // < 1 mm
    }
  });

  it('(3) leaves positions identical to rest when delta is 0', () => {
    const itemDeltaZero: StoreItemDef = {
      ...item,
      chart: { chest: [100] }, // native is 100 -> delta = 0
    };
    const outer = fit(jacket, itemDeltaZero);
    expect(Array.from(outer.graded)).toEqual(Array.from(outer.rest));
  });

  it('(4) produces identical result to standard gradeGarment for non-outer garments', () => {
    const rings = buildGradeRings(item, tee, {}, planes);
    expect(rings).toHaveLength(1);

    const plain = fit(tee);
    expect(Array.from(plain.graded)).not.toEqual(Array.from(plain.rest));

    const expected = new Float32Array(6);
    gradeGarment(plain.rest, binding, body, rings, expected, sections);
    expect(Array.from(plain.graded)).toEqual(Array.from(expected));
  });

  it('still applies length grading to outer garments when the chart length differs', () => {
    const jacketWithLength: GarmentTemplateDef = {
      ...jacket,
      nativeMeasures: { ...jacket.nativeMeasures, length: 60 },
    };
    const itemWithLength: StoreItemDef = {
      ...item,
      chart: { length: [70] },
    };
    const planesWithWaist: BodyPlanes = {
      planeY: { waist: 1.2, chest: 1.4 },
    };

    const outer = fit(jacketWithLength, itemWithLength, planesWithWaist);
    expect(Array.from(outer.graded)).not.toEqual(Array.from(outer.rest));
    expect(outer.graded[1]).toBeCloseTo(0.9, 5);
    expect(outer.graded[4]).toBeCloseTo(0.9, 5);
    // Girth was not specified in chart: x and z positions remain identical to rest
    expect(outer.graded[0]).toBe(outer.rest[0]);
    expect(outer.graded[2]).toBe(outer.rest[2]);
    expect(outer.graded[3]).toBe(outer.rest[3]);
    expect(outer.graded[5]).toBe(outer.rest[5]);
  });
});
