import { describe, expect, it } from 'vitest';
import {
  garmentClearance,
  gradeGarment,
  type GarmentBinding,
  type GarmentTemplateDef,
  type MeasuresDef,
  type StoreItemDef,
} from '@dt/avatar-core';
import {
  bodyPlanes,
  buildGradeRings,
  clearanceLegAware,
  gradeGarmentLegAware,
  SplitScratch,
} from './garmentGrading';

const template: GarmentTemplateDef = {
  id: 'tshirt',
  kind: 'tshirt',
  category: 'top',
  label: { tr: 'Tişört', en: 'T-shirt' },
  license: 'CC0-1.0',
  mesh: 'tshirt.glb',
  nativeMeasures: { chest: 90, waist: 76 },
  defaultEase: { chest: 4, waist: 6 },
  layer: 1,
  baseColor: '#838383',
};

const item: StoreItemDef = {
  id: 'i',
  name: 'Tee',
  templateId: 'tshirt',
  color: '#ff0000',
  sizes: ['S', 'M', 'L'],
  chart: { chest: [92, 100, 108], waist: [84, 92, 100] },
  selectedSize: 'M',
};

describe('bodyPlanes', () => {
  it('takes the mean height of the loop vertices and the crotch height', () => {
    // vertices 0..3 at y 1.2, 1.4 (chest ring), 4..5 at y 0.9 (waist ring), vertex 6 crotch at y 0.8
    const positions = [0, 1.2, 0, 0, 1.4, 0, 0, 1.2, 0, 0, 1.4, 0, 0, 0.9, 0, 0, 0.9, 0, 0, 0.8, 0];
    const measures: MeasuresDef = {
      version: 1,
      measures: [
        { id: 'chest', type: 'circumference', verts: [0, 1, 2, 3], drivers: [] },
        { id: 'waist', type: 'circumference', verts: [4, 5], drivers: [] },
        { id: 'neck', type: 'circumference', verts: [0, 1], drivers: [] },
        { id: 'inseam', type: 'vertexHeight', vert: 6, drivers: [] },
      ],
    };
    const planes = bodyPlanes(measures, positions);
    expect(planes.planeY.chest).toBeCloseTo(1.3, 6);
    expect(planes.planeY.waist).toBeCloseTo(0.9, 6);
    expect(planes.planeY.hip).toBeUndefined();
    expect(planes.crotchY).toBeCloseTo(0.8, 6);
  });
});

describe('buildGradeRings', () => {
  const planes = { planeY: { chest: 1.3, waist: 1.0 } };

  it('grades by chart minus (body + template ease), so the template ease is not counted twice', () => {
    // body chest 96, native ease 4 -> the bound garment already is 100 cm; chart M = 100 -> no extra delta
    const rings = buildGradeRings(item, template, { chest: 96, waist: 80 }, planes);
    expect(rings).toHaveLength(2);
    expect(rings.find((r) => r.planeY === 1.3)!.deltaCircumferenceM).toBeCloseTo(0, 6);
    // waist: 80 + 6 = 86 currently, chart 92 -> +6 cm
    expect(rings.find((r) => r.planeY === 1.0)!.deltaCircumferenceM).toBeCloseTo(0.06, 6);
  });

  it('follows the selected size', () => {
    const large = buildGradeRings({ ...item, selectedSize: 'L' }, template, { chest: 96, waist: 80 }, planes);
    expect(large.find((r) => r.planeY === 1.3)!.deltaCircumferenceM).toBeCloseTo(0.08, 6);
    const small = buildGradeRings({ ...item, selectedSize: 'S' }, template, { chest: 96, waist: 80 }, planes);
    expect(small.find((r) => r.planeY === 1.3)!.deltaCircumferenceM).toBeCloseTo(-0.08, 6);
  });

  it('falls back to the native measurement without a body value, and skips rows without plane or chart', () => {
    const noBody = buildGradeRings(item, template, {}, planes);
    expect(noBody.find((r) => r.planeY === 1.3)!.deltaCircumferenceM).toBeCloseTo(0.1, 6); // 100 - native 90
    expect(buildGradeRings(item, template, { chest: 96 }, { planeY: {} })).toEqual([]);
    expect(buildGradeRings({ ...item, chart: { length: [60, 62, 64] } }, template, { chest: 96 }, planes)).toEqual([]);
    expect(buildGradeRings({ ...item, selectedSize: 'XXL' }, template, { chest: 96 }, planes)).toEqual([]);
  });
});

/** Two vertical "legs" (rings of 8 vertices, radius 0.08) at x = +-0.1, garment 1 cm outside of each. */
function twoLegs(): { body: Float32Array; rest: Float32Array; binding: GarmentBinding } {
  const levels = 10;
  const around = 8;
  const count = 2 * levels * around;
  const body = new Float32Array(count * 3);
  const rest = new Float32Array(count * 3);
  const indices = new Uint32Array(count * 3);
  const weights = new Float32Array(count * 3);
  const offsets = new Float32Array(count * 3);
  let n = 0;
  for (const cx of [0.1, -0.1]) {
    for (let level = 0; level < levels; level++) {
      for (let k = 0; k < around; k++) {
        const angle = (k / around) * Math.PI * 2;
        const dx = Math.cos(angle);
        const dz = Math.sin(angle);
        body.set([cx + 0.08 * dx, level * 0.1, 0.08 * dz], n * 3);
        rest.set([cx + 0.09 * dx, level * 0.1, 0.09 * dz], n * 3);
        indices.set([n, n, n], n * 3);
        weights.set([1, 0, 0], n * 3);
        n++;
      }
    }
  }
  return { body, rest, binding: { count, indices, weights, offsets } };
}

describe('leg-aware grading', () => {
  const { body, rest, binding } = twoLegs();
  const rings = [{ planeY: 0.5, deltaCircumferenceM: 0.1 }]; // +10 cm around
  const crotchY = 1.0; // above every vertex of the fixture

  it('the plain section-centroid model pushes inner-thigh vertices into the leg (the reason for the split)', () => {
    const plain = new Float32Array(rest.length);
    gradeGarment(rest, binding, body, rings, plain);
    // vertex of the +x leg on its inner side (angle pi): x = 0.1 - 0.09 = 0.01, moved towards +x by the pooled centroid at 0
    const inner = 4 + 8 * 5; // level 5, k = 4
    expect(plain[inner * 3]!).toBeGreaterThan(rest[inner * 3]!);
  });

  it('moves every vertex away from its own leg axis, on both legs', () => {
    const graded = new Float32Array(rest.length);
    gradeGarmentLegAware(rest, binding, body, rings, crotchY, graded, new SplitScratch());
    const expectedShift = 0.1 / (2 * Math.PI);
    for (let v = 0; v < binding.count; v++) {
      const cx = v < binding.count / 2 ? 0.1 : -0.1;
      const before = Math.hypot(rest[v * 3]! - cx, rest[v * 3 + 2]!);
      const after = Math.hypot(graded[v * 3]! - cx, graded[v * 3 + 2]!);
      expect(after - before).toBeCloseTo(expectedShift, 4);
      expect(graded[v * 3 + 1]).toBe(rest[v * 3 + 1]); // heights are untouched
    }
  });

  it('is identical to the plain pass without a crotch height', () => {
    const plain = new Float32Array(rest.length);
    const aware = new Float32Array(rest.length);
    gradeGarment(rest, binding, body, rings, plain);
    gradeGarmentLegAware(rest, binding, body, rings, undefined, aware, new SplitScratch());
    expect(Array.from(aware)).toEqual(Array.from(plain));
  });

  it('reports positive clearance all around both legs', () => {
    const clearance = new Float32Array(binding.count);
    clearanceLegAware(rest, binding, body, crotchY, clearance, new SplitScratch());
    expect(Math.min(...clearance)).toBeCloseTo(0.01, 4);
    // the pooled model reports the inner side as penetrating
    const pooled = new Float32Array(binding.count);
    garmentClearance(rest, binding, body, pooled);
    expect(Math.min(...pooled)).toBeLessThan(0);
  });

  it('never leaves a vertex inside the body after a shrinking grade (2 mm floor)', () => {
    const graded = new Float32Array(rest.length);
    gradeGarmentLegAware(rest, binding, body, [{ planeY: 0.5, deltaCircumferenceM: -0.3 }], crotchY, graded, new SplitScratch());
    const clearance = new Float32Array(binding.count);
    clearanceLegAware(graded, binding, body, crotchY, clearance, new SplitScratch());
    expect(Math.min(...clearance)).toBeGreaterThanOrEqual(0.0019);
  });
});
