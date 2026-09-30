import { describe, expect, it } from 'vitest';
import { type GarmentTemplateDef, type MeasuresDef, type StoreItemDef } from '@dt/avatar-core';
import { bodyPlanes, buildGradeRings, buildLengthGrades } from './garmentGrading';

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

describe('buildLengthGrades', () => {
  it('uses the chart delta for a top hem and emits one sleeve pass per arm', () => {
    const top: GarmentTemplateDef = {
      ...template,
      kind: 'sweatshirt',
      nativeMeasures: { ...template.nativeMeasures, length: 45 },
      defaultEase: { ...template.defaultEase, sleeve: -1 },
    };
    const sized: StoreItemDef = {
      ...item,
      chart: { ...item.chart, length: [47, 48, 49], sleeve: [54, 55, 56] },
    };
    const specs = buildLengthGrades(
      sized,
      top,
      { height: 165.9, armLength: 54 },
      { planeY: { waist: 1 } },
      [0, 1.1, 0, 0, 0.9, 0, 0, 0.8, 0],
      (name) =>
        name.startsWith('upperarm')
          ? [name.endsWith('_l') ? 0.2 : -0.2, 1.4, 0]
          : [name.endsWith('_l') ? 0.7 : -0.7, 1.0, 0],
    );
    expect(specs).toHaveLength(3);
    expect(specs[0]!.chartCm - specs[0]!.referenceCm).toBeCloseTo(3);
    expect(specs[1]!.chartCm - specs[1]!.referenceCm).toBeCloseTo(2);
    expect(specs[1]!.side).toBe(1);
    expect(specs[2]!.side).toBe(-1);
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
    const large = buildGradeRings(
      { ...item, selectedSize: 'L' },
      template,
      { chest: 96, waist: 80 },
      planes,
    );
    expect(large.find((r) => r.planeY === 1.3)!.deltaCircumferenceM).toBeCloseTo(0.08, 6);
    const small = buildGradeRings(
      { ...item, selectedSize: 'S' },
      template,
      { chest: 96, waist: 80 },
      planes,
    );
    expect(small.find((r) => r.planeY === 1.3)!.deltaCircumferenceM).toBeCloseTo(-0.08, 6);
  });

  it('falls back to the native measurement without a body value, and skips rows without plane or chart', () => {
    const noBody = buildGradeRings(item, template, {}, planes);
    expect(noBody.find((r) => r.planeY === 1.3)!.deltaCircumferenceM).toBeCloseTo(0.1, 6); // 100 - native 90
    expect(buildGradeRings(item, template, { chest: 96 }, { planeY: {} })).toEqual([]);
    expect(
      buildGradeRings(
        { ...item, chart: { length: [60, 62, 64] } },
        template,
        { chest: 96 },
        planes,
      ),
    ).toEqual([]);
    expect(
      buildGradeRings({ ...item, selectedSize: 'XXL' }, template, { chest: 96 }, planes),
    ).toEqual([]);
  });
});
