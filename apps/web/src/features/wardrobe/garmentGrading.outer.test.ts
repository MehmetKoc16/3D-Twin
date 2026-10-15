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

  it("leaves an outer-slot template's graded positions equal to rest when only girth rings would apply", () => {
    const rings = buildGradeRings(item, jacket, {}, planes);
    expect(rings).toHaveLength(1); // chart 94 vs native 100 -> delta -6 cm would apply if not outer

    const outer = fit(jacket);
    expect(Array.from(outer.graded)).toEqual(Array.from(outer.rest));
  });

  it('still applies girth grading to non-outer garments when the chart differs from the body', () => {
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
      chart: { ...item.chart, length: [70] },
    };
    const planesWithWaist: BodyPlanes = {
      planeY: { waist: 1.2, chest: 1.4 },
    };

    const outer = fit(jacketWithLength, itemWithLength, planesWithWaist);
    expect(Array.from(outer.graded)).not.toEqual(Array.from(outer.rest));
    expect(outer.graded[1]).toBeCloseTo(0.9, 5);
    expect(outer.graded[4]).toBeCloseTo(0.9, 5);
    // Girth was still skipped: x and z positions remain identical to rest
    expect(outer.graded[0]).toBe(outer.rest[0]);
    expect(outer.graded[2]).toBe(outer.rest[2]);
    expect(outer.graded[3]).toBe(outer.rest[3]);
    expect(outer.graded[5]).toBe(outer.rest[5]);
  });
});
