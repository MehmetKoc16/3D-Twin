import { describe, expect, it } from 'vitest';
import {
  createGarmentSections,
  gradeGarment,
  type GarmentBinding,
  type GarmentTemplateDef,
  type StoreItemDef,
} from '@dt/avatar-core';
import {
  buildGradeRings,
  fitToChart,
  OUTER_SMOOTH_RADIUS_M,
  type BodyPlanes,
} from './garmentGrading';

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

describe('fitToChart grading options', () => {
  const body = cylinderBody();
  const sections = createGarmentSections(body);
  const binding = stackedBinding(body);
  const planes: BodyPlanes = { planeY: { chest: 1 } };
  const none = (): undefined => undefined;

  function fit(def: GarmentTemplateDef): { rest: Float32Array; graded: Float32Array } {
    const rest = new Float32Array(6);
    const graded = new Float32Array(6);
    fitToChart(
      { def, item, binding, body, sections, achievedCm: {}, planes, joint: none },
      rest,
      graded,
    );
    return { rest, graded };
  }

  it('smooths the girth grading of outer-slot garments only; other slots keep legacy grading', () => {
    const rings = buildGradeRings(item, tee, {}, planes);
    expect(rings).toHaveLength(1); // chart 94 vs native 100 -> -6 cm

    const plain = fit(tee);
    const legacy = new Float32Array(6);
    gradeGarment(plain.rest, binding, body, rings, legacy, sections);
    expect(Array.from(plain.graded)).toEqual(Array.from(legacy));

    const outer = fit(jacket);
    const smoothed = new Float32Array(6);
    gradeGarment(outer.rest, binding, body, rings, smoothed, sections, {
      smoothRadiusM: OUTER_SMOOTH_RADIUS_M,
    });
    expect(Array.from(outer.graded)).toEqual(Array.from(smoothed));
    // The two paths really differ on this stacked pair.
    expect(Array.from(outer.graded)).not.toEqual(Array.from(plain.graded));
  });
});
