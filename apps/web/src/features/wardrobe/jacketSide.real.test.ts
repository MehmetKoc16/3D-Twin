// @ts-expect-error Node types are supplied by Vitest.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  bindGarment,
  createGarmentSections,
  gradeGarment,
  parseGarmentBinding,
  type GarmentTemplateDef,
} from '@dt/avatar-core';
import { bytes, parseGlb, primitive, root } from './realAssets.testkit';
import { boundBodyPoints } from './garmentGrading';
import { OUTER_PUSH_MARGIN_M, pushOutside } from './garmentCollide';

describe.skipIf(!root)('denim side over the real k4 garments', () => {
  it('keeps neighbouring side-panel collision displacements continuous', () => {
    const dir = root ?? '';
    const body = primitive(parseGlb(readFileSync(`${dir}/body/base.glb`))).position;
    const catalogue = JSON.parse(readFileSync(`${dir}/garments/index.json`, 'utf8')) as {
      garments: GarmentTemplateDef[];
    };
    const sections = createGarmentSections(body);
    const load = (id: string, outer = false) => {
      const def = catalogue.garments.find((g) => g.id === id)!;
      const mesh = primitive(parseGlb(readFileSync(`${dir}/garments/${def.mesh}`)));
      const binding = parseGarmentBinding(bytes(`${dir}/garments/${def.binding!}`));
      const rest = new Float32Array(binding.count * 3);
      const positions = new Float32Array(rest.length);
      const bound = new Float32Array(rest.length);
      bindGarment(binding, def.scaleRefs, body, rest);
      gradeGarment(
        rest,
        binding,
        body,
        outer ? [{ planeY: 1, deltaCircumferenceM: -0.12 }] : [],
        positions,
        sections,
        { outer },
      );
      boundBodyPoints(binding, body, bound);
      return { positions, bound, index: mesh.indices };
    };
    const jacket = load('mh-denim-jacket', true);
    const polo = load('mh-male-polo-shirt');
    const pants = load('mh-m-trousers-02');
    const shoes = load('sf-sneakers-lowpoly');
    const options = { orientation: 'surface' as const, marginM: OUTER_PUSH_MARGIN_M };
    const normal = jacket.positions.slice();
    const fitted = jacket.positions.slice();
    expect(pushOutside(normal, normal.length / 3, [polo, pants, shoes], options)).toBeGreaterThan(
      100,
    );
    expect(
      pushOutside(fitted, fitted.length / 3, [polo, pants, shoes], {
        ...options,
        frontDepthOnly: true,
      }),
    ).toBeGreaterThan(100);
    const side = (v: number) =>
      Math.abs(jacket.positions[v * 3]!) > 0.11 &&
      Math.abs(jacket.positions[v * 3]!) < 0.24 &&
      jacket.positions[v * 3 + 1]! > 0.95 &&
      jacket.positions[v * 3 + 1]! < 1.3;
    const edges = new Map<string, [number, number]>();
    for (let t = 0; t < jacket.index.length; t += 3)
      for (let k = 0; k < 3; k++) {
        const a = jacket.index[t + k]!;
        const b = jacket.index[t + ((k + 1) % 3)]!;
        if (side(a) && side(b)) edges.set(a < b ? `${a},${b}` : `${b},${a}`, [a, b]);
      }
    const jump = (p: Float32Array, a: number, b: number) =>
      Math.hypot(
        ...[0, 1, 2].map(
          (k) =>
            p[a * 3 + k]! -
            jacket.positions[a * 3 + k]! -
            p[b * 3 + k]! +
            jacket.positions[b * 3 + k]!,
        ),
      );
    const values = [...edges.values()].map(([a, b]) => ({
      a,
      b,
      fitted: jump(fitted, a, b) * 1000,
      normal: jump(normal, a, b) * 1000,
    }));
    // These are mesh neighbours on both side panels, not nearest points across garment layers.
    expect(values.length).toBeGreaterThan(1000);
    for (const sign of [-1, 1])
      expect(values.filter((e) => sign * jacket.positions[e.a * 3]! > 0).length).toBeGreaterThan(
        100,
      );
    // Existing armpit folds produce up to 58 mm on long edges; depth protection must not add a spike.
    expect(Math.max(...values.map((e) => e.fitted))).toBeLessThan(60);
    expect(Math.max(...values.map((e) => Math.abs(e.fitted - e.normal)))).toBeLessThan(0.1);
  });
});
