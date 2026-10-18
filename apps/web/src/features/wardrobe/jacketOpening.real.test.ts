// @ts-expect-error Node types are supplied by Vitest.
import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';
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

describe.skipIf(!root)('denim opening on the real asset', () => {
  it.each([-0.06, -0.12, -0.18])(
    'keeps the opening continuous for a %s m girth change',
    (delta) => {
      const dir = root ?? '';
      const body = primitive(parseGlb(readFileSync(`${dir}/body/base.glb`))).position;
      const catalogue = JSON.parse(readFileSync(`${dir}/garments/index.json`, 'utf8')) as {
        garments: GarmentTemplateDef[];
      };
      const def = catalogue.garments.find((g) => g.id === 'mh-denim-jacket')!;
      const mesh = primitive(parseGlb(readFileSync(`${dir}/garments/${def.mesh}`)));
      const binding = parseGarmentBinding(bytes(`${dir}/garments/${def.binding!}`));
      const rest = new Float32Array(binding.count * 3);
      bindGarment(binding, def.scaleRefs, body, rest);
      const sections = createGarmentSections(body);
      const graded = new Float32Array(rest.length);
      const rings = [{ planeY: 1, deltaCircumferenceM: delta }];
      gradeGarment(rest, binding, body, rings, graded, sections, { outer: true });
      const groups = new Map<string, number>();
      const weld = Array.from({ length: binding.count }, (_, v) => {
        const key = [0, 1, 2].map((k) => Math.round(rest[v * 3 + k]! * 100000)).join(',');
        if (!groups.has(key)) groups.set(key, v);
        return groups.get(key)!;
      });
      const edges = new Map<string, { a: number; b: number; count: number }>();
      for (let t = 0; t < mesh.indices.length; t += 3) {
        for (let k = 0; k < 3; k++) {
          const a = weld[mesh.indices[t + k]!]!;
          const b = weld[mesh.indices[t + ((k + 1) % 3)]!]!;
          const key = a < b ? `${a},${b}` : `${b},${a}`;
          const edge = edges.get(key);
          if (edge) edge.count++;
          else edges.set(key, { a, b, count: 1 });
        }
      }
      // Both plackets from hem to collar; exclude the horizontal hem and pocket edges.
      const front = (v: number) =>
        Math.abs(rest[v * 3]!) < 0.045 &&
        rest[v * 3 + 2]! > 0.05 &&
        rest[v * 3 + 1]! > 0.85 &&
        rest[v * 3 + 1]! < 1.35;
      const opening = [...edges.values()].filter((e) => e.count === 1 && front(e.a) && front(e.b));
      const jumps = opening.map(({ a, b }) => ({
        a,
        b,
        mm:
          1000 *
          Math.hypot(
            ...[0, 1, 2].map(
              (k) => graded[a * 3 + k]! - rest[a * 3 + k]! - graded[b * 3 + k]! + rest[b * 3 + k]!,
            ),
          ),
      }));
      expect(opening.length).toBeGreaterThan(100);
      for (const side of [-1, 1]) {
        const vertices = [...new Set(opening.flatMap((e) => [e.a, e.b]))].filter(
          (v) => side * rest[v * 3]! > 0,
        );
        expect(vertices.length).toBeGreaterThan(20);
        const inward = vertices.map((v) => rest[v * 3 + 2]! - graded[v * 3 + 2]!);
        expect(inward.reduce((sum, d) => sum + d, 0) / inward.length).toBeGreaterThan(
          (Math.abs(delta) / (2 * Math.PI)) * 0.7,
        );
      }
      expect(Math.max(...jumps.map((e) => e.mm))).toBeLessThan(3);

      const tee = catalogue.garments.find((g) => g.id === 'mh-jacket-t01')!;
      const teeMesh = primitive(parseGlb(readFileSync(`${dir}/garments/${tee.mesh}`)));
      const teeBinding = parseGarmentBinding(bytes(`${dir}/garments/${tee.binding!}`));
      const teeRest = new Float32Array(teeBinding.count * 3);
      const teeBound = new Float32Array(teeRest.length);
      bindGarment(teeBinding, tee.scaleRefs, body, teeRest);
      boundBodyPoints(teeBinding, body, teeBound);
      const layered = graded.slice();
      expect(
        pushOutside(
          layered,
          binding.count,
          [{ positions: teeRest, index: teeMesh.indices, bound: teeBound }],
          {
            orientation: 'surface',
            marginM: OUTER_PUSH_MARGIN_M,
            frontDepthOnly: true,
          },
        ),
      ).toBeGreaterThan(0);
      const torso = [...new Set(opening.flatMap((e) => [e.a, e.b]))].filter(
        (v) => rest[v * 3 + 1]! < 1.2,
      );
      const lateralPush = torso.map((v) =>
        Math.hypot(layered[v * 3]! - graded[v * 3]!, layered[v * 3 + 1]! - graded[v * 3 + 1]!),
      );
      expect(Math.max(...lateralPush)).toBeLessThan(0.0001);
      expect(
        Math.max(...torso.map((v) => layered[v * 3 + 2]! - graded[v * 3 + 2]!)),
      ).toBeGreaterThan(0.005);
    },
  );
});
