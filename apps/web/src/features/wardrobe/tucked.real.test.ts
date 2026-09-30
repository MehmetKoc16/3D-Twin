// @ts-expect-error Node types are not in the browser app's tsconfig; Vitest supplies this module.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  buildBasePositions,
  createBodySolver,
  createGarmentSections,
  parseGarmentBinding,
  type BodyManifest,
  type BodyParams,
  type GarmentBinding,
  type GarmentTemplateDef,
  type MeasuresDef,
  type StoreItemDef,
} from '@dt/avatar-core';
import { groundRenderPositions } from '../../workers/ground';
import { coveredBodyVertices } from './bodyHide';
import { pushInside, pushOutside, type Surface } from './garmentCollide';
import { boundBodyPoints, bodyPlanes, fitToChart } from './garmentGrading';
import { bytes, parseGlb, primitive, root } from './realAssets.testkit';

/**
 * Regression tests for the tucked / untucked layering rules and the chart-length pass, on the real assets:
 * a lengthened hem keeps its rows together (no ragged band), a tucked top ends inside the trousers, and the outer
 * layer never has vertices inside the inner one.
 */

const bodies: [string, BodyParams][] = [
  [
    'mid 170/70',
    {
      gender: 0.5,
      heightCm: 170,
      weightKg: 70,
      chestCm: 96,
      waistCm: 80,
      hipCm: 98,
      shoe: { system: 'EU', size: 42 },
    },
  ],
  [
    'male 182/95',
    {
      gender: 1,
      heightCm: 182,
      weightKg: 95,
      chestCm: 106,
      waistCm: 94,
      hipCm: 104,
      shoe: { system: 'EU', size: 42 },
    },
  ],
];

describe.skipIf(!root)('tucked and untucked layering on the real assets', () => {
  const dir = root ?? '';
  const manifest = JSON.parse(readFileSync(`${dir}/body/manifest.json`, 'utf8')) as BodyManifest;
  const measures = JSON.parse(readFileSync(`${dir}/body/measures.json`, 'utf8')) as MeasuresDef;
  const baseGlb = primitive(parseGlb(readFileSync(`${dir}/body/base.glb`)));
  const base = buildBasePositions(baseGlb.position, manifest);
  const morphs = bytes(`${dir}/body/morphs.bin`);
  const catalogue = JSON.parse(readFileSync(`${dir}/garments/index.json`, 'utf8')) as {
    garments: GarmentTemplateDef[];
  };
  const solver = createBodySolver({ manifest, base, morphs, measures, indices: baseGlb.indices });

  interface Loaded {
    def: GarmentTemplateDef;
    binding: GarmentBinding;
    index: Uint32Array;
  }
  const load = (id: string): Loaded => {
    const def = catalogue.garments.find((g) => g.id === id)!;
    const mesh = primitive(parseGlb(readFileSync(`${dir}/garments/${def.mesh}`)));
    const binding = parseGarmentBinding(
      bytes(`${dir}/garments/${def.binding!}`),
      manifest.renderVertexCount,
    );
    return { def, binding, index: mesh.indices };
  };
  const item = (def: GarmentTemplateDef, chart: StoreItemDef['chart']): StoreItemDef => ({
    id: def.id,
    name: def.id,
    templateId: def.id,
    color: '#ffffff',
    sizes: ['S', 'M', 'L'],
    chart,
    selectedSize: 'M',
  });

  const tee = load('tshirt');
  const sweater = load('sweatshirt');
  const jeans = load('jeans');
  const pants = load('pants');
  const teeItem = item(tee.def, {
    chest: [98, 106, 114],
    waist: [90, 98, 106],
    length: [68, 70, 72],
    sleeve: [19, 20, 21],
  });
  const sweaterItem = item(sweater.def, {
    chest: [94, 100, 106],
    waist: [84, 90, 96],
    length: [58, 60, 62],
    sleeve: [52, 53, 54],
  });
  const bottomChart: StoreItemDef['chart'] = {
    waist: [80, 86, 92],
    hip: [98, 104, 110],
    inseam: [78, 79, 80],
    thigh: [60, 63, 66],
  };
  const jeansItem = item(jeans.def, bottomChart);
  const pantsItem = item(pants.def, bottomChart);

  interface Fit {
    rest: Float32Array;
    graded: Float32Array;
    surface: Surface;
  }

  const scenario = (params: BodyParams) => {
    const res = solver.solve(params);
    const body = groundRenderPositions(res.positions, manifest.renderVertexCount).positions;
    const planes = bodyPlanes(measures, body);
    const sections = createGarmentSections(body);
    const fit = (g: Loaded, it: StoreItemDef, tuckedHem = false): Fit => {
      const rest = new Float32Array(g.binding.count * 3);
      const graded = new Float32Array(g.binding.count * 3);
      fitToChart(
        {
          def: g.def,
          item: it,
          binding: g.binding,
          body,
          sections,
          achievedCm: res.achievedCm,
          planes,
          joint: () => undefined,
          tuckedHem,
        },
        rest,
        graded,
      );
      const bound = new Float32Array(g.binding.count * 3);
      boundBodyPoints(g.binding, body, bound);
      return { rest, graded, surface: { positions: graded, index: g.index, bound } };
    };
    return { fit, planes };
  };

  /** Lowest torso height (sleeves excluded) of a fitted top. */
  const hemMin = (positions: Float32Array): number => {
    let min = Infinity;
    for (let v = 0; v < positions.length / 3; v++)
      if (Math.abs(positions[v * 3]!) < 0.2) min = Math.min(min, positions[v * 3 + 1]!);
    return min;
  };

  /** Highest torso height of a fitted bottom (its waistband edge). */
  const waistbandTop = (positions: Float32Array): number => {
    let max = -Infinity;
    for (let v = 0; v < positions.length / 3; v++)
      if (Math.abs(positions[v * 3]!) < 0.25) max = Math.max(max, positions[v * 3 + 1]!);
    return max;
  };

  /** Vertical stretch of the torso's mesh edges: displaced height difference / rest height difference. */
  const edgeStretch = (g: Loaded, f: Fit): number[] => {
    const out: number[] = [];
    for (let t = 0; t < g.index.length; t += 3)
      for (const [a, b] of [
        [0, 1],
        [1, 2],
        [2, 0],
      ] as const) {
        const va = g.index[t + a]!;
        const vb = g.index[t + b]!;
        if (Math.abs(f.rest[va * 3]!) >= 0.2) continue;
        const dr = f.rest[va * 3 + 1]! - f.rest[vb * 3 + 1]!;
        if (Math.abs(dr) < 0.004) continue;
        out.push((f.graded[va * 3 + 1]! - f.graded[vb * 3 + 1]!) / dr);
      }
    return out;
  };

  /** Points that lie in front of (outside of) `surface` within its footprint. */
  const inFront = (points: Float32Array, surface: Surface): number => {
    // pushOutside counts points that are not yet outside; mirroring the bound points flips the surface orientation
    const mirrored = Float32Array.from(surface.positions as ArrayLike<number>, (p, i) => {
      return 2 * p - (surface.bound as ArrayLike<number>)[i]!;
    });
    const flipped: Surface = {
      positions: surface.positions,
      index: surface.index,
      bound: mirrored,
    };
    return pushOutside(Float32Array.from(points), points.length / 3, [flipped], {
      marginM: -0.0005,
      passes: 1,
    });
  };

  for (const [name, params] of bodies) {
    describe(name, () => {
      const { fit, planes } = scenario(params);

      it('a lengthened hem keeps its rows together: the hem band is rigid and no edge is over-stretched', () => {
        const f = fit(tee, teeItem, false);
        const restMin = hemMin(f.rest);
        expect(hemMin(f.graded)).toBeLessThan(restMin - 0.05); // the chart asks for a much longer tee
        const before: number[] = [];
        const after: number[] = [];
        for (let v = 0; v < f.rest.length / 3; v++) {
          if (Math.abs(f.rest[v * 3]!) >= 0.2 || f.rest[v * 3 + 1]! > restMin + 0.03) continue;
          before.push(f.rest[v * 3 + 1]!);
          after.push(f.graded[v * 3 + 1]!);
        }
        expect(before.length).toBeGreaterThan(20);
        const spread = (a: number[]): number => Math.max(...a) - Math.min(...a);
        expect(spread(after) - spread(before)).toBeLessThan(0.003);
        // the grading anchor is the chest plane: the change is spread over the torso (it was ~5x from the waist)
        const stretch = edgeStretch(tee, f);
        expect(Math.max(...stretch)).toBeLessThan(2.2);
        expect(Math.min(...stretch)).toBeGreaterThan(0.95);
      });

      it('a sweater hem is graded smoothly as well', () => {
        const stretch = edgeStretch(sweater, fit(sweater, sweaterItem));
        expect(Math.max(...stretch)).toBeLessThan(2.2);
        expect(Math.min(...stretch)).toBeGreaterThan(0.95);
      });

      it('a tucked tee worn with jeans skips the hem pass and ends inside the waistband', () => {
        const f = fit(tee, teeItem, true);
        expect(hemMin(f.graded)).toBeCloseTo(hemMin(f.rest), 4); // hem unchanged
        const top = waistbandTop(fit(jeans, jeansItem).graded);
        const hem = hemMin(f.graded);
        expect(hem).toBeLessThan(top - 0.02); // below the waistband edge, so it is covered
        expect(hem).toBeGreaterThan(top - 0.15); // but not down the hips
        expect(planes.planeY.waist).toBeGreaterThan(hem);
      });

      it('tucked tee + jeans: the jeans are the outer layer at the waistband', () => {
        const top = fit(tee, teeItem, true);
        const bottom = fit(jeans, jeansItem);
        // the rig's order: the jeans are pushed out of the tee, then the tee hem is tucked into the jeans
        const moved = pushOutside(bottom.graded, jeans.binding.count, [top.surface]);
        expect(moved).toBeGreaterThan(0); // the tee's waist (chart 98 cm) is wider than the jeans' (86 cm)
        pushInside(top.graded, tee.binding.count, [bottom.surface]);
        // (almost) no jeans vertex is left inside the visible part of the tee (the rig hides the tee triangles behind the jeans)
        const covered = coveredBodyVertices(
          top.graded,
          tee.binding.count,
          bottom.graded,
          jeans.index,
          {
            bandM: 0.02,
          },
        );
        const visible: number[] = [];
        for (let t = 0; t < tee.index.length; t += 3) {
          const [a, b, c] = [tee.index[t]!, tee.index[t + 1]!, tee.index[t + 2]!];
          if (!(covered[a] && covered[b] && covered[c])) visible.push(a, b, c);
        }
        expect(visible.length).toBeGreaterThan(0);
        const visibleTee = [{ ...top.surface, index: visible }];
        const insideTee = (marginM: number): number =>
          pushOutside(Float32Array.from(bottom.graded), jeans.binding.count, visibleTee, {
            marginM,
            passes: 1,
          });
        expect(insideTee(-0.01)).toBeLessThanOrEqual(Math.ceil(jeans.binding.count * 0.002)); // deeper than 1 cm
        expect(insideTee(0.004)).toBeLessThanOrEqual(Math.ceil(jeans.binding.count * 0.02)); // a few edge vertices
        // ... and no tee vertex of the waistband band pokes out of the jeans
        const edge = waistbandTop(bottom.graded);
        const points: number[] = [];
        for (let v = 0; v < tee.binding.count; v++) {
          const y = top.graded[v * 3 + 1]!;
          if (Math.abs(top.graded[v * 3]!) < 0.2 && y <= edge - 0.005 && y >= edge - 0.06)
            points.push(top.graded[v * 3]!, y, top.graded[v * 3 + 2]!);
        }
        expect(points.length / 3).toBeGreaterThan(10);
        // (33 of them poked out before the hem was tucked into the waistband; a stray vertex at a fold may remain)
        expect(inFront(Float32Array.from(points), bottom.surface)).toBeLessThanOrEqual(
          Math.ceil((points.length / 3) * 0.03),
        );
      });

      it('untucked sweater + pants: the sweater is the outer layer and clears the trousers', () => {
        const top = fit(sweater, sweaterItem);
        const bottom = fit(pants, pantsItem);
        pushOutside(top.graded, sweater.binding.count, [bottom.surface]);
        const inside = (marginM: number): number =>
          pushOutside(Float32Array.from(top.graded), sweater.binding.count, [bottom.surface], {
            marginM,
            passes: 1,
          });
        expect(inside(0.004)).toBeLessThanOrEqual(Math.ceil(sweater.binding.count * 0.002));
      });
    });
  }
});
