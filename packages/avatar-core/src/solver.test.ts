import { describe, expect, it } from 'vitest';
import type { BodyParams, MeasureId } from './contracts';
import { macroWeights } from './macro';
import { estimateMassKg, measure } from './measure';
import { combineWeights } from './modifiers';
import { applyMorphs } from './morphs';
import { footLengthCmFromShoe } from './shoe';
import { bmiToWeightValue, createBodySolver, solveBody } from './solver';
import { getSyntheticBody, mulberry32 } from './testing/syntheticBody';

const fx = getSyntheticBody();
const shoe = { system: 'EU', size: 42 } as const;
const RENDER = fx.manifest.renderVertexCount;
const def = (id: MeasureId) => fx.measures.measures.find((m) => m.id === id)!;
const measureCm = (id: MeasureId, p: Float32Array): number => measure(def(id), p, RENDER) * 100;

type LocalKey = 'neckCm' | 'chestCm' | 'waistCm' | 'hipCm' | 'thighCm' | 'shoulderCm' | 'inseamCm';
const LOCAL: [LocalKey, MeasureId][] = [
  ['neckCm', 'neck'],
  ['chestCm', 'chest'],
  ['waistCm', 'waist'],
  ['hipCm', 'hip'],
  ['thighCm', 'thigh'],
  ['shoulderCm', 'shoulder'],
  ['inseamCm', 'inseam'],
];

function randomBase(rand: () => number): BodyParams {
  const heightCm = 155 + 40 * rand();
  const bmi = 19 + 8 * rand();
  return {
    gender: rand(),
    heightCm,
    weightKg: bmi * (heightCm / 100) ** 2,
    shoe,
  };
}

describe('solveBody: height and weight', () => {
  it('hits the requested height within 0.5 cm (tightly, in fact) from seeded random params', () => {
    const rand = mulberry32(2024);
    for (let i = 0; i < 8; i++) {
      const p = randomBase(rand);
      const r = solveBody(fx.data, p);
      expect(Math.abs(r.residualsCm.height!)).toBeLessThan(0.05);
      expect(Math.abs(measureCm('height', r.positions) - p.heightCm)).toBeLessThan(0.05);
      expect(r.unreachable).toEqual([]);
    }
  });

  it('matches the requested mass through the mesh volume', () => {
    for (const weightKg of [55, 70, 90, 110]) {
      const r = solveBody(fx.data, { gender: 0.5, heightCm: 172, weightKg, shoe });
      expect(Math.abs(r.estimatedMassKg - weightKg)).toBeLessThan(0.1);
      expect(estimateMassKg(r.positions, fx.indices, 1.01, RENDER)).toBeCloseTo(
        r.estimatedMassKg,
        6,
      );
    }
  });

  it('without indices falls back to the BMI seed and reports NaN mass', () => {
    const p: BodyParams = { gender: 1, heightCm: 180, weightKg: 81, shoe };
    const r = solveBody({ ...fx.data, indices: undefined }, p);
    expect(r.estimatedMassKg).toBeNaN();
    expect(r.macroVars.weight).toBeCloseTo(bmiToWeightValue(81 / 1.8 ** 2), 12);
    expect(Math.abs(r.residualsCm.height!)).toBeLessThan(0.05);
  });

  it('maps BMI to the macro weight value piecewise-linearly and clamps', () => {
    expect(bmiToWeightValue(10)).toBe(0);
    expect(bmiToWeightValue(16)).toBe(0);
    expect(bmiToWeightValue(22)).toBeCloseTo(0.5, 12);
    expect(bmiToWeightValue(28)).toBeCloseTo(0.5 + (0.5 * 6) / 12, 12);
    expect(bmiToWeightValue(50)).toBe(1);
  });

  it('passes gender through to the macro variable and honours macro overrides', () => {
    const r = solveBody(
      fx.data,
      { gender: 0.2, heightCm: 170, weightKg: 65, shoe },
      { macroOverrides: { muscle: 0.9 } },
    );
    expect(r.macroVars.gender).toBe(0.2);
    expect(r.macroVars.muscle).toBe(0.9);
  });
});

describe('solveBody: local measures', () => {
  it('hits height and all circumferences / distances from seeded random params', () => {
    const rand = mulberry32(31337);
    for (let i = 0; i < 6; i++) {
      const p = randomBase(rand);
      const derived = solveBody(fx.data, p).achievedCm;
      const target: BodyParams = { ...p };
      for (const [key, id] of LOCAL) {
        const d = derived[id]!;
        target[key] = id === 'inseam' ? d + (rand() * 8 - 4) : d * (1 + (rand() * 0.3 - 0.15));
      }
      const r = solveBody(fx.data, target);
      expect(r.unreachable).toEqual([]);
      expect(Math.abs(measureCm('height', r.positions) - target.heightCm)).toBeLessThan(0.5);
      for (const [key, id] of LOCAL) {
        expect(Math.abs(measureCm(id, r.positions) - target[key]!), `${id} case ${i}`).toBeLessThan(
          0.5,
        );
        expect(Math.abs(r.residualsCm[id]!)).toBeLessThan(0.05);
      }
      expect(Math.abs(r.residualsCm.height!)).toBeLessThan(0.05);
    }
  });

  it('recovers the modifier values that generated the measurements (forward model)', () => {
    const rand = mulberry32(555);
    const drivers = new Map(LOCAL.map(([key, id]) => [key, def(id).drivers[0]!]));
    for (let i = 0; i < 4; i++) {
      const macro = { gender: rand(), weight: 0.3 + 0.4 * rand(), height: 0.3 + 0.4 * rand() };
      const truth: Record<string, number> = {};
      for (const [key, driver] of drivers) truth[driver] = key === 'inseamCm' ? 0 : rand() - 0.5;
      const pos = new Float32Array(fx.base.length);
      applyMorphs(fx.base, fx.morphs, fx.manifest, combineWeights(fx.manifest, macro, truth), pos);
      // mass at zero modifiers (the solver fits the weight variable before the modifiers)
      const neutral = new Float32Array(fx.base.length);
      applyMorphs(fx.base, fx.morphs, fx.manifest, macroWeights(fx.manifest, macro), neutral);
      const params: BodyParams = {
        gender: macro.gender,
        heightCm: measureCm('height', pos),
        weightKg: estimateMassKg(neutral, fx.indices, 1.01, RENDER),
        shoe,
      };
      for (const [key, id] of LOCAL) params[key] = measureCm(id, pos);
      const r = solveBody(fx.data, params);
      expect(r.unreachable).toEqual([]);
      for (const [driver, value] of Object.entries(truth))
        expect(r.modifierValues[driver] ?? 0, driver).toBeCloseTo(value, 2);
      expect(r.macroVars.height!).toBeCloseTo(macro.height, 3);
      expect(r.macroVars.weight!).toBeCloseTo(macro.weight, 2);
    }
  });

  it('keeps circumferences on target when the height changes (height/circumference coupling)', () => {
    for (const heightCm of [152, 170, 188]) {
      const r = solveBody(fx.data, {
        gender: 0.5,
        heightCm,
        weightKg: 70,
        waistCm: 80,
        chestCm: 96,
        inseamCm: heightCm * 0.46,
        shoe,
      });
      expect(r.unreachable).toEqual([]);
      expect(r.achievedCm.waist!).toBeCloseTo(80, 1);
      expect(r.achievedCm.chest!).toBeCloseTo(96, 1);
      expect(r.achievedCm.height!).toBeCloseTo(heightCm, 1);
      expect(r.achievedCm.inseam!).toBeCloseTo(heightCm * 0.46, 1);
    }
  });

  it('leaves unprovided measures at their derived value and touches only needed modifiers', () => {
    const base = solveBody(fx.data, { gender: 0.5, heightCm: 172, weightKg: 70, shoe });
    const r = solveBody(fx.data, { gender: 0.5, heightCm: 172, weightKg: 70, waistCm: 85, shoe });
    expect(Object.keys(r.modifierValues)).toEqual(['measure/measure-waist-circ-decr|incr']);
    expect(r.achievedCm.waist!).toBeCloseTo(85, 1);
    expect(r.achievedCm.neck!).toBeCloseTo(base.achievedCm.neck!, 6);
    expect(base.iterations).toBe(0);
  });

  it('reports unreachable targets instead of throwing', () => {
    const r = solveBody(fx.data, {
      gender: 0.5,
      heightCm: 250, // beyond the height macro range
      weightKg: 70,
      waistCm: 250, // beyond the modifier range
      chestCm: 96, // reachable
      upperArmCm: 30, // no such measure in the fixture
      shoe,
    });
    expect(r.unreachable).toContain('height');
    expect(r.unreachable).toContain('waist');
    expect(r.unreachable).toContain('upperArm');
    expect(r.unreachable).not.toContain('chest');
    expect(r.residualsCm.waist!).toBeLessThan(-50);
    expect(r.residualsCm.height!).toBeLessThan(-30);
    expect(r.achievedCm.chest!).toBeCloseTo(96, 1);
    expect(r.positions.every(Number.isFinite)).toBe(true);
    // the unreachable modifier is driven to its bound, not beyond
    expect(r.modifierValues['measure/measure-waist-circ-decr|incr']).toBeCloseTo(1, 6);
  });

  it('reports measures whose driver modifier does not exist as unreachable', () => {
    const data = {
      ...fx.data,
      measures: {
        version: 1 as const,
        measures: fx.measures.measures.map((m) =>
          m.id === 'waist' ? { ...m, drivers: ['ghost'] } : m,
        ),
      },
    };
    const r = solveBody(data, { gender: 0.5, heightCm: 172, weightKg: 70, waistCm: 90, shoe });
    expect(r.unreachable).toEqual(['waist']);
  });

  it('splits the effort over several drivers of one measure (regularized, underdetermined)', () => {
    const data = {
      ...fx.data,
      measures: {
        version: 1 as const,
        measures: fx.measures.measures.map((m) =>
          m.id === 'waist'
            ? { ...m, drivers: [...m.drivers, 'measure/measure-hip-circ-decr|incr'] }
            : m,
        ),
      },
    };
    const r = solveBody(data, { gender: 0.5, heightCm: 172, weightKg: 70, waistCm: 90, shoe });
    expect(r.unreachable).toEqual([]);
    expect(r.achievedCm.waist!).toBeCloseTo(90, 1);
    const waist = r.modifierValues['measure/measure-waist-circ-decr|incr']!;
    const hip = r.modifierValues['measure/measure-hip-circ-decr|incr']!;
    expect(waist).toBeGreaterThan(0);
    expect(Math.abs(hip)).toBeGreaterThan(0.001); // shared effort, not all on one driver
    expect(Math.abs(hip)).toBeLessThan(Math.abs(waist));
  });

  it('takes the footLength target from the shoe size when the measure is defined', () => {
    const shoulder = fx.measures.measures.find((m) => m.id === 'shoulder');
    if (shoulder?.type !== 'distance') throw new Error('fixture shoulder must be a distance');
    const data = {
      ...fx.data,
      measures: {
        version: 1 as const,
        measures: [...fx.measures.measures, { ...shoulder, id: 'footLength' as const }],
      },
    };
    const r = solveBody(data, { gender: 0.5, heightCm: 172, weightKg: 70, shoe });
    expect(r.unreachable).toEqual([]);
    expect(r.achievedCm.footLength!).toBeCloseTo(footLengthCmFromShoe(shoe), 1);
  });

  it('is consistent: applyMorphs(weights) reproduces the returned positions', () => {
    const r = solveBody(fx.data, {
      gender: 0.7,
      heightCm: 181,
      weightKg: 88,
      waistCm: 92,
      hipCm: 104,
      inseamCm: 84,
      shoe,
    });
    const again = new Float32Array(fx.base.length);
    applyMorphs(fx.base, fx.morphs, fx.manifest, r.weights, again);
    let maxDiff = 0;
    for (let i = 0; i < again.length; i++)
      maxDiff = Math.max(maxDiff, Math.abs(again[i]! - r.positions[i]!));
    expect(maxDiff).toBeLessThan(1e-5);
    for (const v of Object.values(r.modifierValues)) expect(Math.abs(v)).toBeLessThanOrEqual(1);
  });

  it('is deterministic and the solver instance is cached per dataset', () => {
    const p: BodyParams = {
      gender: 0.4,
      heightCm: 168,
      weightKg: 62,
      waistCm: 74,
      thighCm: 52,
      shoe,
    };
    const a = solveBody(fx.data, p);
    const b = solveBody(fx.data, p);
    expect(a.positions).toEqual(b.positions);
    expect(a.iterations).toBe(b.iterations);
    expect(createBodySolver(fx.data)).toBe(createBodySolver({ ...fx.data }));
  });

  it('validates the input', () => {
    expect(() => solveBody(fx.data, { gender: 0.5, heightCm: NaN, weightKg: 70, shoe })).toThrow(
      /heightCm/,
    );
    expect(() => solveBody(fx.data, { gender: 0.5, heightCm: 170, weightKg: 0, shoe })).toThrow(
      /weightKg/,
    );
    expect(() =>
      solveBody(fx.data, { gender: 0.5, heightCm: 170, weightKg: 70, waistCm: -3, shoe }),
    ).toThrow(/waistCm/);
    expect(() =>
      solveBody(
        { ...fx.data, base: new Float32Array(9) },
        { gender: 0.5, heightCm: 170, weightKg: 70, shoe },
      ),
    ).toThrow(/base length/);
  });
});

describe('solveBody: performance', () => {
  it('solves a full 7-measure request on ~15k vertices / 20 modifiers quickly', () => {
    const p: BodyParams = {
      gender: 0.35,
      heightCm: 174,
      weightKg: 72,
      neckCm: 35,
      chestCm: 94,
      waistCm: 78,
      hipCm: 99,
      thighCm: 54,
      shoulderCm: 38,
      inseamCm: 80,
      shoe,
    };
    const solver = createBodySolver(fx.data);
    const cold = performance.now();
    solver.solve(p); // includes JIT warm-up
    const coldMs = performance.now() - cold;
    const times: number[] = [];
    for (let i = 0; i < 10; i++) {
      const t0 = performance.now();
      const r = solver.solve({ ...p, waistCm: 76 + i, heightCm: 172 + (i % 4) });
      times.push(performance.now() - t0);
      expect(r.unreachable).toEqual([]);
    }
    times.sort((a, b) => a - b);
    const median = times[Math.floor(times.length / 2)]!;
    console.log(
      `[solveBody] ${fx.manifest.vertexCount} vertices, ${fx.manifest.modifiers.length} modifiers: ` +
        `cold ${coldMs.toFixed(1)} ms, warm median ${median.toFixed(1)} ms, max ${times[times.length - 1]!.toFixed(1)} ms`,
    );
    expect(median).toBeLessThan(250);
  });
});
