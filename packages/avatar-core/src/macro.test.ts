import { describe, expect, it } from 'vitest';
import type { BodyManifest } from './contracts';
import { macroWeights } from './macro';
import { getSyntheticBody, mulberry32 } from './testing/syntheticBody';
import { tinyManifest } from './testing/tiny';

const fx = getSyntheticBody();

/** One-variable manifest with one target per bucket. */
function oneVar(buckets: { name: string; at: number }[], min = 0, max = 1): BodyManifest {
  const m = tinyManifest(
    buckets.map((b) => ({
      id: `t-${b.name}`,
      group: 'macro' as const,
      byteOffset: 0,
      count: 0,
      macroConditions: [{ variable: 'weight' as const, bucket: b.name }],
    })),
  );
  m.macroVariables[0]!.buckets = buckets;
  m.macroVariables[0]!.min = min;
  m.macroVariables[0]!.max = max;
  return m;
}

describe('macroWeights', () => {
  const m = oneVar([
    { name: 'minweight', at: 0 },
    { name: 'averageweight', at: 0.5 },
    { name: 'maxweight', at: 1 },
  ]);

  it('adjacent bucket weights sum to 1 across the whole range', () => {
    for (let v = 0; v <= 1.0000001; v += 0.01) {
      const w = macroWeights(m, { weight: v });
      const sum = [...w.values()].reduce((s, x) => s + x, 0);
      expect(sum).toBeCloseTo(1, 12);
      expect(w.size).toBeLessThanOrEqual(2);
    }
  });

  it('is exact at the knots and linear in between', () => {
    expect(macroWeights(m, { weight: 0 })).toEqual(new Map([['t-minweight', 1]]));
    expect(macroWeights(m, { weight: 0.5 })).toEqual(new Map([['t-averageweight', 1]]));
    expect(macroWeights(m, { weight: 1 })).toEqual(new Map([['t-maxweight', 1]]));
    const w = macroWeights(m, { weight: 0.125 });
    expect(w.get('t-minweight')).toBeCloseTo(0.75, 12);
    expect(w.get('t-averageweight')).toBeCloseTo(0.25, 12);
    expect(w.has('t-maxweight')).toBe(false);
  });

  it('handles non-uniform buckets (age-like) regardless of declaration order', () => {
    const age = oneVar([
      { name: 'old', at: 1 },
      { name: 'baby', at: 0 },
      { name: 'young', at: 0.5 },
      { name: 'child', at: 0.1875 },
    ]);
    const w = macroWeights(age, { weight: 0.34375 }); // halfway between child and young
    expect(w.get('t-child')).toBeCloseTo(0.5, 12);
    expect(w.get('t-young')).toBeCloseTo(0.5, 12);
    expect(w.size).toBe(2);
  });

  it('clamps inputs and falls back to the default for missing / NaN values', () => {
    expect(macroWeights(m, { weight: -3 })).toEqual(new Map([['t-minweight', 1]]));
    expect(macroWeights(m, { weight: 9 })).toEqual(new Map([['t-maxweight', 1]]));
    expect(macroWeights(m, {})).toEqual(new Map([['t-averageweight', 1]]));
    expect(macroWeights(m, { weight: NaN })).toEqual(new Map([['t-averageweight', 1]]));
    const narrow = oneVar(
      [
        { name: 'minweight', at: 0 },
        { name: 'maxweight', at: 1 },
      ],
      0.2,
      0.8,
    );
    // clamped to 0.8 -> weight 0.8 on maxweight
    expect(macroWeights(narrow, { weight: 5 }).get('t-maxweight')).toBeCloseTo(0.8, 12);
  });

  it('multiplies the tent weights over the conditions of a target (synthetic fixture)', () => {
    const w = macroWeights(fx.manifest, { gender: 0.25, muscle: 0.5, weight: 0.25, height: 1 });
    // female 0.75 * averagemuscle 1 * minweight 0.5 * maxheight 1
    expect(w.get('macro-female-averagemuscle-minweight-maxheight')).toBeCloseTo(0.75 * 0.5, 12);
    expect(w.get('macro-male-averagemuscle-averageweight-maxheight')).toBeCloseTo(0.25 * 0.5, 12);
    expect(w.has('macro-female-minmuscle-minweight-maxheight')).toBe(false);
  });

  it('forms a partition of unity over the macro targets for random inputs', () => {
    const rand = mulberry32(99);
    for (let i = 0; i < 50; i++) {
      const w = macroWeights(fx.manifest, {
        gender: rand(),
        muscle: rand(),
        weight: rand(),
        height: rand(),
      });
      expect([...w.values()].reduce((s, x) => s + x, 0)).toBeCloseTo(1, 12);
    }
  });

  it('ignores non-macro targets (no macroConditions)', () => {
    const w = macroWeights(fx.manifest, {});
    for (const id of w.keys()) expect(id.startsWith('macro-')).toBe(true);
  });
});
