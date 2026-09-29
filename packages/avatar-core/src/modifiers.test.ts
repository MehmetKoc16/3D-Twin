import { describe, expect, it } from 'vitest';
import { combineWeights, mergeWeights, modifierWeights } from './modifiers';
import { getSyntheticBody } from './testing/syntheticBody';

const fx = getSyntheticBody();
const WAIST = 'measure/measure-waist-circ-decr|incr';

describe('modifierWeights', () => {
  it('maps negative values to |v| on the decr target and positive to v on the incr target', () => {
    expect(modifierWeights(fx.manifest, { [WAIST]: -0.4 })).toEqual(
      new Map([['measure-waist-circ-decr', 0.4]]),
    );
    expect(modifierWeights(fx.manifest, { [WAIST]: 0.25 })).toEqual(
      new Map([['measure-waist-circ-incr', 0.25]]),
    );
    expect(modifierWeights(fx.manifest, { [WAIST]: 0 }).size).toBe(0);
    expect(modifierWeights(fx.manifest, {}).size).toBe(0);
  });

  it('clamps to the modifier range and ignores a missing side', () => {
    expect(modifierWeights(fx.manifest, { [WAIST]: 7 }).get('measure-waist-circ-incr')).toBe(1);
    expect(modifierWeights(fx.manifest, { [WAIST]: -7 }).get('measure-waist-circ-decr')).toBe(1);
    const only = fx.manifest.modifiers.find((m) => m.decrTarget === undefined)!;
    expect(only.min).toBe(0);
    expect(modifierWeights(fx.manifest, { [only.id]: -1 }).size).toBe(0);
  });

  it('rejects unknown modifiers and non-finite values', () => {
    expect(() => modifierWeights(fx.manifest, { ghost: 1 })).toThrow(/unknown modifier "ghost"/);
    expect(() => modifierWeights(fx.manifest, { [WAIST]: NaN })).toThrow(/non-finite/);
  });

  it('applies non-zero defaults for modifiers that are not given', () => {
    const m = structuredClone(fx.manifest);
    m.modifiers.find((x) => x.id === WAIST)!.default = 0.5;
    expect(modifierWeights(m, {}).get('measure-waist-circ-incr')).toBe(0.5);
    expect(modifierWeights(m, { [WAIST]: 0 }).size).toBe(0);
  });
});

describe('mergeWeights / combineWeights', () => {
  it('sums shared targets', () => {
    const merged = mergeWeights(
      new Map([
        ['a', 0.5],
        ['b', 1],
      ]),
      new Map([
        ['a', 0.25],
        ['c', 2],
      ]),
    );
    expect(merged).toEqual(
      new Map([
        ['a', 0.75],
        ['b', 1],
        ['c', 2],
      ]),
    );
  });

  it('merges macro and modifier weights into one map', () => {
    const w = combineWeights(fx.manifest, { gender: 1, weight: 1 }, { [WAIST]: 0.3 });
    expect(w.get('measure-waist-circ-incr')).toBe(0.3);
    expect(w.get('macro-male-averagemuscle-maxweight-averageheight')).toBeCloseTo(1, 12);
  });
});
