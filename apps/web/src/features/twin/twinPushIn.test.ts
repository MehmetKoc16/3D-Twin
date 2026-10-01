import { describe, expect, it } from 'vitest';
import { MAX_PUSH_M, pushInPositions, pushInWeights } from './twinPushIn';

describe('scan push-in', () => {
  const rest = Float32Array.from([0, 0, 0, 0.0125, 0, 0, 0.025, 0, 0, 0.04, 0, 0]);
  it('covers the garment fully and smoothly falls to zero through the 2.5 cm margin', () => {
    const weights = pushInWeights(rest, Uint8Array.from([1, 0, 0, 0]));
    expect([...weights]).toEqual([1, 0.5, 0, 0]);
    const near = pushInWeights(
      Float32Array.from([0, 0, 0, 0.00001, 0, 0, 0.02499, 0, 0]),
      Uint8Array.from([1, 0, 0]),
    );
    expect(near[1]).toBeGreaterThan(0.999);
    expect(near[2]).toBeLessThan(0.001);
  });
  it('moves inward at most 8 mm, normalizes normals and restores exactly when taken off', () => {
    const normal = Float32Array.from([0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0]);
    const weights = pushInWeights(rest, Uint8Array.from([1, 0, 0, 0]));
    const out = rest.slice();
    pushInPositions(rest, normal, weights, out);
    expect(out[2]).toBeCloseTo(-MAX_PUSH_M);
    expect(out[5]).toBeCloseTo(-MAX_PUSH_M / 2);
    expect(out[8]).toBe(0);
    const first = out.slice();
    pushInPositions(rest, normal, weights, out);
    expect(out).toEqual(first);
    pushInPositions(rest, normal, pushInWeights(rest, new Uint8Array(4)), out);
    expect(out).toEqual(rest);
  });
});
