import { describe, expect, it } from 'vitest';
import { solveLinear, solveScalar } from './numeric';

describe('solveScalar', () => {
  it('finds the root of a linear function', () => {
    const r = solveScalar((x) => 3 * x - 1, 0.9, 0, 1, 1e-9);
    expect(r.converged).toBe(true);
    expect(r.x).toBeCloseTo(1 / 3, 8);
  });

  it('handles a piecewise-linear function with a kink (tent buckets)', () => {
    const f = (x: number): number =>
      (x < 0.5 ? 1.6 + 0.2 * (x / 0.5) : 1.8 + 0.4 * ((x - 0.5) / 0.5)) - 1.95;
    for (const x0 of [0, 0.2, 0.5, 0.8, 1]) {
      const r = solveScalar(f, x0, 0, 1, 1e-9);
      expect(r.converged).toBe(true);
      expect(Math.abs(f(r.x))).toBeLessThan(1e-9);
    }
  });

  it('reports an unreachable root and returns the closest bound', () => {
    const above = solveScalar((x) => x + 5, 0.5, 0, 1, 1e-6);
    expect(above.converged).toBe(false);
    expect(above.x).toBe(0);
    const below = solveScalar((x) => x - 5, 0.5, 0, 1, 1e-6);
    expect(below.converged).toBe(false);
    expect(below.x).toBe(1);
  });

  it('accepts a start that is already a root and a root exactly at a bound', () => {
    expect(solveScalar((x) => x - 0.25, 0.25, 0, 1, 1e-9).evaluations).toBe(1);
    const r = solveScalar((x) => x, 0.7, 0, 1, 1e-9);
    expect(r.converged).toBe(true);
    expect(r.x).toBeCloseTo(0, 8);
  });

  it('converges on a smooth nonlinear function', () => {
    const r = solveScalar((x) => x ** 3 + x - 0.4, 1, 0, 1, 1e-10);
    expect(r.converged).toBe(true);
    expect(r.x ** 3 + r.x).toBeCloseTo(0.4, 9);
  });
});

describe('solveLinear', () => {
  it('solves a 3x3 system (needs pivoting)', () => {
    const a = Float64Array.from([0, 2, 1, 1, 1, 1, 2, 1, 0]);
    const b = Float64Array.from([7, 6, 4]);
    expect(solveLinear(a, b, 3)).toBe(true);
    // x = [1, 2, 3]
    expect(Array.from(b)).toEqual([1, 2, 3].map((v) => expect.closeTo(v, 10)));
  });

  it('detects a singular matrix', () => {
    const a = Float64Array.from([1, 2, 2, 4]);
    expect(solveLinear(a, Float64Array.from([1, 2]), 2)).toBe(false);
  });
});
