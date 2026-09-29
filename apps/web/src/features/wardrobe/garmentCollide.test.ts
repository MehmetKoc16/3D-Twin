import { describe, expect, it } from 'vitest';
import { pushOutside, type Surface } from './garmentCollide';

/** An inner "trouser" sheet: z = 0 plane over x, y in [0, 1], the body lies behind it (z < 0). */
function innerSheet(flipWinding = false): Surface {
  const positions = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0];
  const bound = positions.map((c, i) => (i % 3 === 2 ? -0.01 : c));
  const index = flipWinding ? [0, 2, 1, 0, 3, 2] : [0, 1, 2, 0, 2, 3];
  return { positions, index, bound };
}

const upper = (z: number, x = 0.5, y = 0.5): Float32Array => Float32Array.from([x, y, z]);

describe('pushOutside', () => {
  it('pushes an outer vertex that sits inside the inner garment out to the margin', () => {
    const p = upper(-0.006);
    expect(pushOutside(p, 1, [innerSheet()])).toBe(1);
    expect(p[2]).toBeCloseTo(0.006, 6);
    expect(p[0]).toBeCloseTo(0.5, 6);
  });

  it('works for either winding of the inner mesh (outward is decided by the body side)', () => {
    const p = upper(-0.006);
    pushOutside(p, 1, [innerSheet(true)]);
    expect(p[2]).toBeCloseTo(0.006, 6);
  });

  it('leaves vertices that are already outside, far behind, or beside the footprint alone', () => {
    const outside = upper(0.01);
    const deep = upper(-0.1);
    const beside = upper(-0.005, 3, 0.5);
    for (const p of [outside, deep, beside]) {
      const before = Array.from(p);
      expect(pushOutside(p, 1, [innerSheet()])).toBe(0);
      expect(Array.from(p)).toEqual(before);
    }
  });

  it('moves only the offending vertices of a batch', () => {
    const p = Float32Array.from([0.2, 0.2, -0.004, 0.6, 0.6, 0.02, 0.8, 0.3, -0.009]);
    expect(pushOutside(p, 3, [innerSheet()])).toBe(2);
    expect(p[2]).toBeCloseTo(0.006, 6);
    expect(p[5]).toBeCloseTo(0.02, 6);
    expect(p[8]).toBeCloseTo(0.006, 6);
  });
});
