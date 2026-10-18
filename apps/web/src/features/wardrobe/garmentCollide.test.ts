import { describe, expect, it } from 'vitest';
import {
  orientSurfaceNormals,
  pushInside,
  pushOutside,
  triangleSurfaceNormals,
  type Surface,
} from './garmentCollide';

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

  it('pushes outward even when tight fabric has noisy bound points on the wrong side', () => {
    // 4-vertex sheet in z = 0 plane.
    // In a tight garment, fabric hugs the body closely (small clearance).
    // Suppose vertex 1 has a noisy bound point slightly on the wrong side (z > 0),
    // while vertices 0 and 2 are tight (+0.5 mm), and vertex 3 has clear clearance (+10 mm).
    // For triangle 0 (vertices 0, 1, 2), the local sum of (position - bound) is negative (-0.0005 m),
    // which caused the old per-triangle code to flip the triangle normal inward into the body.
    // With consistent orientation, the sheet is correctly oriented outward (+z).
    const positions = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0];
    const bound = [
      0, 0, -0.0005, // v0: 0.5 mm outside body
      1, 0, 0.0015, // v1: noisy, 1.5 mm inside body (wrong side)
      1, 1, -0.0005, // v2: 0.5 mm outside body
      0, 1, -0.01, // v3: 10 mm outside body (clearance)
    ];
    const surface: Surface = {
      positions,
      index: [0, 1, 2, 0, 2, 3],
      bound,
    };

    // Vertex sits inside the sheet at z = -0.002 over triangle 0:
    const p1 = Float32Array.from([0.7, 0.3, -0.002]);
    expect(pushOutside(p1, 1, [surface], { orientation: 'surface' })).toBe(1);
    expect(p1[2]).toBeCloseTo(0.006, 6);
    expect(p1[0]).toBeCloseTo(0.7, 6);
    expect(p1[1]).toBeCloseTo(0.3, 6);

    // Vertex sits slightly in front of the sheet at z = +0.001 (less than margin 0.006) over triangle 0:
    // The old code pushed this inward to z = -0.006 (behind the tee)!
    const p2 = Float32Array.from([0.7, 0.3, 0.001]);
    expect(pushOutside(p2, 1, [surface], { orientation: 'surface' })).toBe(1);
    expect(p2[2]).toBeCloseTo(0.006, 6);
    expect(p2[0]).toBeCloseTo(0.7, 6);
    expect(p2[1]).toBeCloseTo(0.3, 6);
  });

  it('defaults to per-triangle vote when orientation is omitted or triangle', () => {
    const positions = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0];
    const bound = [
      0, 0, -0.0005,
      1, 0, 0.0015,
      1, 1, -0.0005,
      0, 1, -0.01,
    ];
    const surface: Surface = {
      positions,
      index: [0, 1, 2, 0, 2, 3],
      bound,
    };
    // Triangle 0 normal flips inward (-z) under per-triangle vote, pushing p inward to -0.006:
    const p1 = Float32Array.from([0.7, 0.3, 0.001]);
    expect(pushOutside(p1, 1, [surface])).toBe(1);
    expect(p1[2]).toBeCloseTo(-0.006, 6);

    const p2 = Float32Array.from([0.7, 0.3, 0.001]);
    expect(pushOutside(p2, 1, [surface], { orientation: 'triangle' })).toBe(1);
    expect(p2[2]).toBeCloseTo(-0.006, 6);
  });
});

describe('triangleSurfaceNormals', () => {
  it('votes per-triangle on outward normal using position - bound', () => {
    const positions = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0];
    const bound = [
      0, 0, -0.0005,
      1, 0, 0.0015,
      1, 1, -0.0005,
      0, 1, -0.01,
    ];
    const index = [0, 1, 2, 0, 2, 3];
    const normals = triangleSurfaceNormals(positions, index, bound);
    // Triangle 0 normal is flipped to (0, 0, -1), triangle 1 stays (0, 0, 1)
    expect(normals[2]).toBeCloseTo(-1, 6);
    expect(normals[5]).toBeCloseTo(1, 6);
  });
});

describe('orientSurfaceNormals', () => {
  it('orients surface normals consistently outward despite noisy bound points', () => {
    const positions = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0];
    const bound = [
      0, 0, -0.0005,
      1, 0, 0.0015,
      1, 1, -0.0005,
      0, 1, -0.01,
    ];
    const index = [0, 1, 2, 0, 2, 3];
    const normals = orientSurfaceNormals(positions, index, bound);
    // Both triangles should have outward normal (0, 0, 1)
    expect(normals[0]).toBeCloseTo(0, 6);
    expect(normals[1]).toBeCloseTo(0, 6);
    expect(normals[2]).toBeCloseTo(1, 6);
    expect(normals[3]).toBeCloseTo(0, 6);
    expect(normals[4]).toBeCloseTo(0, 6);
    expect(normals[5]).toBeCloseTo(1, 6);
  });
});

describe('pushInside', () => {
  it('moves a vertex in front of the inner surface (or less than the margin behind it) to the margin behind it', () => {
    const p = Float32Array.from([0.5, 0.5, 0.02, 0.4, 0.4, -0.002, 0.6, 0.6, -0.05]);
    expect(pushInside(p, 3, [innerSheet()])).toBe(2);
    expect(p[2]).toBeCloseTo(-0.006, 6);
    expect(p[5]).toBeCloseTo(-0.006, 6);
    expect(p[8]).toBeCloseTo(-0.05, 6); // already well behind
  });
});
