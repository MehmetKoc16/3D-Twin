import { describe, expect, it } from 'vitest';
import type { Vec3 } from '@dt/avatar-core';
import {
  orientSurfaceNormals,
  OUTER_PUSH_MARGIN_M,
  OUTLIER_FACTOR,
  OUTLIER_FLOOR_M,
  OUTLIER_SLACK_M,
  pushInside,
  pushOutside,
  shoulderBandWeight,
  SHOULDER_BAND_BELOW_M,
  SHOULDER_BAND_RAMP_M,
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
  function frontContact(x: number, nz: number): Surface {
    const nx = Math.sqrt(1 - nz * nz);
    const positions = [x - 0.04, 0.46, 0.04 * nx / nz, x + 0.04, 0.46, -0.04 * nx / nz, x, 0.54, 0];
    return {
      positions,
      index: [0, 1, 2],
      bound: positions.map((p, i) => p - (i % 3 === 0 ? nx : i % 3 === 2 ? nz : 0) * 0.01),
    };
  }

  it('keeps side contacts on their normal even when their normals face forward', () => {
    for (const x of [-0.15, 0.15]) {
      const surface = frontContact(x, 0.8);
      const normal = upper(-0.01, x);
      const localized = normal.slice();
      pushOutside(normal, 1, [surface]);
      pushOutside(localized, 1, [surface], { frontDepthOnly: true });
      expect(Array.from(localized)).toEqual(Array.from(normal));
      expect(localized[0]! - x).toBeGreaterThan(0.005);
    }
  });

  it('changes contact direction continuously through angular and opening-band boundaries', () => {
    const displacement = (x: number, nz: number): number[] => {
      const p = upper(-0.01, x);
      const rest = p.slice();
      pushOutside(p, 1, [frontContact(x, nz)], { frontDepthOnly: true, passes: 1 });
      return Array.from(p, (value, k) => value - rest[k]!);
    };
    for (const boundary of [0.5, 0.7]) {
      const a = displacement(0, boundary - 0.0001);
      const b = displacement(0, boundary + 0.0001);
      expect(Math.hypot(...a.map((value, k) => value - b[k]!))).toBeLessThan(0.00005);
    }
    for (const boundary of [0.06, 0.08, 0.1]) {
      const a = displacement(boundary - 0.0001, 0.8);
      const b = displacement(boundary + 0.0001, 0.8);
      expect(Math.hypot(...a.map((value, k) => value - b[k]!))).toBeLessThan(0.00005);
    }
  });

  it('bounds direction amplification while converging to the outer clearance', () => {
    for (const nz of [0.51, 0.6, 0.65, 0.7, 0.8, 0.95]) {
      const p = upper(-0.01, 0);
      const surface = frontContact(0, nz);
      const options = { frontDepthOnly: true, marginM: OUTER_PUSH_MARGIN_M };
      pushOutside(p, 1, [surface], { ...options, passes: 1 });
      const need = OUTER_PUSH_MARGIN_M + 0.01 * nz;
      expect(Math.hypot(p[0]!, p[1]! - 0.5, p[2]! + 0.01)).toBeLessThanOrEqual(1.25 * need + 1e-7);
      pushOutside(p, 1, [surface], options);
      const clearance = p[0]! * Math.sqrt(1 - nz * nz) + p[2]! * nz;
      expect(clearance).toBeGreaterThan(OUTER_PUSH_MARGIN_M - 0.0001);
    }
  });

  it('preserves an opening in X/Y while clearing alternating slanted front contacts', () => {
    const surface = (slope: number, y: number): Surface => {
      const positions = [-0.05, y, -0.05 * slope, 0.05, y, 0.05 * slope, 0, y + 0.02, 0];
      return {
        positions,
        index: [0, 1, 2],
        bound: positions.map((p, i) => (i % 3 === 2 ? p - 0.01 : p)),
      };
    };
    const surfaces = [surface(0.6, 1), surface(-0.6, 1.04)];
    const rest = new Float32Array([0, 1.005, -0.01, 0, 1.045, -0.01, 0, 1.005, -0.007]);
    const normalPush = rest.slice();
    const depthPush = rest.slice();
    const options = { orientation: 'surface' as const, marginM: OUTER_PUSH_MARGIN_M };
    pushOutside(normalPush, 3, surfaces, options);
    // Opposite triangle slopes create a visible zigzag in a straight opening.
    expect(Math.abs(normalPush[0]! - normalPush[3]!)).toBeGreaterThan(0.01);
    pushOutside(depthPush, 3, surfaces, { ...options, frontDepthOnly: true });
    for (let v = 0; v < 3; v++) {
      expect(depthPush[v * 3]).toBe(rest[v * 3]);
      expect(depthPush[v * 3 + 1]).toBe(rest[v * 3 + 1]);
      expect(depthPush[v * 3 + 2]! / Math.hypot(1, 0.6)).toBeCloseTo(OUTER_PUSH_MARGIN_M, 6);
    }
  });

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

  it('pushes an outer sheet just inside an inner sheet to >= 12 mm outside with OUTER_PUSH_MARGIN_M, ~6 mm without', () => {
    // 4-vertex outer sheet at z = -0.002 m (just inside the inner sheet at z = 0).
    const sheetWithout = Float32Array.from([
      0.2, 0.2, -0.002,
      0.8, 0.2, -0.002,
      0.8, 0.8, -0.002,
      0.2, 0.8, -0.002,
    ]);
    expect(pushOutside(sheetWithout, 4, [innerSheet()])).toBe(4);
    for (let i = 0; i < 4; i++) {
      expect(sheetWithout[i * 3 + 2]!).toBeCloseTo(0.006, 6);
    }

    // Outer garment uses orientation: 'surface' and marginM: OUTER_PUSH_MARGIN_M (0.012 m).
    const sheetWith = Float32Array.from([
      0.2, 0.2, -0.002,
      0.8, 0.2, -0.002,
      0.8, 0.8, -0.002,
      0.2, 0.8, -0.002,
    ]);
    expect(
      pushOutside(sheetWith, 4, [innerSheet()], {
        orientation: 'surface',
        marginM: OUTER_PUSH_MARGIN_M,
      }),
    ).toBe(4);
    for (let i = 0; i < 4; i++) {
      expect(sheetWith[i * 3 + 2]!).toBeCloseTo(OUTER_PUSH_MARGIN_M, 6);
      expect(sheetWith[i * 3 + 2]!).toBeGreaterThanOrEqual(0.012);
    }
  });

  it('leaves non-outer vertices between 6 mm and 12 mm untouched while pushing outer vertices to >= 12 mm', () => {
    // Vertex at z = 0.008 m: beyond default 6 mm margin, but inside outer 12 mm margin.
    const nonOuter = upper(0.008);
    expect(pushOutside(nonOuter, 1, [innerSheet()])).toBe(0);
    expect(nonOuter[2]!).toBeCloseTo(0.008, 6);

    const outer = upper(0.008);
    expect(
      pushOutside(outer, 1, [innerSheet()], {
        orientation: 'surface',
        marginM: OUTER_PUSH_MARGIN_M,
      }),
    ).toBe(1);
    expect(outer[2]!).toBeCloseTo(OUTER_PUSH_MARGIN_M, 6);
    expect(outer[2]!).toBeGreaterThanOrEqual(0.012);
  });

  it('does not push a vertex whose projection lies 1 cm beyond the edge of a downward-facing hem triangle', () => {
    // Why: prevents hem/cuff boundary spikes (e.g. vertex 887) where barycentric SLACK
    // would extrapolate contact onto downward-facing boundary triangles beyond MAX_PROJECTION_OVERHANG_M.
    const surface: Surface = {
      positions: [0, 1.0, 0, 0.05, 1.0, 0, 0, 1.0, 0.05],
      index: [0, 1, 2],
      // Body is above/inside the hem (+Y), so (position - bound) points downward (-Y):
      bound: [0, 1.01, 0, 0.05, 1.01, 0, 0, 1.01, 0.05],
    };
    // Vertex sits 5 mm above the hem (d = -0.005 m, within band), but 1 cm beyond the z = 0 edge:
    // Projection onto triangle plane is (0.02, 1.0, -0.01), which overhangs the edge z = 0 by 10 mm.
    // In barycentric coordinates, w = -0.2 is within SLACK = 0.2, but overhang 10 mm exceeds MAX_PROJECTION_OVERHANG_M (4 mm).
    const p = Float32Array.from([0.02, 1.005, -0.01]);
    const before = Array.from(p);
    expect(pushOutside(p, 1, [surface])).toBe(0);
    expect(Array.from(p)).toEqual(before);
  });

  it('pushes a vertex projecting 2 mm across an edge between two triangles', () => {
    // Why: ensures inter-triangle coverage across mesh edges is maintained when
    // overhang is within MAX_PROJECTION_OVERHANG_M (4 mm).
    const positions = [
      0, 0, 0,
      0.05, 0, 0,
      0.05, 0.1, 0,
      0.1, 0, 0,
    ];
    const surface: Surface = {
      positions,
      index: [0, 1, 2, 1, 3, 2],
      bound: positions.map((c, i) => (i % 3 === 2 ? -0.01 : c)),
    };
    // Vertex is at x = 0.052 (2 mm across the shared edge x = 0.05), inside the fabric at z = -0.002:
    const p = Float32Array.from([0.052, 0.05, -0.002]);
    expect(pushOutside(p, 1, [surface])).toBe(1);
    expect(p[2]).toBeCloseTo(0.006, 6);
    expect(p[0]).toBeCloseTo(0.052, 6);
  });

  it('allows displacement beyond margin + 12 mm when a vertex legitimately needs larger clearance', () => {
    // WHY: The total cap was removed because real garments (such as untucked sweaters over trousers)
    // legitimately require more than margin + 12 mm displacement. A vertex 2.5 cm inside the surface
    // must reach the margin (31 mm push) rather than being clamped to margin + 12 mm (18 mm).
    const surface = innerSheet();
    // Default margin (0.006 m): needs 0.006 - (-0.025) = 0.031 m push (> 18 mm).
    const p1 = upper(-0.025);
    expect(pushOutside(p1, 1, [surface])).toBe(1);
    const disp1 = p1[2]! - (-0.025);
    expect(disp1).toBeCloseTo(0.031, 6);
    expect(p1[2]!).toBeCloseTo(0.006, 6);

    // Outer margin (OUTER_PUSH_MARGIN_M = 0.012 m): needs 0.012 - (-0.025) = 0.037 m push (> 24 mm).
    const p2 = upper(-0.025);
    expect(pushOutside(p2, 1, [surface], { marginM: OUTER_PUSH_MARGIN_M })).toBe(1);
    const disp2 = p2[2]! - (-0.025);
    expect(disp2).toBeCloseTo(0.037, 6);
    expect(p2[2]!).toBeCloseTo(OUTER_PUSH_MARGIN_M, 6);
  });

  describe('outlier limiting with topology', () => {
    function makePatch(
      centerZ: number,
      surroundZ: number,
    ): { positions: Float32Array; index: number[] } {
      // 3x3 grid of vertices over [0.3, 0.7] x [0.3, 0.7]
      const positions: number[] = [];
      for (let r = 0; r < 3; r++) {
        const y = 0.3 + r * 0.2;
        for (let c = 0; c < 3; c++) {
          const x = 0.3 + c * 0.2;
          const isCenter = r === 1 && c === 1; // vertex 4
          positions.push(x, y, isCenter ? centerZ : surroundZ);
        }
      }
      // 4 quad cells = 8 triangles
      const index: number[] = [
        0, 1, 4, 0, 4, 3,
        1, 2, 5, 1, 5, 4,
        3, 4, 7, 3, 7, 6,
        4, 5, 8, 4, 8, 7,
      ];
      return { positions: Float32Array.from(positions), index };
    }

    it('limits isolated outlier vertex push to 1.5 * medianNeighbour + 3 mm (10.5 mm) while neighbours move 5 mm', () => {
      // WHY: A stray collision at an inner garment edge/opening should not spike out from the garment.
      // Clamping to max(medianNeighbour * 1.5 + 0.003, 0.004) limits the isolated vertex to 10.5 mm
      // while allowing all surrounding neighbours to reach their required 5 mm clearance.
      const { positions, index } = makePatch(-0.028, 0.007);
      const initialZ = positions.slice();
      const count = 9;
      const movedCount = pushOutside(positions, count, [innerSheet()], {
        orientation: 'surface',
        marginM: OUTER_PUSH_MARGIN_M,
        topology: { index },
        passes: 1,
      });
      expect(movedCount).toBe(9);

      // Center vertex (index 4) needed 40 mm, but is clamped to at most 1.5 * 5 mm + 3 mm = 10.5 mm:
      const isolatedDelta = positions[4 * 3 + 2]! - initialZ[4 * 3 + 2]!;
      const expectedLimit = Math.max(0.005 * OUTLIER_FACTOR + OUTLIER_SLACK_M, OUTLIER_FLOOR_M);
      expect(isolatedDelta).toBeLessThanOrEqual(expectedLimit + 1e-6);
      expect(isolatedDelta).toBeCloseTo(expectedLimit, 5);

      // All surrounding 8 vertices move by their needed 5 mm to margin:
      for (let v = 0; v < count; v++) {
        if (v === 4) continue;
        const delta = positions[v * 3 + 2]! - initialZ[v * 3 + 2]!;
        expect(delta).toBeCloseTo(0.005, 5);
        expect(positions[v * 3 + 2]!).toBeCloseTo(OUTER_PUSH_MARGIN_M, 5);
      }
    });

    it('allows consistent large pushes when the entire patch needs 20 mm', () => {
      // WHY: Legitimate bulk clearance adjustments (e.g. jacket sitting 20 mm outside a thick inner top)
      // must not be restricted when adjacent vertices agree on the large shift.
      // surroundZ = centerZ = -0.008 -> need = 0.012 - (-0.008) = 0.020 m (20 mm).
      const { positions, index } = makePatch(-0.008, -0.008);
      const initialZ = positions.slice();
      const count = 9;
      const movedCount = pushOutside(positions, count, [innerSheet()], {
        orientation: 'surface',
        marginM: OUTER_PUSH_MARGIN_M,
        topology: { index },
      });
      expect(movedCount).toBe(9);

      // Every vertex should move by the full 20 mm because its neighbours also shift by 20 mm:
      for (let v = 0; v < count; v++) {
        const delta = positions[v * 3 + 2]! - initialZ[v * 3 + 2]!;
        expect(delta).toBeCloseTo(0.020, 5);
        expect(positions[v * 3 + 2]!).toBeCloseTo(OUTER_PUSH_MARGIN_M, 5);
      }
    });

    it('preserves unconstrained outlier push when topology is omitted', () => {
      // WHY: Backward compatibility: when topology is not provided, pushOutside must behave as before,
      // allowing the isolated vertex to move the full 40 mm to the margin.
      const { positions } = makePatch(-0.028, 0.007);
      const count = 9;
      pushOutside(positions, count, [innerSheet()], {
        orientation: 'surface',
        marginM: OUTER_PUSH_MARGIN_M,
        passes: 1,
      });

      // Without topology, isolated vertex moves the full 40 mm:
      const isolatedDelta = positions[4 * 3 + 2]! - (-0.028);
      expect(isolatedDelta).toBeCloseTo(0.040, 5);
      expect(positions[4 * 3 + 2]!).toBeCloseTo(OUTER_PUSH_MARGIN_M, 5);

      // Surrounding vertices still move 5 mm:
      for (let v = 0; v < count; v++) {
        if (v === 4) continue;
        const delta = positions[v * 3 + 2]! - 0.007;
        expect(delta).toBeCloseTo(0.005, 5);
      }
    });

    it('harmonizes shift length and limits across welded duplicate vertices', () => {
      // WHY: Garment meshes duplicate vertices at UV seams. Welded copies must share the group's
      // maximum shift and limit so the UV seam does not tear open.
      // Vertex 4 and vertex 9 are coincident at (0.5, 0.5, -0.028) in the same weld group.
      const { positions: basePos, index: baseIndex } = makePatch(-0.028, 0.007);
      // Duplicate center vertex 4 as vertex 9:
      const positions = Float32Array.from([...basePos, 0.5, 0.5, -0.028]);
      // Duplicate one triangle [4, 5, 8] with [9, 5, 8]
      const index = [...baseIndex, 9, 5, 8];
      const groupOf = new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 4]); // 4 and 9 share group 4
      const count = 10;
      pushOutside(positions, count, [innerSheet()], {
        orientation: 'surface',
        marginM: OUTER_PUSH_MARGIN_M,
        topology: { index, weld: { groupOf, groupCount: 9 } },
        passes: 1,
      });

      // Both vertex 4 and vertex 9 must receive the exact same clamped push:
      expect(positions[4 * 3 + 2]!).toBeCloseTo(positions[9 * 3 + 2]!, 6);
      const expectedLimit = Math.max(0.005 * OUTLIER_FACTOR + OUTLIER_SLACK_M, OUTLIER_FLOOR_M);
      expect(positions[4 * 3 + 2]! - (-0.028)).toBeCloseTo(expectedLimit, 5);
    });
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

  it('allows pushInside to tuck vertices that sit more than margin + 12 mm in front without capping', () => {
    // WHY: pushInside flips inner surfaces to tuck outer hems inside; vertices sitting well in front
    // of the waistband must be moved to the margin behind it without an artificial displacement cap.
    const p = upper(0.025);
    expect(pushInside(p, 1, [innerSheet()])).toBe(1);
    expect(p[2]!).toBeCloseTo(-0.006, 6);
    const disp = 0.025 - p[2]!;
    expect(disp).toBeCloseTo(0.031, 6);
  });
});

describe('shoulderBandWeight', () => {
  const armL = { shoulder: [-0.2, 1.5, 0] as Vec3, elbow: [-0.4, 1.1, 0] as Vec3 };
  const armR = { shoulder: [0.2, 1.5, 0] as Vec3, elbow: [0.4, 1.1, 0] as Vec3 };
  const arms = [armL, armR];

  it('is 0 at chest height (below shoulder - SHOULDER_BAND_BELOW_M)', () => {
    const atRampStart = [0, 1.5 - SHOULDER_BAND_BELOW_M, 0];
    expect(shoulderBandWeight(atRampStart, 0, arms)).toBe(0);
    const chestBound = [0, 1.5 - 0.2, 0];
    expect(shoulderBandWeight(chestBound, 0, arms)).toBe(0);
  });

  it('is 1 at shoulder height and ramp end (shoulder - SHOULDER_BAND_BELOW_M + SHOULDER_BAND_RAMP_M)', () => {
    const atRampEnd = [0, 1.5 - SHOULDER_BAND_BELOW_M + SHOULDER_BAND_RAMP_M, 0];
    expect(shoulderBandWeight(atRampEnd, 0, arms)).toBe(1);
    const shoulderBound = [0, 1.5, 0];
    expect(shoulderBandWeight(shoulderBound, 0, arms)).toBe(1);
  });

  it('is monotonic and smooth (no jump > 0.3 between points 1 cm apart) in between', () => {
    let previousWeight = shoulderBandWeight([0, 1.5 - 0.2, 0], 0, arms);
    for (let y = 1.5 - 0.2 + 0.01; y <= 1.5 + 0.05; y += 0.01) {
      const currentWeight = shoulderBandWeight([0, y, 0], 0, arms);
      expect(currentWeight).toBeGreaterThanOrEqual(previousWeight);
      expect(currentWeight - previousWeight).toBeLessThanOrEqual(0.3);
      previousWeight = currentWeight;
    }
  });

  it('returns 0 with no arms', () => {
    expect(shoulderBandWeight([0, 1.5, 0], 0, [])).toBe(0);
  });

  it('uses the mean of two shoulders', () => {
    const asymmetricL = { shoulder: [-0.2, 1.4, 0] as Vec3, elbow: [-0.4, 1.0, 0] as Vec3 };
    const asymmetricR = { shoulder: [0.2, 1.6, 0] as Vec3, elbow: [0.4, 1.2, 0] as Vec3 };
    const asymmetricArms = [asymmetricL, asymmetricR]; // mean shoulder y = 1.5
    // At chest height relative to mean shoulder (1.5 - SHOULDER_BAND_BELOW_M = 1.36): 0
    expect(shoulderBandWeight([0, 1.5 - SHOULDER_BAND_BELOW_M, 0], 0, asymmetricArms)).toBe(0);
    // At ramp end (1.5 - SHOULDER_BAND_BELOW_M + SHOULDER_BAND_RAMP_M = 1.42) and at mean shoulder height (1.50): 1
    expect(
      shoulderBandWeight(
        [0, 1.5 - SHOULDER_BAND_BELOW_M + SHOULDER_BAND_RAMP_M, 0],
        0,
        asymmetricArms,
      ),
    ).toBe(1);
    expect(shoulderBandWeight([0, 1.5, 0], 0, asymmetricArms)).toBe(1);
    // At y = 1.40 (armL shoulder height): startY = 1.5 - SHOULDER_BAND_BELOW_M = 1.36.
    // t = (1.40 - 1.36) / SHOULDER_BAND_RAMP_M = (1.40 - 1.36) / 0.06 = 4/6 = 2/3.
    // smoothstep(2/3) = (4/9) * (3 - 4/3) = 20/27 ~= 0.7407.
    // If only armL were used, at y = 1.40 it would have been at shoulder height (1.0).
    const weightAt140 = shoulderBandWeight([0, 1.4, 0], 0, asymmetricArms);
    expect(weightAt140).toBeGreaterThan(0);
    expect(weightAt140).toBeLessThan(1);
    expect(weightAt140).toBeCloseTo(20 / 27, 4);
  });
});
