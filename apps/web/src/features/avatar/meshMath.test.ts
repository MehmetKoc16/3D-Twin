import { describe, expect, it } from 'vitest';
import { buildWeldMap, computeWeldedNormals, headsFromJoints, restLocalPositions } from './meshMath';

/** Two triangles folded along the shared edge (0,0,0)-(1,0,0), built with DUPLICATED edge vertices (a seam). */
const positions = Float32Array.from([
  // triangle A in the z = 0 plane, facing +z (ccw seen from +z)
  0, 0, 0, 1, 0, 0, 0.5, 1, 0,
  // triangle B, folded up 90 degrees: edge vertices duplicated, apex at z = -1 above the edge... facing +y
  0, 0, 0, 1, 0, 0, 0.5, 0, -1,
]);
const indices = [0, 1, 2, 3, 4, 5];

describe('buildWeldMap', () => {
  it('groups vertices with identical positions', () => {
    const w = buildWeldMap(positions);
    expect(w.groupCount).toBe(4);
    expect(w.groupOf[0]).toBe(w.groupOf[3]);
    expect(w.groupOf[1]).toBe(w.groupOf[4]);
    expect(w.groupOf[2]).not.toBe(w.groupOf[5]);
    const g = w.groupOf[0]!;
    expect(Array.from(w.members.slice(w.start[g], w.start[g + 1]))).toEqual([0, 3]);
  });
});

describe('computeWeldedNormals', () => {
  it('gives seam duplicates the same normal, averaged over both faces', () => {
    const w = buildWeldMap(positions);
    const out = new Float32Array(positions.length);
    computeWeldedNormals(positions, indices, w, out);
    for (let k = 0; k < 3; k++) {
      expect(out[k]).toBeCloseTo(out[9 + k]!, 6);
      expect(out[3 + k]).toBeCloseTo(out[12 + k]!, 6);
    }
    // faces have normals +z and +y (equal area), so the seam normal points between them
    const n = [out[0]!, out[1]!, out[2]!];
    expect(Math.hypot(...n)).toBeCloseTo(1, 6);
    expect(n[1]).toBeGreaterThan(0.3);
    expect(n[2]).toBeGreaterThan(0.3);
    // the apex of A keeps the pure face normal
    expect(out[8]).toBeCloseTo(1, 6);
  });
  it('without welding, seam vertices would disagree (sanity check of the helper)', () => {
    const identity = buildWeldMap(Float32Array.from(Array.from({ length: 18 }, (_, i) => i * 7.3)));
    const out = new Float32Array(positions.length);
    computeWeldedNormals(positions, indices, identity, out);
    expect(out[2]).toBeCloseTo(1, 6); // vertex 0 sees only face A
    expect(out[10]).toBeCloseTo(1, 6); // vertex 3 sees only face B (+y)
  });
});

describe('rest rebuild math', () => {
  it('computes head - parentHead, root = head', () => {
    const parents = [-1, 0, 1, 0];
    const heads = Float32Array.from([0, 0.9, 0, 0, 1.1, 0.05, 0, 1.4, 0.05, 0.1, 1.0, 0]);
    const local = restLocalPositions(parents, heads);
    expect(local[1]).toBeCloseTo(0.9);
    expect(Array.from(local.slice(3, 6)).map((x) => Math.round(x * 1e4) / 1e4)).toEqual([0, 0.2, 0.05]);
    expect(Array.from(local.slice(6, 9)).map((x) => Math.round(x * 1e4) / 1e4)).toEqual([0, 0.3, 0]);
    expect(Array.from(local.slice(9, 12)).map((x) => Math.round(x * 1e4) / 1e4)).toEqual([0.1, 0.1, 0]);
  });
  it('accumulating the local positions along the hierarchy reproduces the heads', () => {
    const parents = [-1, 0, 1, 0];
    const heads = Float32Array.from([0, 0.9, 0, 0, 1.1, 0.05, 0, 1.4, 0.05, 0.1, 1.0, 0]);
    const local = restLocalPositions(parents, heads);
    for (let i = 0; i < 4; i++) {
      const acc = [0, 0, 0];
      for (let b = i; b >= 0; b = parents[b]!) for (let k = 0; k < 3; k++) acc[k] = acc[k]! + local[b * 3 + k]!;
      for (let k = 0; k < 3; k++) expect(acc[k]).toBeCloseTo(heads[i * 3 + k]!, 5);
    }
  });
  it('maps rig-order joints to skeleton order', () => {
    const joints = Float32Array.from([1, 2, 3, 0, 0, 0, 4, 5, 6, 0, 0, 0]); // rig bone 0 and 1
    expect(Array.from(headsFromJoints(joints, [1, 0]))).toEqual([4, 5, 6, 1, 2, 3]);
  });
});
