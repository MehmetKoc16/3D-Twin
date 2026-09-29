/** Pure mesh / skeleton math for the avatar bridge (no three.js, unit tested). */

/** Groups of vertices that share the same rest position (UV-seam duplicates). */
export interface WeldMap {
  /** groupOf[v] = group id of vertex v. */
  groupOf: Uint32Array;
  /** CSR: members of group g are members[start[g] .. start[g + 1]). */
  start: Uint32Array;
  members: Uint32Array;
  groupCount: number;
}

/**
 * Builds a weld map from the rest positions: vertices at (nearly) the same position form one group. The UV-seam
 * split duplicates a vertex with an identical position, and morph deltas are duplicated too, so groups stay
 * coincident after every morph and only need to be found once.
 */
export function buildWeldMap(positions: ArrayLike<number>, quantum = 1e-5): WeldMap {
  const n = Math.floor(positions.length / 3);
  const groupOf = new Uint32Array(n);
  const ids = new Map<string, number>();
  let groupCount = 0;
  const inv = 1 / quantum;
  for (let v = 0; v < n; v++) {
    const key = `${Math.round(positions[v * 3]! * inv)},${Math.round(positions[v * 3 + 1]! * inv)},${Math.round(positions[v * 3 + 2]! * inv)}`;
    let g = ids.get(key);
    if (g === undefined) {
      g = groupCount++;
      ids.set(key, g);
    }
    groupOf[v] = g;
  }
  const start = new Uint32Array(groupCount + 1);
  for (let v = 0; v < n; v++) start[groupOf[v]! + 1] = start[groupOf[v]! + 1]! + 1;
  for (let g = 0; g < groupCount; g++) start[g + 1] = start[g + 1]! + start[g]!;
  const fill = start.slice(0, groupCount);
  const members = new Uint32Array(n);
  for (let v = 0; v < n; v++) {
    const g = groupOf[v]!;
    members[fill[g]!] = v;
    fill[g] = fill[g]! + 1;
  }
  return { groupOf, start, members, groupCount };
}

/**
 * Area-weighted smooth vertex normals, averaged across welded duplicates so UV seams do not show.
 * Writes 3 floats per vertex into `out` (length >= positions.length). `scratch` (length 3 * groupCount) is optional.
 */
export function computeWeldedNormals(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  weld: WeldMap,
  out: Float32Array,
  scratch: Float32Array = new Float32Array(weld.groupCount * 3),
): void {
  scratch.fill(0);
  const { groupOf } = weld;
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t]!;
    const b = indices[t + 1]!;
    const c = indices[t + 2]!;
    const ax = positions[a * 3]!;
    const ay = positions[a * 3 + 1]!;
    const az = positions[a * 3 + 2]!;
    const e1x = positions[b * 3]! - ax;
    const e1y = positions[b * 3 + 1]! - ay;
    const e1z = positions[b * 3 + 2]! - az;
    const e2x = positions[c * 3]! - ax;
    const e2y = positions[c * 3 + 1]! - ay;
    const e2z = positions[c * 3 + 2]! - az;
    // cross product magnitude = 2 * area, so accumulating it directly is area weighting
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    for (const v of [a, b, c]) {
      const g = groupOf[v]! * 3;
      scratch[g] = scratch[g]! + nx;
      scratch[g + 1] = scratch[g + 1]! + ny;
      scratch[g + 2] = scratch[g + 2]! + nz;
    }
  }
  for (let g = 0; g < weld.groupCount; g++) {
    const o = g * 3;
    const len = Math.hypot(scratch[o]!, scratch[o + 1]!, scratch[o + 2]!);
    const s = len > 1e-20 ? 1 / len : 0;
    scratch[o] = scratch[o]! * s;
    scratch[o + 1] = scratch[o + 1]! * s;
    scratch[o + 2] = scratch[o + 2]! * s;
  }
  const n = Math.floor(positions.length / 3);
  for (let v = 0; v < n; v++) {
    const g = groupOf[v]! * 3;
    out[v * 3] = scratch[g]!;
    out[v * 3 + 1] = scratch[g + 1]!;
    out[v * 3 + 2] = scratch[g + 2]!;
  }
}

/**
 * Local bone translations for a world-aligned skeleton (identity rest rotations): a root gets its head, every other
 * bone gets head - parentHead. `parents[i]` is the index of the parent bone or -1; `heads` holds 3 floats per bone.
 */
export function restLocalPositions(parents: ArrayLike<number>, heads: ArrayLike<number>): Float32Array {
  const n = parents.length;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const p = parents[i]!;
    for (let k = 0; k < 3; k++) out[i * 3 + k] = heads[i * 3 + k]! - (p >= 0 ? heads[p * 3 + k]! : 0);
  }
  return out;
}

/** Extracts the head positions (3 floats per bone) from a joints array with 6 floats per bone. */
export function headsFromJoints(joints: ArrayLike<number>, order: ArrayLike<number>): Float32Array {
  const out = new Float32Array(order.length * 3);
  for (let i = 0; i < order.length; i++) {
    const j = order[i]! * 6;
    out[i * 3] = joints[j]!;
    out[i * 3 + 1] = joints[j + 1]!;
    out[i * 3 + 2] = joints[j + 2]!;
  }
  return out;
}
