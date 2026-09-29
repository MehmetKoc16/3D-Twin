import { at } from '../util';
import type { FaceMapDef } from '@dt/avatar-core';

export const MESH_VERTEX_COUNT = 468;

/** Triangles whose UV area is below this (uv units squared) are collapsed slivers at the eye / lip holes. */
export const MIN_UV_AREA = 1e-6;

export interface BakeGeometryData {
  /** Vertex positions in output pixels (x = u * W, y = v * H, z = 0), 468 vertices. */
  positions: Float32Array;
  /** Photo texture coordinates (x, y normalized, y measured from the top) per vertex. */
  photoUvs: Float32Array;
  /** Triangle indices; only triangles whose 3 landmarks are bound in the face map. */
  indices: Uint32Array;
}

/**
 * Assembles the piecewise-affine warp mesh: each canonical triangle is placed at the body UV of its
 * landmarks and samples the photo at the detected landmark positions.
 */
export function assembleBakeGeometry(
  faceMap: FaceMapDef,
  landmarks: ArrayLike<number>,
  width: number,
  height: number,
): BakeGeometryData {
  const positions = new Float32Array(MESH_VERTEX_COUNT * 3);
  const photoUvs = new Float32Array(MESH_VERTEX_COUNT * 2);
  const bound = new Uint8Array(MESH_VERTEX_COUNT);
  for (const lm of faceMap.landmarks) {
    if (lm.index < 0 || lm.index >= MESH_VERTEX_COUNT) continue;
    positions[lm.index * 3] = lm.uv[0] * width;
    positions[lm.index * 3 + 1] = lm.uv[1] * height;
    photoUvs[lm.index * 2] = at(landmarks, lm.index * 3);
    photoUvs[lm.index * 2 + 1] = at(landmarks, lm.index * 3 + 1);
    bound[lm.index] = 1;
  }
  const uvOf = new Map<number, [number, number]>();
  for (const lm of faceMap.landmarks) uvOf.set(lm.index, lm.uv);
  const candidates: { tri: [number, number, number]; area: number }[] = [];
  let positive = 0;
  let negative = 0;
  for (const tri of faceMap.triangles) {
    const [a, b, c] = tri;
    if (!(a < MESH_VERTEX_COUNT && b < MESH_VERTEX_COUNT && c < MESH_VERTEX_COUNT && bound[a] && bound[b] && bound[c])) continue;
    const pa = uvOf.get(a);
    const pb = uvOf.get(b);
    const pc = uvOf.get(c);
    if (!pa || !pb || !pc) continue;
    const area = 0.5 * ((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pc[0] - pa[0]) * (pb[1] - pa[1]));
    if (Math.abs(area) < MIN_UV_AREA) continue; // collapsed sliver
    if (area > 0) positive += 1;
    else negative += 1;
    candidates.push({ tri, area });
  }
  // Canonical triangles keep one UV orientation; the minority sign are flipped slivers at the eye / lip holes.
  const majority = positive >= negative ? 1 : -1;
  const indices: number[] = [];
  for (const { tri, area } of candidates) if (Math.sign(area) === majority) indices.push(tri[0], tri[1], tri[2]);
  return { positions, photoUvs, indices: Uint32Array.from(indices) };
}
