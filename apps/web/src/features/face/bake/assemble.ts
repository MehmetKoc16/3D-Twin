import { at } from '../util';
import type { FaceMapDef } from '@dt/avatar-core';

export const MESH_VERTEX_COUNT = 468;

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
  const indices: number[] = [];
  for (const [a, b, c] of faceMap.triangles) {
    if (a < MESH_VERTEX_COUNT && b < MESH_VERTEX_COUNT && c < MESH_VERTEX_COUNT && bound[a] && bound[b] && bound[c]) {
      indices.push(a, b, c);
    }
  }
  return { positions, photoUvs, indices: Uint32Array.from(indices) };
}
