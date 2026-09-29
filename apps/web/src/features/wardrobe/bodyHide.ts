/** Hiding body triangles under worn garments (pure, no three.js). */

interface WeldLike {
  groupOf: ArrayLike<number>;
  start: ArrayLike<number>;
  members: ArrayLike<number>;
}

/**
 * Marks the deleted body render vertices (union of all worn items) and every UV-seam copy of them: the pipeline
 * lists MakeHuman vertices, but the render mesh holds one copy per UV island.
 */
export function hiddenVertexMask(
  deleteLists: readonly ArrayLike<number>[],
  vertexCount: number,
  weld?: WeldLike,
): Uint8Array {
  const mask = new Uint8Array(vertexCount);
  for (const list of deleteLists) {
    for (let i = 0; i < list.length; i++) {
      const v = list[i]!;
      if (v >= vertexCount) continue;
      if (weld) {
        const g = weld.groupOf[v]!;
        for (let m = weld.start[g]!; m < weld.start[g + 1]!; m++) mask[weld.members[m]!] = 1;
      } else {
        mask[v] = 1;
      }
    }
  }
  return mask;
}

/** Body triangle list without the triangles whose three vertices are all hidden. Returns `indices` itself if none. */
export function filterBodyIndex(indices: Uint32Array, hidden: Uint8Array): Uint32Array {
  let kept = 0;
  for (let t = 0; t < indices.length; t += 3) {
    if (!(hidden[indices[t]!] && hidden[indices[t + 1]!] && hidden[indices[t + 2]!])) kept += 3;
  }
  if (kept === indices.length) return indices;
  const out = new Uint32Array(kept);
  let o = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t]!;
    const b = indices[t + 1]!;
    const c = indices[t + 2]!;
    if (hidden[a] && hidden[b] && hidden[c]) continue;
    out[o++] = a;
    out[o++] = b;
    out[o++] = c;
  }
  return out;
}

/** Parses a `.delete.bin` (little-endian uint32 list). */
export function parseDeleteVerts(buffer: ArrayBuffer): Uint32Array {
  if (buffer.byteLength % 4 !== 0) throw new Error('garment delete list: byte length must be a multiple of 4');
  const view = new DataView(buffer);
  const out = new Uint32Array(buffer.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getUint32(i * 4, true);
  return out;
}

/** Barycentric slack of the footprint test (fraction of a triangle). */
const SLACK = 0.3;

export interface CoverOptions {
  /** Largest distance of a body vertex from the garment surface (either side) that counts as covered, metres. */
  bandM?: number;
  /** Grid cell size, metres. */
  cellM?: number;
}

/**
 * Body vertices that lie under the garment's footprint: the vertex projects inside a garment triangle (along that
 * triangle's normal) and is at most `bandM` away from its plane. MakeHuman delete lists leave small islands (the navel
 * ring) whose triangles poke through the fabric between sparse garment vertices. A body triangle whose three
 * vertices are all covered can be dropped; triangles at the garment edge always keep an uncovered vertex, so no hole
 * opens next to the hem. Only garment surfaces within a centimetre or so count: farther fabric (loose clothes) hides
 * the skin by itself. Returns one byte per body vertex.
 */
export function coveredBodyVertices(
  body: ArrayLike<number>,
  vertexCount: number,
  garment: ArrayLike<number>,
  garmentIndex: ArrayLike<number>,
  options: CoverOptions = {},
): Uint8Array {
  const band = options.bandM ?? 0.02;
  const cell = options.cellM ?? 0.03;
  const key = (ix: number, iy: number, iz: number): number => ((ix + 1024) * 2048 + (iy + 1024)) * 2048 + (iz + 1024);
  const grid = new Map<number, number[]>();
  for (let v = 0; v < vertexCount; v++) {
    const k = key(Math.floor(body[v * 3]! / cell), Math.floor(body[v * 3 + 1]! / cell), Math.floor(body[v * 3 + 2]! / cell));
    const list = grid.get(k);
    if (list) list.push(v);
    else grid.set(k, [v]);
  }
  const covered = new Uint8Array(vertexCount);
  const triangleCount = Math.floor(garmentIndex.length / 3);
  for (let t = 0; t < triangleCount; t++) {
    const i0 = garmentIndex[t * 3]! * 3;
    const i1 = garmentIndex[t * 3 + 1]! * 3;
    const i2 = garmentIndex[t * 3 + 2]! * 3;
    const e1x = garment[i1]! - garment[i0]!;
    const e1y = garment[i1 + 1]! - garment[i0 + 1]!;
    const e1z = garment[i1 + 2]! - garment[i0 + 2]!;
    const e2x = garment[i2]! - garment[i0]!;
    const e2y = garment[i2 + 1]! - garment[i0 + 1]!;
    const e2z = garment[i2 + 2]! - garment[i0 + 2]!;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const area2 = Math.hypot(nx, ny, nz);
    if (area2 < 1e-12) continue;
    nx /= area2;
    ny /= area2;
    nz /= area2;
    const d11 = e1x * e1x + e1y * e1y + e1z * e1z;
    const d12 = e1x * e2x + e1y * e2y + e1z * e2z;
    const d22 = e2x * e2x + e2y * e2y + e2z * e2z;
    const det = d11 * d22 - d12 * d12;
    if (Math.abs(det) < 1e-18) continue;
    const lo = [
      Math.min(garment[i0]!, garment[i1]!, garment[i2]!) - band,
      Math.min(garment[i0 + 1]!, garment[i1 + 1]!, garment[i2 + 1]!) - band,
      Math.min(garment[i0 + 2]!, garment[i1 + 2]!, garment[i2 + 2]!) - band,
    ];
    const hi = [
      Math.max(garment[i0]!, garment[i1]!, garment[i2]!) + band,
      Math.max(garment[i0 + 1]!, garment[i1 + 1]!, garment[i2 + 1]!) + band,
      Math.max(garment[i0 + 2]!, garment[i1 + 2]!, garment[i2 + 2]!) + band,
    ];
    for (let ix = Math.floor(lo[0]! / cell); ix <= Math.floor(hi[0]! / cell); ix++)
      for (let iy = Math.floor(lo[1]! / cell); iy <= Math.floor(hi[1]! / cell); iy++)
        for (let iz = Math.floor(lo[2]! / cell); iz <= Math.floor(hi[2]! / cell); iz++) {
          const list = grid.get(key(ix, iy, iz));
          if (!list) continue;
          for (const v of list) {
            if (covered[v] === 1) continue;
            const px = body[v * 3]! - garment[i0]!;
            const py = body[v * 3 + 1]! - garment[i0 + 1]!;
            const pz = body[v * 3 + 2]! - garment[i0 + 2]!;
            if (Math.abs(px * nx + py * ny + pz * nz) > band) continue;
            // barycentric coordinates of the projection onto the triangle plane
            const p1 = px * e1x + py * e1y + pz * e1z;
            const p2 = px * e2x + py * e2y + pz * e2z;
            const u = (d22 * p1 - d12 * p2) / det;
            const w = (d11 * p2 - d12 * p1) / det;
            // a little slack closes the wedges between the prisms of neighbouring triangles at folds
            if (u >= -SLACK && w >= -SLACK && u + w <= 1 + SLACK) covered[v] = 1;
          }
        }
  }
  return covered;
}
