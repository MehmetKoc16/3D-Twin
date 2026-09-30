/** Garment-over-garment layering: pushes the vertices of an outer garment out of an inner one (pure, no three.js). */

export interface Surface {
  /** Graded vertex positions of the inner garment (3 floats per vertex). */
  positions: ArrayLike<number>;
  index: ArrayLike<number>;
  /** The body point each vertex is bound to (3 floats per vertex): `position - bound` points away from the body. */
  bound: ArrayLike<number>;
}

export interface PushOptions {
  /** Fabric closer than this to the inner surface (either side) is considered, metres. */
  bandM?: number;
  /** Clearance the outer garment keeps outside of the inner one, metres. */
  marginM?: number;
  cellM?: number;
  /**
   * Push passes (default 5). A vertex pushed out along one triangle's normal can end up inside a neighbouring triangle
   * of a curved inner surface (a waistband, a hem); further passes settle those. Stops early when nothing moves.
   */
  passes?: number;
}

/** Barycentric slack of the footprint test (fraction of a triangle). */
const SLACK = 0.2;
/** A vertex is only moved when it needs more than this (metres): float noise at exactly the margin is not a push. */
const MIN_PUSH = 1e-4;

/**
 * Moves every vertex of `positions` that lies inside the footprint of an inner surface, less than `marginM` in front
 * of it and at most `bandM` behind it, out along the inner triangle's outward normal (oriented away from the body).
 * Used for a jumper over trousers, trousers over shoes or trousers over a tucked top; returns the number of vertices
 * that were moved (in any pass).
 */
export function pushOutside(
  positions: Float32Array,
  vertexCount: number,
  surfaces: readonly Surface[],
  options: PushOptions = {},
): number {
  const band = options.bandM ?? 0.03;
  const margin = options.marginM ?? 0.006;
  const cell = options.cellM ?? 0.03;
  const key = (ix: number, iy: number, iz: number): number =>
    ((ix + 1024) * 2048 + (iy + 1024)) * 2048 + (iz + 1024);
  const passes = Math.max(1, options.passes ?? 5);
  const shiftX = new Float32Array(vertexCount);
  const shiftY = new Float32Array(vertexCount);
  const shiftZ = new Float32Array(vertexCount);
  const best = new Float32Array(vertexCount);
  const everMoved = new Uint8Array(vertexCount);

  interface Prepared {
    surface: Surface;
    normals: Float32Array;
    grid: Map<number, number[]>;
  }
  const prepared: Prepared[] = [];
  for (const surface of surfaces) {
    const { positions: sp, index, bound } = surface;
    const triangleCount = Math.floor(index.length / 3);
    const normals = new Float32Array(triangleCount * 3);
    const grid = new Map<number, number[]>();
    for (let t = 0; t < triangleCount; t++) {
      const i0 = index[t * 3]!;
      const i1 = index[t * 3 + 1]!;
      const i2 = index[t * 3 + 2]!;
      const e1x = sp[i1 * 3]! - sp[i0 * 3]!;
      const e1y = sp[i1 * 3 + 1]! - sp[i0 * 3 + 1]!;
      const e1z = sp[i1 * 3 + 2]! - sp[i0 * 3 + 2]!;
      const e2x = sp[i2 * 3]! - sp[i0 * 3]!;
      const e2y = sp[i2 * 3 + 1]! - sp[i0 * 3 + 1]!;
      const e2z = sp[i2 * 3 + 2]! - sp[i0 * 3 + 2]!;
      let nx = e1y * e2z - e1z * e2y;
      let ny = e1z * e2x - e1x * e2z;
      let nz = e1x * e2y - e1y * e2x;
      const length = Math.hypot(nx, ny, nz);
      if (length < 1e-12) continue;
      nx /= length;
      ny /= length;
      nz /= length;
      // orient away from the body: the fabric lies outside of its bound body points
      let outX = 0;
      let outY = 0;
      let outZ = 0;
      for (const i of [i0, i1, i2]) {
        outX += sp[i * 3]! - bound[i * 3]!;
        outY += sp[i * 3 + 1]! - bound[i * 3 + 1]!;
        outZ += sp[i * 3 + 2]! - bound[i * 3 + 2]!;
      }
      if (nx * outX + ny * outY + nz * outZ < 0) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
      }
      normals[t * 3] = nx;
      normals[t * 3 + 1] = ny;
      normals[t * 3 + 2] = nz;
      const lo = [0, 0, 0];
      const hi = [0, 0, 0];
      for (let k = 0; k < 3; k++) {
        const a = sp[i0 * 3 + k]!;
        const b = sp[i1 * 3 + k]!;
        const c = sp[i2 * 3 + k]!;
        lo[k] = Math.min(a, b, c) - band;
        hi[k] = Math.max(a, b, c) + band;
      }
      for (let ix = Math.floor(lo[0]! / cell); ix <= Math.floor(hi[0]! / cell); ix++)
        for (let iy = Math.floor(lo[1]! / cell); iy <= Math.floor(hi[1]! / cell); iy++)
          for (let iz = Math.floor(lo[2]! / cell); iz <= Math.floor(hi[2]! / cell); iz++) {
            const k = key(ix, iy, iz);
            const list = grid.get(k);
            if (list) list.push(t);
            else grid.set(k, [t]);
          }
    }
    prepared.push({ surface, normals, grid });
  }

  for (let pass = 0; pass < passes; pass++) {
    best.fill(0);
    for (const { surface, normals, grid } of prepared) {
      const { positions: sp, index } = surface;
      for (let v = 0; v < vertexCount; v++) {
        const px = positions[v * 3]!;
        const py = positions[v * 3 + 1]!;
        const pz = positions[v * 3 + 2]!;
        const list = grid.get(
          key(Math.floor(px / cell), Math.floor(py / cell), Math.floor(pz / cell)),
        );
        if (!list) continue;
        for (const t of list) {
          const nx = normals[t * 3]!;
          const ny = normals[t * 3 + 1]!;
          const nz = normals[t * 3 + 2]!;
          if (nx === 0 && ny === 0 && nz === 0) continue;
          const i0 = index[t * 3]!;
          const i1 = index[t * 3 + 1]!;
          const i2 = index[t * 3 + 2]!;
          const ox = px - sp[i0 * 3]!;
          const oy = py - sp[i0 * 3 + 1]!;
          const oz = pz - sp[i0 * 3 + 2]!;
          const d = ox * nx + oy * ny + oz * nz;
          if (d >= margin || d < -band) continue;
          const e1x = sp[i1 * 3]! - sp[i0 * 3]!;
          const e1y = sp[i1 * 3 + 1]! - sp[i0 * 3 + 1]!;
          const e1z = sp[i1 * 3 + 2]! - sp[i0 * 3 + 2]!;
          const e2x = sp[i2 * 3]! - sp[i0 * 3]!;
          const e2y = sp[i2 * 3 + 1]! - sp[i0 * 3 + 1]!;
          const e2z = sp[i2 * 3 + 2]! - sp[i0 * 3 + 2]!;
          const d11 = e1x * e1x + e1y * e1y + e1z * e1z;
          const d12 = e1x * e2x + e1y * e2y + e1z * e2z;
          const d22 = e2x * e2x + e2y * e2y + e2z * e2z;
          const det = d11 * d22 - d12 * d12;
          if (Math.abs(det) < 1e-18) continue;
          const p1 = ox * e1x + oy * e1y + oz * e1z;
          const p2 = ox * e2x + oy * e2y + oz * e2z;
          const u = (d22 * p1 - d12 * p2) / det;
          const w = (d11 * p2 - d12 * p1) / det;
          if (u < -SLACK || w < -SLACK || u + w > 1 + SLACK) continue;
          const need = margin - d;
          if (need > best[v]!) {
            best[v] = need;
            shiftX[v] = nx * need;
            shiftY[v] = ny * need;
            shiftZ[v] = nz * need;
          }
        }
      }
    }
    let movedInPass = 0;
    for (let v = 0; v < vertexCount; v++) {
      if (best[v]! <= MIN_PUSH) continue;
      positions[v * 3] = positions[v * 3]! + shiftX[v]!;
      positions[v * 3 + 1] = positions[v * 3 + 1]! + shiftY[v]!;
      positions[v * 3 + 2] = positions[v * 3 + 2]! + shiftZ[v]!;
      everMoved[v] = 1;
      movedInPass++;
    }
    if (movedInPass === 0) break;
  }
  let moved = 0;
  for (let v = 0; v < vertexCount; v++) moved += everMoved[v]!;
  return moved;
}

/**
 * The opposite of `pushOutside`: moves every vertex of `positions` that lies in front of (outside of) the inner
 * surface, or less than `marginM` behind it, to `marginM` behind it. Used to tuck the hem of a top into the waistband
 * of the trousers that were pushed out over it. Implemented by mirroring the surface's bound points, which flips the
 * outward orientation of every triangle.
 */
export function pushInside(
  positions: Float32Array,
  vertexCount: number,
  surfaces: readonly Surface[],
  options: PushOptions = {},
): number {
  const flipped = surfaces.map((surface): Surface => {
    const bound = new Float32Array(surface.positions.length);
    for (let i = 0; i < bound.length; i++) bound[i] = 2 * surface.positions[i]! - surface.bound[i]!;
    return { positions: surface.positions, index: surface.index, bound };
  });
  return pushOutside(positions, vertexCount, flipped, options);
}
