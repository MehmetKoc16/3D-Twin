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
  /**
   * Normal orientation strategy: 'triangle' votes per-triangle on outward direction (default),
   * 'surface' votes globally / per connected component (opt-in for outer garments).
   */
  orientation?: 'triangle' | 'surface';
  /** Preserve the front opening with a local, smoothly blended +Z contact direction. */
  frontDepthOnly?: boolean;
}

/**
 * Clearance the outer layer (e.g. denim jacket) keeps outside of the inner garments it is worn over (metres).
 * Outer garments need a larger clearance (12 mm vs default 6 mm) over inner garments (such as polo shirts)
 * to prevent inner fabric patches from showing through near welt pockets and side seams, while keeping
 * lighter outfits clean.
 */
export const OUTER_PUSH_MARGIN_M = 0.012;

/** Barycentric slack of the footprint test (fraction of a triangle). */
const SLACK = 0.2;
/** A vertex is only moved when it needs more than this (metres): float noise at exactly the margin is not a push. */
const MIN_PUSH = 1e-4;

function smoothstep(value: number): number {
  const t = Math.max(0, Math.min(1, value));
  return t * t * (3 - 2 * t);
}

/**
 * Computes consistently oriented outward triangle normals for a garment surface.
 * Rather than voting per-triangle on `position - bound` (which is noisy on tight garments),
 * this computes area-weighted vertex normals from mesh winding, evaluates orientation
 * with a weighted majority vote over connected components (and fallback to the global vote),
 * and flips triangle normals to agree with their vertices' oriented outward direction.
 */
export function orientSurfaceNormals(
  positions: ArrayLike<number>,
  index: ArrayLike<number>,
  bound: ArrayLike<number>,
): Float32Array {
  const triangleCount = Math.floor(index.length / 3);
  const vertexCount = Math.min(
    Math.floor(positions.length / 3),
    Math.floor(bound.length / 3),
  );
  const normals = new Float32Array(triangleCount * 3);
  if (triangleCount === 0 || vertexCount === 0) return normals;

  // 1. Accumulate unnormalized face normals into vertices (area-weighted, since |cross| = 2 * area).
  const vertexNormals = new Float32Array(vertexCount * 3);
  for (let t = 0; t < triangleCount; t++) {
    const i0 = index[t * 3]!;
    const i1 = index[t * 3 + 1]!;
    const i2 = index[t * 3 + 2]!;
    if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) continue;
    const e1x = positions[i1 * 3]! - positions[i0 * 3]!;
    const e1y = positions[i1 * 3 + 1]! - positions[i0 * 3 + 1]!;
    const e1z = positions[i1 * 3 + 2]! - positions[i0 * 3 + 2]!;
    const e2x = positions[i2 * 3]! - positions[i0 * 3]!;
    const e2y = positions[i2 * 3 + 1]! - positions[i0 * 3 + 1]!;
    const e2z = positions[i2 * 3 + 2]! - positions[i0 * 3 + 2]!;
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    const len = Math.hypot(nx, ny, nz);
    if (len < 1e-12) continue;
    normals[t * 3] = nx / len;
    normals[t * 3 + 1] = ny / len;
    normals[t * 3 + 2] = nz / len;

    vertexNormals[i0 * 3] = vertexNormals[i0 * 3]! + nx;
    vertexNormals[i0 * 3 + 1] = vertexNormals[i0 * 3 + 1]! + ny;
    vertexNormals[i0 * 3 + 2] = vertexNormals[i0 * 3 + 2]! + nz;
    vertexNormals[i1 * 3] = vertexNormals[i1 * 3]! + nx;
    vertexNormals[i1 * 3 + 1] = vertexNormals[i1 * 3 + 1]! + ny;
    vertexNormals[i1 * 3 + 2] = vertexNormals[i1 * 3 + 2]! + nz;
    vertexNormals[i2 * 3] = vertexNormals[i2 * 3]! + nx;
    vertexNormals[i2 * 3 + 1] = vertexNormals[i2 * 3 + 1]! + ny;
    vertexNormals[i2 * 3 + 2] = vertexNormals[i2 * 3 + 2]! + nz;
  }

  // 2. Normalize vertex normals and compute per-vertex outward vote weighted by clearance.
  for (let v = 0; v < vertexCount; v++) {
    const vx = vertexNormals[v * 3]!;
    const vy = vertexNormals[v * 3 + 1]!;
    const vz = vertexNormals[v * 3 + 2]!;
    const len = Math.hypot(vx, vy, vz);
    if (len > 1e-12) {
      vertexNormals[v * 3] = vx / len;
      vertexNormals[v * 3 + 1] = vy / len;
      vertexNormals[v * 3 + 2] = vz / len;
    }
  }

  const vertexVotes = new Float32Array(vertexCount);
  const vertexWeights = new Float32Array(vertexCount);
  let globalVote = 0;
  for (let v = 0; v < vertexCount; v++) {
    const dx = positions[v * 3]! - bound[v * 3]!;
    const dy = positions[v * 3 + 1]! - bound[v * 3 + 1]!;
    const dz = positions[v * 3 + 2]! - bound[v * 3 + 2]!;
    const dist = Math.hypot(dx, dy, dz);
    const dot =
      vertexNormals[v * 3]! * dx +
      vertexNormals[v * 3 + 1]! * dy +
      vertexNormals[v * 3 + 2]! * dz;
    // Weighted by clearance so tight/noisy vertices count little.
    const vote = dot * dist;
    vertexVotes[v] = vote;
    vertexWeights[v] = dist;
    globalVote += vote;
  }

  // 3. Group vertices into connected components using Disjoint Set Union (DSU).
  const parent = new Int32Array(vertexCount);
  for (let v = 0; v < vertexCount; v++) parent[v] = v;
  const find = (v: number): number => {
    let root = v;
    while (root !== parent[root]!) root = parent[root]!;
    let curr = v;
    while (curr !== root) {
      const next = parent[curr]!;
      parent[curr] = root;
      curr = next;
    }
    return root;
  };
  for (let t = 0; t < triangleCount; t++) {
    const i0 = index[t * 3]!;
    const i1 = index[t * 3 + 1]!;
    const i2 = index[t * 3 + 2]!;
    if (i0 < vertexCount && i1 < vertexCount) {
      const r0 = find(i0);
      const r1 = find(i1);
      if (r0 !== r1) parent[r0] = r1;
    }
    if (i1 < vertexCount && i2 < vertexCount) {
      const r1 = find(i1);
      const r2 = find(i2);
      if (r1 !== r2) parent[r1] = r2;
    }
  }

  // 4. Sum votes per connected component.
  const compVotes = new Map<number, number>();
  const compWeights = new Map<number, number>();
  for (let v = 0; v < vertexCount; v++) {
    const root = find(v);
    compVotes.set(root, (compVotes.get(root) ?? 0) + vertexVotes[v]!);
    compWeights.set(root, (compWeights.get(root) ?? 0) + vertexWeights[v]!);
  }

  // 5. Flip triangle normals to agree with their vertices' oriented outward normals.
  const globalSign = globalVote < 0 ? -1 : 1;
  for (let t = 0; t < triangleCount; t++) {
    let nx = normals[t * 3]!;
    let ny = normals[t * 3 + 1]!;
    let nz = normals[t * 3 + 2]!;
    if (nx === 0 && ny === 0 && nz === 0) continue;
    const i0 = index[t * 3]!;
    const i1 = index[t * 3 + 1]!;
    const i2 = index[t * 3 + 2]!;
    const root = i0 < vertexCount ? find(i0) : -1;
    const cv = root >= 0 ? compVotes.get(root) : undefined;
    const cw = root >= 0 ? compWeights.get(root) : undefined;
    const sign =
      cv !== undefined && cw !== undefined && cw > 1e-12 && Math.abs(cv) > 1e-12
        ? cv < 0
          ? -1
          : 1
        : globalSign;

    let vx = 0;
    let vy = 0;
    let vz = 0;
    if (i0 < vertexCount) {
      vx += vertexNormals[i0 * 3]!;
      vy += vertexNormals[i0 * 3 + 1]!;
      vz += vertexNormals[i0 * 3 + 2]!;
    }
    if (i1 < vertexCount) {
      vx += vertexNormals[i1 * 3]!;
      vy += vertexNormals[i1 * 3 + 1]!;
      vz += vertexNormals[i1 * 3 + 2]!;
    }
    if (i2 < vertexCount) {
      vx += vertexNormals[i2 * 3]!;
      vy += vertexNormals[i2 * 3 + 1]!;
      vz += vertexNormals[i2 * 3 + 2]!;
    }
    const align = nx * vx + ny * vy + nz * vz;
    const shouldFlip = Math.abs(align) > 1e-12 ? align * sign < 0 : sign < 0;
    if (shouldFlip) {
      nx = -nx;
      ny = -ny;
      nz = -nz;
    }
    normals[t * 3] = nx;
    normals[t * 3 + 1] = ny;
    normals[t * 3 + 2] = nz;
  }

  return normals;
}

/**
 * Computes per-triangle outward normals by voting locally on `position - bound`.
 * Fast and robust on partial / clipped meshes (e.g. waistband boundary in tuck tests),
 * used by default for non-outer garments.
 */
export function triangleSurfaceNormals(
  positions: ArrayLike<number>,
  index: ArrayLike<number>,
  bound: ArrayLike<number>,
): Float32Array {
  const triangleCount = Math.floor(index.length / 3);
  const vertexCount = Math.min(
    Math.floor(positions.length / 3),
    Math.floor(bound.length / 3),
  );
  const normals = new Float32Array(triangleCount * 3);
  for (let t = 0; t < triangleCount; t++) {
    const i0 = index[t * 3]!;
    const i1 = index[t * 3 + 1]!;
    const i2 = index[t * 3 + 2]!;
    if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) continue;
    const e1x = positions[i1 * 3]! - positions[i0 * 3]!;
    const e1y = positions[i1 * 3 + 1]! - positions[i0 * 3 + 1]!;
    const e1z = positions[i1 * 3 + 2]! - positions[i0 * 3 + 2]!;
    const e2x = positions[i2 * 3]! - positions[i0 * 3]!;
    const e2y = positions[i2 * 3 + 1]! - positions[i0 * 3 + 1]!;
    const e2z = positions[i2 * 3 + 2]! - positions[i0 * 3 + 2]!;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const len = Math.hypot(nx, ny, nz);
    if (len < 1e-12) continue;
    nx /= len;
    ny /= len;
    nz /= len;
    let ox = 0;
    let oy = 0;
    let oz = 0;
    for (const v of [i0, i1, i2]) {
      ox += positions[v * 3]! - bound[v * 3]!;
      oy += positions[v * 3 + 1]! - bound[v * 3 + 1]!;
      oz += positions[v * 3 + 2]! - bound[v * 3 + 2]!;
    }
    if (ox * nx + oy * ny + oz * nz < 0) {
      nx = -nx;
      ny = -ny;
      nz = -nz;
    }
    normals[t * 3] = nx;
    normals[t * 3 + 1] = ny;
    normals[t * 3 + 2] = nz;
  }
  return normals;
}

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
  const orientation = options.orientation ?? 'triangle';
  const key = (ix: number, iy: number, iz: number): number =>
    ((ix + 1024) * 2048 + (iy + 1024)) * 2048 + (iz + 1024);
  const passes = Math.max(1, options.passes ?? 5);
  const shiftX = new Float32Array(vertexCount);
  const shiftY = new Float32Array(vertexCount);
  const shiftZ = new Float32Array(vertexCount);
  const best = new Float32Array(vertexCount);
  const everMoved = new Uint8Array(vertexCount);
  // Freeze the opening band before any passes: contacts must not change direction
  // just because an earlier pass moved the vertex across a band boundary.
  const openingWeights = options.frontDepthOnly ? new Float32Array(vertexCount) : null;
  if (openingWeights) {
    for (let v = 0; v < vertexCount; v++) {
      // Full protection within 8 cm of the midline, fading to normal contact by 10 cm.
      openingWeights[v] = smoothstep((0.1 - Math.abs(positions[v * 3]!)) / 0.02);
    }
  }

  interface Prepared {
    surface: Surface;
    normals: Float32Array;
    grid: Map<number, number[]>;
  }
  const prepared: Prepared[] = [];
  for (const surface of surfaces) {
    const { positions: sp, index, bound } = surface;
    const triangleCount = Math.floor(index.length / 3);
    const normals =
      orientation === 'surface'
        ? orientSurfaceNormals(sp, index, bound)
        : triangleSurfaceNormals(sp, index, bound);
    const grid = new Map<number, number[]>();
    for (let t = 0; t < triangleCount; t++) {
      const nx = normals[t * 3]!;
      const ny = normals[t * 3 + 1]!;
      const nz = normals[t * 3 + 2]!;
      if (nx === 0 && ny === 0 && nz === 0) continue;
      const i0 = index[t * 3]!;
      const i1 = index[t * 3 + 1]!;
      const i2 = index[t * 3 + 2]!;
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
          const blend = (openingWeights?.[v] ?? 0) * smoothstep((nz - 0.5) / 0.2);
          const directionX = (1 - blend) * nx;
          const directionY = (1 - blend) * ny;
          const directionZ = (1 - blend) * nz + blend;
          // Solve the same normal clearance along the blended direction. Limit amplification
          // to 25%; grazing/side contacts retain the normal direction instead of dividing by nz.
          const scale = need * Math.min(1.25, 1 / (1 - blend + blend * nz));
          const distance = blend > 0 ? scale * Math.hypot(directionX, directionY, directionZ) : need;
          if (distance > best[v]!) {
            best[v] = distance;
            shiftX[v] = directionX * scale;
            shiftY[v] = directionY * scale;
            shiftZ[v] = directionZ * scale;
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
