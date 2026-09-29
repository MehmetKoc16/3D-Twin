import type { MeasureDef } from './contracts';
import { AvatarCoreError } from './errors';

/**
 * Reusable buffers for the convex-hull tape measure. `order` persists between calls: when the loop moves only
 * slightly (finite-difference Jacobians, LM steps) the previous x-order is nearly sorted again, so the
 * insertion sort is ~O(n).
 */
export class LoopWorkspace {
  private px: Float64Array;
  private py: Float64Array;
  private order: Int32Array;
  private hull: Int32Array;
  private n = -1;

  constructor(capacity: number) {
    this.px = new Float64Array(capacity);
    this.py = new Float64Array(capacity);
    this.order = new Int32Array(capacity);
    this.hull = new Int32Array(2 * capacity + 2);
  }

  /**
   * Perimeter of the 2-D convex hull of the loop points projected on the loop's plane. Point i of the loop is
   * `coords[3 * idx[i]..]` for i in [start, end). The plane normal comes from Newell's method over the ORDERED
   * loop (falls back to +Y when degenerate); the hull itself does not depend on the vertex order.
   */
  hullPerimeter(
    coords: ArrayLike<number>,
    idx: ArrayLike<number>,
    start: number,
    end: number,
  ): number {
    const n = end - start;
    if (n < 2) return 0;
    if (n > this.px.length)
      throw new AvatarCoreError('measure: loop longer than workspace capacity');
    if (n === 2) {
      const a = idx[start]! * 3;
      const b = idx[start + 1]! * 3;
      return (
        2 *
        Math.hypot(
          coords[a]! - coords[b]!,
          coords[a + 1]! - coords[b + 1]!,
          coords[a + 2]! - coords[b + 2]!,
        )
      );
    }
    // centroid
    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (let i = start; i < end; i++) {
      const o = idx[i]! * 3;
      cx += coords[o]!;
      cy += coords[o + 1]!;
      cz += coords[o + 2]!;
    }
    cx /= n;
    cy /= n;
    cz /= n;
    // Newell normal on centered coordinates
    let nx = 0;
    let ny = 0;
    let nz = 0;
    let scale = 0;
    {
      let o = idx[end - 1]! * 3;
      let ax = coords[o]! - cx;
      let ay = coords[o + 1]! - cy;
      let az = coords[o + 2]! - cz;
      for (let i = start; i < end; i++) {
        o = idx[i]! * 3;
        const bx = coords[o]! - cx;
        const by = coords[o + 1]! - cy;
        const bz = coords[o + 2]! - cz;
        nx += (ay - by) * (az + bz);
        ny += (az - bz) * (ax + bx);
        nz += (ax - bx) * (ay + by);
        scale = Math.max(scale, Math.abs(bx), Math.abs(by), Math.abs(bz));
        ax = bx;
        ay = by;
        az = bz;
      }
    }
    let nl = Math.hypot(nx, ny, nz);
    if (!(nl > 1e-12 * scale * scale)) {
      nx = 0;
      ny = 1;
      nz = 0;
      nl = 1;
    }
    nx /= nl;
    ny /= nl;
    nz /= nl;
    // in-plane basis (u, v): u = normalize(n x axis) with the axis least aligned with n
    const anx = Math.abs(nx);
    const any = Math.abs(ny);
    const anz = Math.abs(nz);
    let ux: number;
    let uy: number;
    let uz: number;
    if (anx <= any && anx <= anz) {
      ux = 0;
      uy = nz;
      uz = -ny;
    } else if (any <= anz) {
      ux = -nz;
      uy = 0;
      uz = nx;
    } else {
      ux = ny;
      uy = -nx;
      uz = 0;
    }
    const ul = Math.hypot(ux, uy, uz);
    ux /= ul;
    uy /= ul;
    uz /= ul;
    const vx = ny * uz - nz * uy;
    const vy = nz * ux - nx * uz;
    const vz = nx * uy - ny * ux;

    const { px, py, order, hull } = this;
    for (let i = 0; i < n; i++) {
      const o = idx[start + i]! * 3;
      const dx = coords[o]! - cx;
      const dy = coords[o + 1]! - cy;
      const dz = coords[o + 2]! - cz;
      px[i] = dx * ux + dy * uy + dz * uz;
      py[i] = dx * vx + dy * vy + dz * vz;
    }
    if (this.n !== n) {
      for (let i = 0; i < n; i++) order[i] = i;
      this.n = n;
    }
    // insertion sort by (px, py)
    for (let i = 1; i < n; i++) {
      const cur = order[i]!;
      const cxp = px[cur]!;
      const cyp = py[cur]!;
      let j = i - 1;
      while (j >= 0) {
        const o = order[j]!;
        const ox = px[o]!;
        if (ox > cxp || (ox === cxp && py[o]! > cyp)) {
          order[j + 1] = o;
          j--;
        } else break;
      }
      order[j + 1] = cur;
    }
    // Andrew's monotone chain
    const cross = (a: number, b: number, c: number): number =>
      (px[b]! - px[a]!) * (py[c]! - py[a]!) - (py[b]! - py[a]!) * (px[c]! - px[a]!);
    let k = 0;
    for (let i = 0; i < n; i++) {
      const p = order[i]!;
      while (k >= 2 && cross(hull[k - 2]!, hull[k - 1]!, p) <= 0) k--;
      hull[k++] = p;
    }
    const lower = k + 1;
    for (let i = n - 2; i >= 0; i--) {
      const p = order[i]!;
      while (k >= lower && cross(hull[k - 2]!, hull[k - 1]!, p) <= 0) k--;
      hull[k++] = p;
    }
    let perimeter = 0;
    for (let i = 0; i + 1 < k; i++) {
      const a = hull[i]!;
      const b = hull[i + 1]!;
      perimeter += Math.hypot(px[b]! - px[a]!, py[b]! - py[a]!);
    }
    return perimeter;
  }
}

/** Sum of consecutive segment lengths of the vertex list (optionally closing the loop). */
export function polylineLength(
  coords: ArrayLike<number>,
  idx: ArrayLike<number>,
  start: number,
  end: number,
  closed: boolean,
): number {
  let len = 0;
  let prev = idx[start]! * 3;
  for (let i = start + 1; i < end; i++) {
    const o = idx[i]! * 3;
    len += Math.hypot(
      coords[o]! - coords[prev]!,
      coords[o + 1]! - coords[prev + 1]!,
      coords[o + 2]! - coords[prev + 2]!,
    );
    prev = o;
  }
  if (closed && end - start > 2) {
    const o = idx[start]! * 3;
    len += Math.hypot(
      coords[o]! - coords[prev]!,
      coords[o + 1]! - coords[prev + 1]!,
      coords[o + 2]! - coords[prev + 2]!,
    );
  }
  return len;
}

/** Throws unless every vertex index of the definition is a valid integer < vertexCount and counts are sane. */
export function validateMeasure(def: MeasureDef, vertexCount: number): void {
  const check = (verts: readonly number[], min: number): void => {
    if (verts.length < min)
      throw new AvatarCoreError(
        `measure "${def.id}" (${def.type}): needs at least ${min} vertices`,
      );
    for (const v of verts)
      if (!Number.isInteger(v) || v < 0 || v >= vertexCount)
        throw new AvatarCoreError(
          `measure "${def.id}" (${def.type}): vertex index ${v} out of range [0, ${vertexCount})`,
        );
  };
  switch (def.type) {
    case 'circumference':
      return check(def.verts, 3);
    case 'polyline':
      return check(def.verts, 2);
    case 'distance':
      return check(def.verts, 2);
    case 'vertexHeight':
      return check([def.vert], 1);
    case 'height':
      return;
  }
}

/** Bounding-box Y extent of the first `count` vertices. */
export function bboxYRange(
  positions: ArrayLike<number>,
  count: number,
): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0, o = 1; i < count; i++, o += 3) {
    const y = positions[o]!;
    if (y < min) min = y;
    if (y > max) max = y;
  }
  return { min, max };
}

const AXIS: Record<'x' | 'y' | 'z', number> = { x: 0, y: 1, z: 2 };

/**
 * Measures a definition on a position array (flat xyz, meters). `renderVertexCount` limits the vertices used
 * for `height` and for the floor of `vertexHeight` (joint points are excluded); default = all vertices.
 * Returns meters.
 */
export function measure(
  def: MeasureDef,
  positions: ArrayLike<number>,
  renderVertexCount?: number,
): number {
  const total = Math.floor(positions.length / 3);
  validateMeasure(def, total);
  const rc = renderVertexCount ?? total;
  switch (def.type) {
    case 'circumference': {
      const ws = new LoopWorkspace(def.verts.length);
      return ws.hullPerimeter(positions, def.verts, 0, def.verts.length);
    }
    case 'polyline':
      return polylineLength(positions, def.verts, 0, def.verts.length, false);
    case 'distance': {
      const a = def.verts[0] * 3;
      const b = def.verts[1] * 3;
      if (def.axis !== undefined) {
        const k = AXIS[def.axis];
        return Math.abs(positions[a + k]! - positions[b + k]!);
      }
      return Math.hypot(
        positions[a]! - positions[b]!,
        positions[a + 1]! - positions[b + 1]!,
        positions[a + 2]! - positions[b + 2]!,
      );
    }
    case 'vertexHeight':
      return positions[def.vert * 3 + 1]! - bboxYRange(positions, rc).min;
    case 'height': {
      const r = bboxYRange(positions, rc);
      return r.max - r.min;
    }
  }
}

/**
 * Enclosed volume (m^3) of a triangle mesh via signed tetrahedra. Coordinates are taken relative to the
 * centroid of the first `vertexLimit` vertices, which keeps the result stable for slightly open meshes.
 * Winding-agnostic (absolute value). `indices` = 3 per triangle.
 */
export function meshVolumeM3(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  vertexLimit?: number,
): number {
  const total = Math.floor(positions.length / 3);
  if (indices.length % 3 !== 0)
    throw new AvatarCoreError('mesh: index count is not a multiple of 3');
  const lim = Math.min(vertexLimit ?? total, total);
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let i = 0, o = 0; i < lim; i++, o += 3) {
    cx += positions[o]!;
    cy += positions[o + 1]!;
    cz += positions[o + 2]!;
  }
  if (lim > 0) {
    cx /= lim;
    cy /= lim;
    cz /= lim;
  }
  let vol = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const ia = indices[t]!;
    const ib = indices[t + 1]!;
    const ic = indices[t + 2]!;
    if (!(ia >= 0 && ia < total && ib >= 0 && ib < total && ic >= 0 && ic < total))
      throw new AvatarCoreError(
        `mesh: triangle ${t / 3} references a vertex outside [0, ${total})`,
      );
    const a = ia * 3;
    const b = ib * 3;
    const c = ic * 3;
    const ax = positions[a]! - cx;
    const ay = positions[a + 1]! - cy;
    const az = positions[a + 2]! - cz;
    const bx = positions[b]! - cx;
    const by = positions[b + 1]! - cy;
    const bz = positions[b + 2]! - cz;
    const ccx = positions[c]! - cx;
    const ccy = positions[c + 1]! - cy;
    const ccz = positions[c + 2]! - cz;
    vol += ax * (by * ccz - bz * ccy) - ay * (bx * ccz - bz * ccx) + az * (bx * ccy - by * ccx);
  }
  return Math.abs(vol) / 6;
}

/** Body mass estimate in kg = mesh volume (L) x density (kg/L, default 1.01). */
export function estimateMassKg(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  densityKgPerL = 1.01,
  vertexLimit?: number,
): number {
  return meshVolumeM3(positions, indices, vertexLimit) * 1000 * densityKgPerL;
}
