import type { GarmentBinding } from './binding';
import { GarmentSections } from './centroid';

export interface GradeRing {
  planeY: number;
  deltaCircumferenceM: number;
}

/** Minimum clearance (m) between a graded garment vertex and its bound body point, along its grading ray. */
const MIN_CLEARANCE_M = 0.002;

export interface GradeGarmentOptions {
  /**
   * Spatial smoothing kernel radius in meters. Default 0: legacy per-vertex grading (every vertex moves
   * independently). A radius > 0 (e.g. 0.015 m for open-front outer garments) smooths the displacement field
   * so co-located stacked parts (buttons, flaps, doubled plackets) move coherently instead of tearing apart.
   */
  smoothRadiusM?: number;
}

/**
 * Circular-section approximation: a circumference change Δc shifts each vertex by Δc / 2π
 * along the horizontal ray from its bound body point's limb-section centroid, clamped so the vertex
 * keeps at least 2 mm clearance from that body point. Rings interpolate linearly by body height and
 * hold their end values outside the given range.
 *
 * With `smoothRadiusM > 0` the per-vertex displacements are smoothed with a bilateral kernel (spatial
 * distance × direction agreement), so panels or limbs facing other ways do not mix. After smoothing, a
 * vertex whose smoothed shift along its own ray would leave less than 2 mm clearance gets the max of
 * (smoothed shift, its own legacy clamped shift) along that ray, so the body clearance guarantee of the
 * legacy grading still holds.
 */
export function gradeGarment(
  restPositions: ArrayLike<number>,
  binding: GarmentBinding,
  bodyPositions: ArrayLike<number>,
  rings: GradeRing[],
  out: Float32Array,
  sections: GarmentSections = new GarmentSections(bodyPositions),
  options: GradeGarmentOptions = {},
): void {
  if (restPositions.length < binding.count * 3 || out.length < binding.count * 3)
    throw new Error('garment grade: position buffer too short');
  const sorted = [...rings].sort((a, b) => a.planeY - b.planeY);
  if (sorted.some((r) => !Number.isFinite(r.planeY) || !Number.isFinite(r.deltaCircumferenceM)))
    throw new Error('garment grade: invalid ring');

  const count = binding.count;
  if (count === 0) return;

  const smoothRadiusM = options.smoothRadiusM ?? 0;
  const smooth = smoothRadiusM > 0;

  // Per-vertex data, only kept when smoothing (legacy mode writes `out` directly; buffers stay empty).
  const keep = smooth ? count : 0;
  const dirX = new Float64Array(keep);
  const dirZ = new Float64Array(keep);
  /** Own legacy shift along the ray: raw shift + 2 mm clearance extra. */
  const ownShift = new Float64Array(keep);
  /** Smallest shift along the ray that keeps 2 mm clearance. */
  const minShift = new Float64Array(keep);

  // Pass 1: horizontal ray direction and legacy per-vertex shift (raw shift + clearance clamp).
  for (let v = 0; v < count; v++) {
    const o = v * 3;
    let bx = 0,
      by = 0,
      bz = 0;
    for (let j = 0; j < 3; j++) {
      const p = binding.indices[o + j]! * 3;
      const w = binding.weights[o + j]!;
      bx += w * bodyPositions[p]!;
      by += w * bodyPositions[p + 1]!;
      bz += w * bodyPositions[p + 2]!;
    }
    const { x: cx, z: cz } = sections.centroidAt(by, bx, bz);
    let dx = bx - cx,
      dz = bz - cz;
    let radius = Math.hypot(dx, dz);
    if (radius < 1e-9) {
      dx = restPositions[o]! - cx;
      dz = restPositions[o + 2]! - cz;
      radius = Math.hypot(dx, dz);
    }
    if (radius < 1e-9) {
      dx = 1;
      dz = 0;
      radius = 1;
    }
    dx /= radius;
    dz /= radius;

    let delta = 0;
    if (sorted.length) {
      let upper = 0;
      while (upper < sorted.length && sorted[upper]!.planeY < by) upper++;
      if (upper === 0) delta = sorted[0]!.deltaCircumferenceM;
      else if (upper === sorted.length) delta = sorted[sorted.length - 1]!.deltaCircumferenceM;
      else {
        const low = sorted[upper - 1]!;
        const high = sorted[upper]!;
        const t = high.planeY === low.planeY ? 1 : (by - low.planeY) / (high.planeY - low.planeY);
        delta = low.deltaCircumferenceM + t * (high.deltaCircumferenceM - low.deltaCircumferenceM);
      }
    }
    const shift = delta / (2 * Math.PI);
    const clearance = (restPositions[o]! - bx) * dx + (restPositions[o + 2]! - bz) * dz;
    const extra = Math.max(0, MIN_CLEARANCE_M - clearance - shift);

    if (!smooth) {
      // Legacy mode: every vertex moves independently.
      out[o] = restPositions[o]! + (shift + extra) * dx;
      out[o + 1] = restPositions[o + 1]!;
      out[o + 2] = restPositions[o + 2]! + (shift + extra) * dz;
      continue;
    }
    dirX[v] = dx;
    dirZ[v] = dz;
    ownShift[v] = shift + extra;
    minShift[v] = MIN_CLEARANCE_M - clearance;
  }
  if (!smooth) return;

  // Pass 2: bilateral smoothing over rest positions (spatial distance × direction agreement).
  const cell = smoothRadiusM;
  const radiusSq = smoothRadiusM * smoothRadiusM;
  const invRadius = 1 / smoothRadiusM;

  const key = (ix: number, iy: number, iz: number): number =>
    ((ix + 1024) * 2048 + (iy + 1024)) * 2048 + (iz + 1024);
  const grid = new Map<number, number[]>();

  for (let v = 0; v < count; v++) {
    const o = v * 3;
    const ix = Math.floor(restPositions[o]! / cell);
    const iy = Math.floor(restPositions[o + 1]! / cell);
    const iz = Math.floor(restPositions[o + 2]! / cell);
    const k = key(ix, iy, iz);
    const cellList = grid.get(k);
    if (cellList) cellList.push(v);
    else grid.set(k, [v]);
  }

  // Results are buffered so `out` may alias `restPositions` (neighbours read rest positions).
  const resultX = new Float64Array(count);
  const resultZ = new Float64Array(count);

  for (let v = 0; v < count; v++) {
    const o = v * 3;
    const px = restPositions[o]!;
    const py = restPositions[o + 1]!;
    const pz = restPositions[o + 2]!;
    const ix = Math.floor(px / cell);
    const iy = Math.floor(py / cell);
    const iz = Math.floor(pz / cell);
    const vx = dirX[v]!;
    const vz = dirZ[v]!;

    let sumW = 0;
    let sumVx = 0;
    let sumVz = 0;

    for (let di = -1; di <= 1; di++) {
      for (let dj = -1; dj <= 1; dj++) {
        for (let dk = -1; dk <= 1; dk++) {
          const list = grid.get(key(ix + di, iy + dj, iz + dk));
          if (!list) continue;
          for (let i = 0; i < list.length; i++) {
            const u = list[i]!;
            const uo = u * 3;
            const qx = restPositions[uo]!;
            const qy = restPositions[uo + 1]!;
            const qz = restPositions[uo + 2]!;
            const distSq = (qx - px) ** 2 + (qy - py) ** 2 + (qz - pz) ** 2;
            if (distSq > radiusSq) continue;

            const ux = dirX[u]!;
            const uz = dirZ[u]!;
            const dot = vx * ux + vz * uz;
            if (dot < 0.5) continue;

            let dirW = 1;
            if (dot < 0.9) {
              const s = (dot - 0.5) / 0.4;
              dirW = s * s * (3 - 2 * s);
            }

            const dist = Math.sqrt(distSq);
            const t = 1 - dist * invRadius;
            const w = t * t * dirW;

            sumW += w;
            sumVx += w * ownShift[u]! * ux;
            sumVz += w * ownShift[u]! * uz;
          }
        }
      }
    }

    // The vertex itself always contributes weight 1, so sumW >= 1.
    let sx = sumVx / sumW;
    let sz = sumVz / sumW;

    // Re-clamp: never closer than 2 mm to the own bound body point along the own ray.
    const along = sx * vx + sz * vz;
    if (along < minShift[v]!) {
      const target = Math.max(along, ownShift[v]!);
      sx += (target - along) * vx;
      sz += (target - along) * vz;
    }
    resultX[v] = sx;
    resultZ[v] = sz;
  }

  for (let v = 0; v < count; v++) {
    const o = v * 3;
    const px = restPositions[o]!;
    const py = restPositions[o + 1]!;
    const pz = restPositions[o + 2]!;
    out[o] = px + resultX[v]!;
    out[o + 1] = py;
    out[o + 2] = pz + resultZ[v]!;
  }
}
