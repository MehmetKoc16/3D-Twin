import type { GarmentBinding } from './binding';
import { centroidAt, sectionCentroids } from './centroid';

export interface GradeRing {
  planeY: number;
  deltaCircumferenceM: number;
}

/**
 * Circular-section approximation: a circumference change Δc shifts each vertex by Δc / 2π
 * along the horizontal ray from its bound body point's section centroid. Rings interpolate
 * linearly by body height and hold their end values outside the given range. The offset is
 * clamped to leave at least 2 mm of projected clearance; this does not resolve mesh collisions.
 */
export function gradeGarment(
  restPositions: ArrayLike<number>,
  binding: GarmentBinding,
  bodyPositions: ArrayLike<number>,
  rings: GradeRing[],
  out: Float32Array,
): void {
  if (restPositions.length < binding.count * 3 || out.length < binding.count * 3)
    throw new Error('garment grade: position buffer too short');
  const sorted = [...rings].sort((a, b) => a.planeY - b.planeY);
  if (sorted.some((r) => !Number.isFinite(r.planeY) || !Number.isFinite(r.deltaCircumferenceM)))
    throw new Error('garment grade: invalid ring');
  const bins = sectionCentroids(bodyPositions);
  for (let v = 0; v < binding.count; v++) {
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
    const [cx, cz] = centroidAt(bins, by);
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
    let shift = delta / (2 * Math.PI);
    const clearance = (restPositions[o]! - bx) * dx + (restPositions[o + 2]! - bz) * dz;
    shift = Math.max(shift, 0.002 - clearance);
    out[o] = restPositions[o]! + shift * dx;
    out[o + 1] = restPositions[o + 1]!;
    out[o + 2] = restPositions[o + 2]! + shift * dz;
  }
}
