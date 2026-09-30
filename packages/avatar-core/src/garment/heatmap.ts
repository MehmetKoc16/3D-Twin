import type { GarmentBinding } from './binding';
import { GarmentSections } from './centroid';

/** Signed radial clearance from the bound body point; negative values indicate penetration. */
export function garmentClearance(
  garmentPositions: ArrayLike<number>,
  binding: GarmentBinding,
  bodyPositions: ArrayLike<number>,
  out: Float32Array,
  sections: GarmentSections = new GarmentSections(bodyPositions),
): void {
  if (garmentPositions.length < binding.count * 3 || out.length < binding.count)
    throw new Error('garment clearance: position buffer too short');
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
    const { x: cx, z: cz } = sections.centroidAt(by, bx, bz);
    const dx = bx - cx,
      dz = bz - cz;
    const length = Math.hypot(dx, dz);
    out[v] =
      length > 1e-9
        ? ((garmentPositions[o]! - bx) * dx + (garmentPositions[o + 2]! - bz) * dz) / length
        : Math.hypot(garmentPositions[o]! - bx, garmentPositions[o + 2]! - bz);
  }
}

/** RGB stops: red at ≤0 mm, yellow at 5 mm, green from 10–25 mm, cyan at 40 mm, blue at ≥60 mm. */
export function clearanceToColor(clearanceM: number, out: Float32Array): void {
  if (out.length < 3) throw new Error('garment color: output too short');
  const stops: readonly [number, number, number, number][] = [
    [0, 1, 0, 0],
    [0.005, 1, 1, 0],
    [0.01, 0, 1, 0],
    [0.025, 0, 1, 0],
    [0.04, 0, 1, 1],
    [0.06, 0, 0, 1],
  ];
  if (clearanceM <= 0) {
    out[0] = 1;
    out[1] = 0;
    out[2] = 0;
    return;
  }
  for (let i = 1; i < stops.length; i++) {
    const high = stops[i]!;
    if (clearanceM <= high[0]) {
      const low = stops[i - 1]!;
      const t = (clearanceM - low[0]) / (high[0] - low[0]);
      out[0] = low[1] + t * (high[1] - low[1]);
      out[1] = low[2] + t * (high[2] - low[2]);
      out[2] = low[3] + t * (high[3] - low[3]);
      return;
    }
  }
  out[0] = 0;
  out[1] = 0;
  out[2] = 1;
}
