import type { Vec3 } from '../contracts';
import type { GarmentBinding } from './binding';

/**
 * One longitudinal grading pass. `anchor` and `hem` are points on the solved body (or its joints),
 * in metres. A vertex is displaced along anchor -> hem by a fraction of the chart difference:
 * zero at the anchor plane, full at the hem plane, linearly interpolated between. `side` and
 * `minAbsX` restrict a sleeve pass to one arm. Run separate passes for left and right sleeves.
 * The chart difference is limited to ±15 cm; when `floorY` is set, downward displacement is
 * further limited so no affected vertex crosses the floor. Choose `anchor` well above the hem: the whole change is
 * squeezed into the anchor-to-hem span, so a short span over-stretches (and jags) the rows near the hem.
 * This is a longitudinal stretch model,
 * not a cloth simulation or a seam-aware pattern resize.
 */
export interface LengthGradeSpec {
  chartCm: number;
  referenceCm: number;
  anchor: Vec3;
  hem: Vec3;
  side?: -1 | 1;
  minAbsX?: number;
  floorY?: number;
}

export function gradeGarmentLength(
  positions: ArrayLike<number>,
  binding: GarmentBinding,
  bodyPositions: ArrayLike<number>,
  spec: LengthGradeSpec,
  out: Float32Array,
): void {
  const count = binding.count;
  if (positions.length < count * 3 || out.length < count * 3)
    throw new Error('garment length: position buffer too short');
  const { anchor, hem } = spec;
  const ax = hem[0] - anchor[0],
    ay = hem[1] - anchor[1],
    az = hem[2] - anchor[2];
  const length = Math.hypot(ax, ay, az);
  if (
    !Number.isFinite(length) ||
    length < 1e-6 ||
    !Number.isFinite(spec.chartCm) ||
    !Number.isFinite(spec.referenceCm)
  )
    throw new Error('garment length: invalid spec');
  const ux = ax / length,
    uy = ay / length,
    uz = az / length;
  let delta = Math.max(-0.15, Math.min(0.15, (spec.chartCm - spec.referenceCm) / 100));
  const fractions = new Float32Array(count);
  for (let v = 0; v < count; v++) {
    const o = v * 3;
    let bx = 0;
    for (let j = 0; j < 3; j++)
      bx += binding.weights[o + j]! * bodyPositions[binding.indices[o + j]! * 3]!;
    if (
      spec.side !== undefined &&
      (Math.sign(bx) !== spec.side || Math.abs(bx) < (spec.minAbsX ?? 0))
    )
      continue;
    const projection =
      ((positions[o]! - anchor[0]) * ux +
        (positions[o + 1]! - anchor[1]) * uy +
        (positions[o + 2]! - anchor[2]) * uz) /
      length;
    const fraction = Math.max(0, Math.min(1, projection));
    fractions[v] = fraction;
    if (spec.floorY !== undefined && delta > 0 && uy < 0 && fraction > 0) {
      delta = Math.min(delta, Math.max(0, (positions[o + 1]! - spec.floorY) / (-uy * fraction)));
    }
  }
  for (let v = 0; v < count; v++) {
    const o = v * 3;
    const shift = delta * fractions[v]!;
    out[o] = positions[o]! + ux * shift;
    out[o + 1] = positions[o + 1]! + uy * shift;
    out[o + 2] = positions[o + 2]! + uz * shift;
  }
}
