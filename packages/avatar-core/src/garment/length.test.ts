import { describe, expect, it } from 'vitest';
import type { Vec3 } from '../contracts';
import type { GarmentBinding } from './binding';
import { gradeGarmentLength, type LengthGradeSpec } from './length';

// A synthetic sleeve: a tube of rings along a slanted shoulder -> wrist chord (0.25 m long), every vertex bound to one
// body vertex on the +x side, so the `side` filter keeps all of them.
const anchor: Vec3 = [0.2, 1.4, 0];
const hem: Vec3 = [0.35, 1.2, 0];
const chord = 0.25;
const u: Vec3 = [0.6, -0.8, 0];
const e1: Vec3 = [0, 0, 1];
const e2: Vec3 = [0.8, 0.6, 0];
const body = [0.3, 1.3, 0];

/** Sleeve whose cuff ends `cuffPastHemM` beyond the hem point along the chord (negative: short of it). */
function sleeve(cuffPastHemM: number): { positions: Float32Array; binding: GarmentBinding } {
  const rows = 12;
  const around = 6;
  const radius = 0.04;
  const positions = new Float32Array((rows + 1) * around * 3);
  let o = 0;
  for (let r = 0; r <= rows; r++) {
    const p = (r / rows) * (chord + cuffPastHemM);
    for (let a = 0; a < around; a++) {
      const angle = (a / around) * Math.PI * 2;
      const c = Math.cos(angle) * radius;
      const s = Math.sin(angle) * radius;
      for (let k = 0; k < 3; k++)
        positions[o + k] = anchor[k]! + u[k]! * p + e1[k]! * c + e2[k]! * s;
      o += 3;
    }
  }
  const count = positions.length / 3;
  const indices = new Uint32Array(count * 3);
  const weights = new Float32Array(count * 3);
  for (let v = 0; v < count; v++) weights[v * 3] = 1;
  return { positions, binding: { count, indices, weights, offsets: new Float32Array(count * 3) } };
}

function projection(positions: ArrayLike<number>, v: number): number {
  const o = v * 3;
  return (
    (positions[o]! - anchor[0]) * u[0] +
    (positions[o + 1]! - anchor[1]) * u[1] +
    (positions[o + 2]! - anchor[2]) * u[2]
  );
}

function maxProjection(positions: ArrayLike<number>): number {
  let max = -Infinity;
  for (let v = 0; v < positions.length / 3; v++) max = Math.max(max, projection(positions, v));
  return max;
}

function grade(
  cuffPastHemM: number,
  deltaCm: number,
  maxBeyondHemM?: number,
): { positions: Float32Array; out: Float32Array } {
  const { positions, binding } = sleeve(cuffPastHemM);
  const spec: LengthGradeSpec = { chartCm: 58 + deltaCm, referenceCm: 58, anchor, hem, side: 1 };
  if (maxBeyondHemM !== undefined) spec.maxBeyondHemM = maxBeyondHemM;
  const out = new Float32Array(positions.length);
  gradeGarmentLength(positions, binding, body, spec, out);
  return { positions, out };
}

describe('gradeGarmentLength maxBeyondHemM', () => {
  it('does not lengthen a cuff that already ends past the limit', () => {
    // WHY: the female-body bug - the rest cuff already ended 1.4 cm past the wrist and the chart's +3 cm pushed it
    // over the hand, where hand skin poked through. The cap must stop that lengthening completely (and not pull the
    // cuff back, since shortening is only ever driven by the chart).
    const { positions, out } = grade(0.014, 3, 0.01);
    for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(positions[i]!, 6);
    expect(maxProjection(out)).toBeLessThanOrEqual(maxProjection(positions) + 1e-6);
  });

  it('lengthens only up to the limit when the cuff is just short of it', () => {
    // WHY: a cuff 0.5 cm past the hem may still grow, but its end must stop at hem + 1 cm, not at hem + 3.5 cm.
    const { out } = grade(0.005, 3, 0.01);
    expect(maxProjection(out)).toBeLessThanOrEqual(chord + 0.01 + 1e-6);
    expect(maxProjection(out)).toBeCloseTo(chord + 0.01, 5);
  });

  it('lengthens a short cuff until it reaches the limit', () => {
    // WHY: a cuff 2 cm short of the wrist with a +5 cm chart may move 3 cm (to hem + 1 cm); the cap must use the
    // vertex's real displacement (delta * fraction), not cut the chart delta to 3 cm and leave the cuff short.
    const { positions, out } = grade(-0.02, 5, 0.01);
    expect(maxProjection(out) - maxProjection(positions)).toBeCloseTo(0.03, 5);
    expect(maxProjection(out)).toBeLessThanOrEqual(chord + 0.01 + 1e-6);
  });

  it('never limits shortening', () => {
    // WHY: the cap is only about cuffs running over the hand; a smaller chart sleeve must still shorten fully.
    const { positions, out } = grade(0.014, -2, 0.01);
    expect(maxProjection(out) - maxProjection(positions)).toBeCloseTo(-0.02, 5);
  });

  it('keeps the uncapped behaviour without the field', () => {
    // WHY: top and trouser hems do not set the field and must keep the plain stretch (shift = delta * fraction).
    const { positions, out } = grade(0.014, 3);
    for (let v = 0; v < positions.length / 3; v++) {
      const shift = 0.03 * Math.max(0, Math.min(1, projection(positions, v) / chord));
      for (let k = 0; k < 3; k++)
        expect(out[v * 3 + k]).toBeCloseTo(positions[v * 3 + k]! + u[k]! * shift, 5);
    }
    expect(maxProjection(out) - maxProjection(positions)).toBeCloseTo(0.03, 5);
  });

  it('rejects a non-finite limit', () => {
    // WHY: a NaN limit would silently disable the cap (every comparison with NaN is false).
    const { positions, binding } = sleeve(0);
    const out = new Float32Array(positions.length);
    expect(() =>
      gradeGarmentLength(
        positions,
        binding,
        body,
        { chartCm: 60, referenceCm: 58, anchor, hem, maxBeyondHemM: Number.NaN },
        out,
      ),
    ).toThrow(/invalid spec/);
  });
});
