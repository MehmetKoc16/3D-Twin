import { describe, expect, it } from 'vitest';
import type { Vec3 } from '@dt/avatar-core';
import { computeBoneFocusTargets } from './boneFocusTargets';

const bones = new Map<string, Vec3>([
  ['head', [0, 1.5, 0]], ['spine_03', [0, 1.3, 0]], ['pelvis', [0, 0.95, 0]],
  ['thigh_l', [0.09, 0.9, 0]], ['thigh_r', [-0.09, 0.9, 0]], ['calf_l', [0.09, 0.5, 0]], ['calf_r', [-0.09, 0.5, 0]],
  ['foot_l', [0.09, 0.08, 0]], ['foot_r', [-0.09, 0.08, 0]], ['ball_l', [0.09, 0.02, 0.13]], ['ball_r', [-0.09, 0.02, 0.13]],
]);

describe('computeBoneFocusTargets', () => {
  it('centres the feet preset on the shoe area, close enough to frame it', () => {
    const t = computeBoneFocusTargets(bones, 1.75);
    expect(t.feet.target[0]).toBeCloseTo(0);
    expect(t.feet.target[1]).toBeCloseTo(0.05);
    expect(t.feet.target[2]).toBeCloseTo(0.065);
    expect(t.feet.distance).toBeLessThan(1.2);
    expect(t.feet.distance).toBeGreaterThanOrEqual(0.25);
  });
  it('follows the bones for face, upper and lower', () => {
    const t = computeBoneFocusTargets(bones, 1.75);
    expect(t.face.target[1]).toBeGreaterThan(1.5);
    expect(t.upper.target).toEqual([0, 1.3, 0]);
    expect(t.lower.target[1]).toBeGreaterThan(0.5);
    expect(t.lower.target[1]).toBeLessThan(0.95);
    expect(t.full.target[1]).toBeCloseTo(0.875);
  });
  it('falls back to height fractions when bones are missing', () => {
    const t = computeBoneFocusTargets(new Map(), 2);
    expect(t.feet.target[1]).toBeCloseTo(0.08);
  });
});
