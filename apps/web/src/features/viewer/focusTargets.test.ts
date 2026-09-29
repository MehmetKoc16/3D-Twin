import { describe, expect, it } from 'vitest';
import { getFocusTargets } from './focusTargets';

describe('getFocusTargets', () => {
  it('places body targets at the intended height fractions', () => {
    const targets = getFocusTargets(2);
    expect(targets.face.target[1]).toBeCloseTo(1.86);
    expect(targets.upper.target[1]).toBeCloseTo(1.44);
    expect(targets.lower.target[1]).toBeCloseTo(0.6);
    expect(targets.feet.target[1]).toBeCloseTo(0.08);
    expect(targets.full.distance).toBeGreaterThan(targets.upper.distance);
  });
  it('clamps implausible height', () => {
    expect(getFocusTargets(0).face.target[1]).toBeCloseTo(1.4 * 0.93);
  });
});
