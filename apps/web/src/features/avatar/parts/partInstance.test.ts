import { describe, expect, it } from 'vitest';
import { Color } from 'three';
import { tintColor } from './partInstance';

describe('tintColor', () => {
  it('divides a picked colour by the texture mean so the average reads true', () => {
    const picked = new Color('#808080');
    expect(tintColor('#808080', 0.5).r).toBeCloseTo(picked.r / 0.5, 5);
  });

  it('caps very bright results', () => {
    expect(tintColor('#ffffff', 0.05).r).toBeLessThanOrEqual(2.5);
  });
});
