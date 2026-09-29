import { describe, expect, it } from 'vitest';
import { recolorFactor } from './garmentInstance';

describe('recolorFactor', () => {
  it('is 1 at the template base colour (the texture is shown unchanged)', () => {
    const [r, g, b] = recolorFactor('#838383', '#838383');
    expect(r).toBeCloseTo(1, 5);
    expect(g).toBeCloseTo(1, 5);
    expect(b).toBeCloseTo(1, 5);
  });

  it('scales the grey texture up to a brighter or more saturated item colour, per linear channel', () => {
    const [r, g, b] = recolorFactor('#ff0000', '#838383');
    expect(r).toBeGreaterThan(1);
    expect(g).toBeLessThan(0.05);
    expect(b).toBeLessThan(0.05);
  });

  it('caps the factor for near-black textures', () => {
    const [r] = recolorFactor('#ffffff', '#101010');
    expect(r).toBeLessThanOrEqual(12);
  });
});
