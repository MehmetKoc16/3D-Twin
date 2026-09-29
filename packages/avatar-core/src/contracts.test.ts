import { describe, expect, it } from 'vitest';
import type { Vec3 } from './index';

describe('contracts', () => {
  it('Vec3 is a 3-tuple', () => {
    const v: Vec3 = [0, 1, 0];
    expect(v).toHaveLength(3);
  });
});
