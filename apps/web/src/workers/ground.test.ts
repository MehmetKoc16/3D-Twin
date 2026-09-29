import { describe, expect, it } from 'vitest';
import { flattenJoints, groundRenderPositions } from './ground';

describe('grounding', () => {
  it('translates so the lowest render vertex is at y = 0 and ignores joint points', () => {
    // 3 render vertices + 1 joint point far below
    const full = Float32Array.from([0, 1, 0, 1, -0.2, 0, 2, 0.5, 1, 9, -5, 9]);
    const { positions, offsetY } = groundRenderPositions(full, 3);
    expect(offsetY).toBeCloseTo(-0.2);
    expect(positions).toHaveLength(9);
    expect(positions[1]).toBeCloseTo(1.2);
    expect(positions[4]).toBeCloseTo(0);
    expect(positions[7]).toBeCloseTo(0.7);
    expect(positions[6]).toBe(2);
  });
  it('shifts joints by the same offset, in the requested order', () => {
    const joints = new Map([
      ['b', { head: [1, 2, 3], tail: [4, 5, 6] }],
      ['a', { head: [0, 0, 0], tail: [0, 1, 0] }],
    ]);
    const flat = flattenJoints(['a', 'b'], joints, -0.5);
    expect(Array.from(flat)).toEqual([0, 0.5, 0, 0, 1.5, 0, 1, 2.5, 3, 4, 5.5, 6]);
    expect(() => flattenJoints(['missing'], joints, 0)).toThrow();
  });
});
