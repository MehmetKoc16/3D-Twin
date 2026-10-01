import { describe, expect, it } from 'vitest';
import { BODY_WRIST_M, handVertexMask, SCAN_WRIST_M } from './twinHands';

describe('hand selection', () => {
  const names = [
    'Root',
    'lowerarm_l',
    'hand_l',
    'index_01_l',
    'lowerarm_r',
    'hand_r',
    'pinky_03_r',
  ];
  const heads = Float32Array.from([
    0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0,
  ]);
  const positions = Float32Array.from([
    1.1, 0, 0, 1.2, 0, 0, 0.995, 0, 0, 0.985, 0, 0, 0.9, 0, 0, -1.1, 0, 0, -0.995, 0, 0, 0, 1, 0,
  ]);
  const joints = Uint16Array.from([
    2, 1, 0, 0, 3, 0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 0, 2, 0, 0,
  ]);
  const weights = Float32Array.from({ length: 32 }, (_, i) =>
    i % 4 === 0 ? 0.8 : i % 4 === 1 ? 0.2 : 0,
  );
  it('selects dominant hands/fingers and extends only along the corresponding forearm', () => {
    expect([...handVertexMask(positions, joints, weights, names, heads, SCAN_WRIST_M)]).toEqual([
      1, 1, 1, 0, 0, 1, 1, 0,
    ]);
  });
  it('gives the MakeHuman wrist a larger overlap than the scan cut', () => {
    expect([...handVertexMask(positions, joints, weights, names, heads, BODY_WRIST_M)]).toEqual([
      1, 1, 1, 1, 0, 1, 1, 0,
    ]);
  });
  it('uses the largest weight even when it is not the first influence', () => {
    const reordered = weights.slice();
    reordered[28] = 0.1;
    reordered[29] = 0.9;
    expect(handVertexMask(positions, joints, reordered, names, heads, SCAN_WRIST_M)[7]).toBe(1);
  });
});
