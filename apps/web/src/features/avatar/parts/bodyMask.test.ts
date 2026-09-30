import { describe, expect, it } from 'vitest';
import { filterBodyIndex, hiddenVertexMask, parseDeleteVerts } from './bodyMask';

describe('hiddenVertexMask', () => {
  it('marks listed vertices and their UV-seam copies', () => {
    // vertices 1 and 4 are seam copies of one welded group
    const weld = { groupOf: [0, 1, 2, 3, 1], start: [0, 1, 3, 4, 5], members: [0, 1, 4, 2, 3] };
    expect([...hiddenVertexMask([[1]], 5, weld)]).toEqual([0, 1, 0, 0, 1]);
  });

  it('ignores out-of-range vertices and works without a weld map', () => {
    expect([...hiddenVertexMask([[0, 9]], 3)]).toEqual([1, 0, 0]);
  });
});

describe('filterBodyIndex', () => {
  it('drops only triangles whose three vertices are hidden', () => {
    const index = Uint32Array.from([0, 1, 2, 2, 3, 4, 3, 4, 5]);
    const out = filterBodyIndex(index, Uint8Array.from([0, 0, 0, 1, 1, 1]));
    expect(Array.from(out)).toEqual([0, 1, 2, 2, 3, 4]);
  });

  it('returns the same array when nothing is hidden', () => {
    const index = Uint32Array.from([0, 1, 2]);
    expect(filterBodyIndex(index, new Uint8Array(3))).toBe(index);
  });
});

describe('parseDeleteVerts', () => {
  it('reads little-endian uint32 values and rejects ragged buffers', () => {
    const buffer = new ArrayBuffer(8);
    new DataView(buffer).setUint32(0, 7, true);
    new DataView(buffer).setUint32(4, 300, true);
    expect([...parseDeleteVerts(buffer)]).toEqual([7, 300]);
    expect(() => parseDeleteVerts(new ArrayBuffer(6))).toThrow();
  });
});
