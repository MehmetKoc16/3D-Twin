import { describe, expect, it } from 'vitest';
import { filterBodyIndex, hiddenVertexMask } from '../wardrobe/bodyHide';
import { TwinFormatError } from './twinDef';
import {
  compactTwinIndex,
  hiddenTwinVertices,
  parseMapping,
  unreferencedVertices,
  validateMapping,
} from './twinMapping';

/** Body: a strip of 4 quads (10 vertices, 8 triangles) along x; twin: 9 vertices mapped onto it. */
// prettier-ignore
const bodyIndex = Uint32Array.from([
  0, 1, 5, 0, 5, 4, // quad 0
  1, 2, 6, 1, 6, 5, // quad 1
  2, 3, 7, 2, 7, 6, // quad 2
  3, 8, 9, 3, 9, 7, // quad 3 (vertices 8, 9 hang off the right end)
]);
const bodyVertexCount = 10;

describe('mh2twin.bin parsing', () => {
  it('reads little-endian uint32', () => {
    const bytes = new Uint8Array([1, 0, 0, 0, 0, 1, 0, 0, 255, 255, 0, 0]);
    expect([...parseMapping(bytes.buffer)]).toEqual([1, 256, 65535]);
  });

  it('rejects empty and misaligned files', () => {
    expect(() => parseMapping(new ArrayBuffer(0))).toThrow(TwinFormatError);
    expect(() => parseMapping(new ArrayBuffer(6))).toThrow(/multiple of 4/);
  });

  it('validates the entry count and the range', () => {
    const mapping = Uint32Array.from([0, 3, 9]);
    expect(() => validateMapping(mapping, 3, 10)).not.toThrow();
    expect(() => validateMapping(mapping, 4, 10)).toThrow(/3 entries/);
    expect(() => validateMapping(Uint32Array.from([0, 10]), 2, 10)).toThrow(/outside the body/);
  });
});

describe('hiding twin triangles under garments', () => {
  const mapping = Uint32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 9]); // twin vertex i -> body vertex
  const twinIndex = Uint32Array.from([0, 1, 5, 1, 2, 6, 2, 3, 7, 4, 5, 6, 3, 7, 8]);

  it('nothing is hidden while every body vertex is still used', () => {
    const hidden = unreferencedVertices(bodyIndex, bodyVertexCount);
    expect([...hidden].every((v) => v === 0)).toBe(true);
    const target = new Uint32Array(twinIndex.length);
    expect(compactTwinIndex(twinIndex, hiddenTwinVertices(mapping, hidden), target)).toBe(
      twinIndex.length,
    );
    expect([...target]).toEqual([...twinIndex]);
  });

  it('a delete list that leaves border triangles hides no twin triangle at its rim', () => {
    // a garment deletes body vertices 1, 2, 5, 6: only the triangles among them are dropped
    const mask = hiddenVertexMask([Uint32Array.from([1, 2, 5, 6])], bodyVertexCount);
    const kept = filterBodyIndex(bodyIndex, mask);
    expect(kept.length).toBe(bodyIndex.length - 6); // (1,2,6) and (1,6,5) are gone
    // 1, 2, 5, 6 still touch a kept border triangle, so the strict rule hides nothing yet
    expect(unreferencedVertices(kept, bodyVertexCount).every((v) => v === 0)).toBe(true);
  });

  it('a body vertex whose every triangle is gone hides the twin triangles that map onto it only', () => {
    // hide vertices 0..7: every triangle of quads 0-2 is dropped, quad 3 keeps (3,8,9) and (3,9,7)
    const mask = hiddenVertexMask([Uint32Array.from([0, 1, 2, 3, 4, 5, 6, 7])], bodyVertexCount);
    const kept = filterBodyIndex(bodyIndex, mask);
    expect([...kept]).toEqual([3, 8, 9, 3, 9, 7]);
    const unreferenced = unreferencedVertices(kept, bodyVertexCount);
    expect([...unreferenced]).toEqual([1, 1, 1, 0, 1, 1, 1, 0, 0, 0]); // 3 and 7 are still used
    const hiddenTwin = hiddenTwinVertices(mapping, unreferenced);
    expect([...hiddenTwin]).toEqual([1, 1, 1, 0, 1, 1, 1, 0, 0]);
    const target = new Uint32Array(twinIndex.length);
    const count = compactTwinIndex(twinIndex, hiddenTwin, target);
    // (0,1,5), (1,2,6), (4,5,6) are fully hidden; (2,3,7) and (3,7,8) touch a visible vertex
    expect([...target.slice(0, count)]).toEqual([2, 3, 7, 3, 7, 8]);
  });

  it('reuses the output buffers', () => {
    const scratch = new Uint8Array(bodyVertexCount).fill(7);
    expect(unreferencedVertices(bodyIndex, bodyVertexCount, scratch)).toBe(scratch);
    expect(scratch.every((v) => v === 0)).toBe(true);
    const out = new Uint8Array(mapping.length);
    expect(hiddenTwinVertices(mapping, scratch, out)).toBe(out);
  });
});
