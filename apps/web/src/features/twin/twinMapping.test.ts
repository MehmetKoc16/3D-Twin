import { describe, expect, it } from 'vitest';
import { filterBodyIndex, hiddenVertexMask } from '../wardrobe/bodyHide';
import { TwinFormatError } from './twinDef';
import {
  compactTwinIndex,
  filterMappingNearGarments,
  hiddenTwinVertices,
  MAPPING_NEAR_GARMENT_M,
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

describe('filterMappingNearGarments', () => {
  // A garment patch in the XY plane: 10 cm x 10 cm from [-0.05, -0.05] to [0.05, 0.05] at z = 0.
  // prettier-ignore
  const patchPositions = Float32Array.from([
    -0.05, -0.05, 0,
     0.05, -0.05, 0,
     0.05,  0.05, 0,
    -0.05,  0.05, 0,
  ]);
  const patchIndex = Uint32Array.from([0, 1, 2, 0, 2, 3]);
  const surface = { positions: patchPositions, index: patchIndex };

  it('keeps a vertex 2 cm under a garment patch hidden, but leaves a vertex 10 cm beyond it visible', () => {
    // WHY: Twin forearm skin just beyond the cuff was incorrectly hidden because it mapped to MH vertices
    // under the sleeve. Restricting mapping to a wide footprint (slack 0) keeps vertices under the garment
    // hidden while exposing forearm skin beyond the cuff opening.
    // Vertex 0: under the patch center at z = 0.02 (2 cm under the garment surface).
    // Vertex 1: at x = 0.15 (10 cm beyond the patch edge x = 0.05, simulating forearm skin beyond a cuff).
    // Vertex 2: at z = 0.10 (10 cm away perpendicular to the patch, beyond the 6 cm band).
    // prettier-ignore
    const twinPositions = Float32Array.from([
      0, 0, 0.02,
      0.15, 0, 0,
      0, 0, 0.10,
    ]);
    const mappingHidden = Uint8Array.from([1, 1, 1]);
    const filtered = filterMappingNearGarments(mappingHidden, twinPositions, 3, [surface]);
    expect(filtered[0]).toBe(1); // 2 cm under the garment: stays hidden
    expect(filtered[1]).toBe(0); // 10 cm beyond the garment patch: becomes visible
    expect(filtered[2]).toBe(0); // 10 cm perpendicular distance: becomes visible
  });

  it('keeps today mapping behaviour when no garment surface is present', () => {
    // WHY: Before garment surfaces are loaded or when all clothes are taken off, mapping must not be cleared.
    const twinPositions = Float32Array.from([0, 0, 0.02, 0.15, 0, 0]);
    const mappingHidden = Uint8Array.from([1, 1]);
    const filtered = filterMappingNearGarments(mappingHidden, twinPositions, 2, []);
    expect([...filtered]).toEqual([1, 1]);
  });

  it('returns all zeros without computing footprints if no vertex is mapping-hidden', () => {
    // WHY: Avoids expensive geometric footprint checks when no vertices were hidden by mapping.
    const twinPositions = Float32Array.from([0, 0, 0.02, 0.15, 0, 0]);
    const mappingHidden = Uint8Array.from([0, 0]);
    const filtered = filterMappingNearGarments(mappingHidden, twinPositions, 2, [surface]);
    expect([...filtered]).toEqual([0, 0]);
  });

  it('supports writing into a separate output buffer', () => {
    // WHY: Allows callers to preserve input mapping buffer if needed.
    const twinPositions = Float32Array.from([0, 0, 0.02, 0.15, 0, 0]);
    const mappingHidden = Uint8Array.from([1, 1]);
    const out = new Uint8Array(2);
    const result = filterMappingNearGarments(
      mappingHidden,
      twinPositions,
      2,
      [surface],
      MAPPING_NEAR_GARMENT_M,
      out,
    );
    expect(result).toBe(out);
    expect([...out]).toEqual([1, 0]);
    expect([...mappingHidden]).toEqual([1, 1]);
  });
});
