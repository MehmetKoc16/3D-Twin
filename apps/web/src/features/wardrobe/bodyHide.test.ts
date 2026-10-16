import { describe, expect, it } from 'vitest';
import { buildWeldMap } from '../avatar/meshMath';
import {
  clipCoverageToHem,
  clipDeleteVerts,
  combinedHiddenMask,
  coveredBodyVertices,
  filterBodyIndex,
  hiddenVertexMask,
  parseDeleteVerts,
  protectedBodyVertices,
} from './bodyHide';

// Two quads (4 triangles) over 6 vertices; vertex 6 is a UV-seam copy of vertex 1.
const positions = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0, 2, 1, 0, 1, 0, 0];
const indices = new Uint32Array([0, 1, 2, 0, 2, 3, 6, 4, 5, 6, 5, 2]);

describe('hiddenVertexMask', () => {
  it('unions the delete lists of all worn items', () => {
    const mask = hiddenVertexMask([[0, 2], [3]], 7);
    expect(Array.from(mask)).toEqual([1, 0, 1, 1, 0, 0, 0]);
  });

  it('marks UV-seam copies through the weld map and ignores out-of-range ids', () => {
    const weld = buildWeldMap(positions);
    const mask = hiddenVertexMask([[1, 99]], 7, weld);
    expect(mask[1]).toBe(1);
    expect(mask[6]).toBe(1); // copy of vertex 1
    expect(Array.from(mask).reduce((a, b) => a + b, 0)).toBe(2);
  });
});

describe('filterBodyIndex', () => {
  it('drops only triangles whose three vertices are all hidden', () => {
    const mask = hiddenVertexMask([[0, 1, 2]], 7);
    const out = filterBodyIndex(indices, mask);
    expect(Array.from(out)).toEqual([0, 2, 3, 6, 4, 5, 6, 5, 2]);
  });

  it('keeps triangles that only touch a hidden vertex', () => {
    const mask = hiddenVertexMask([[0]], 7);
    expect(filterBodyIndex(indices, mask)).toBe(indices); // nothing removed: same array back
  });

  it('restores the full index when nothing is hidden (take-off)', () => {
    const hidden = filterBodyIndex(indices, hiddenVertexMask([[0, 1, 2, 3]], 7));
    expect(hidden.length).toBeLessThan(indices.length);
    const restored = filterBodyIndex(indices, hiddenVertexMask([], 7));
    expect(Array.from(restored)).toEqual(Array.from(indices));
  });

  it('hides a whole quad worth of seam-split vertices', () => {
    const weld = buildWeldMap(positions);
    const out = filterBodyIndex(indices, hiddenVertexMask([[1, 4, 5, 2]], 7, weld));
    expect(Array.from(out)).toEqual([0, 1, 2, 0, 2, 3]); // second quad gone, first quad untouched
  });
});

describe('parseDeleteVerts', () => {
  it('reads little-endian uint32 lists', () => {
    const buffer = new ArrayBuffer(12);
    const view = new DataView(buffer);
    view.setUint32(0, 5, true);
    view.setUint32(4, 70000, true);
    view.setUint32(8, 1, true);
    expect(Array.from(parseDeleteVerts(buffer))).toEqual([5, 70000, 1]);
    expect(() => parseDeleteVerts(new ArrayBuffer(6))).toThrow();
  });
});

/** A flat "body" sheet (z = 0, normals +z) made of unit quads split in two triangles, x in [0, n). */
function sheet(
  n: number,
  z: (x: number) => number,
): { positions: number[]; index: number[]; normals: number[] } {
  const positions: number[] = [];
  const index: number[] = [];
  const normals: number[] = [];
  for (let x = 0; x <= n; x++) {
    positions.push(x, 0, z(x), x, 1, z(x));
    normals.push(0, 0, 1, 0, 0, 1);
  }
  for (let x = 0; x < n; x++) {
    const a = x * 2;
    index.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
  }
  return { positions, index, normals };
}

describe('coveredBodyVertices', () => {
  const body = sheet(10, () => 0);
  const count = body.positions.length / 3;
  const xOf = (vertex: number): number => body.positions[vertex * 3]!;

  it('covers the body vertices under the garment footprint and none beside it', () => {
    // garment sheet 5 mm above the body, only over x in [0, 5]
    const garment = sheet(5, () => 0.005);
    const covered = coveredBodyVertices(body.positions, count, garment.positions, garment.index, {
      cellM: 2,
      bandM: 0.015,
    });
    for (let v = 0; v < count; v++) {
      if (xOf(v) <= 4) expect(covered[v], `x=${xOf(v)}`).toBe(1);
      if (xOf(v) >= 6) expect(covered[v], `x=${xOf(v)}`).toBe(0);
    }
    // quads 0..3 lie completely under the garment: 8 triangles go, the quads at the edge (x >= 5) stay
    const remaining = filterBodyIndex(Uint32Array.from(body.index), covered);
    expect(remaining.length / 3).toBeLessThanOrEqual(20 - 8);
    expect(remaining.length / 3).toBeGreaterThanOrEqual(20 - 10);
  });

  it('covers a body bump that pokes through the fabric, but not a surface far behind or in front', () => {
    const bump = sheet(10, (x) => (x >= 4 && x <= 6 ? 0.006 : 0)); // body pokes 1 mm above a garment at z = 0.005
    const garment = sheet(10, () => 0.005);
    const under = coveredBodyVertices(bump.positions, count, garment.positions, garment.index, {
      cellM: 2,
      bandM: 0.015,
    });
    expect(under.every((v) => v === 1)).toBe(true);
    const far = sheet(10, () => 0.2);
    expect(
      coveredBodyVertices(body.positions, count, far.positions, far.index, {
        cellM: 2,
        bandM: 0.015,
      }).every((v) => v === 0),
    ).toBe(true);
    const behind = sheet(10, () => -0.2);
    expect(
      coveredBodyVertices(body.positions, count, behind.positions, behind.index, {
        cellM: 2,
        bandM: 0.015,
      }).every((v) => v === 0),
    ).toBe(true);
  });
});

describe('protectedBodyVertices', () => {
  // Bones: 0 spine, 1 neck_01, 2 hand_l, 3 index_02_r, 4 upperarm_l. Vertex 6 is a UV-seam copy of vertex 1.
  const boneNames = ['spine_03', 'neck_01', 'hand_l', 'index_02_r', 'upperarm_l'];
  const skinIndex = [
    0,
    0,
    0,
    0, // 0: spine only
    2,
    0,
    0,
    0, // 1: hand
    4,
    2,
    0,
    0, // 2: upper arm, hand weight below the threshold
    3,
    0,
    0,
    0, // 3: finger
    1,
    0,
    0,
    0, // 4: neck
    0,
    0,
    0,
    0, // 5: spine, but high (above the neck line)
    0,
    0,
    0,
    0, // 6: seam copy of 1, skinned to the spine only
  ];
  const skinWeight = [
    1, 0, 0, 0, 1, 0, 0, 0, 0.9995, 0.0005, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
  ];
  const pos = (y: number[]): number[] => y.flatMap((v, i) => [i, v, 0]);

  it('protects hand, finger and neck skin by bone weight and the head by height', () => {
    const mask = protectedBodyVertices({
      boneNames,
      skinIndex,
      skinWeight,
      positions: pos([1.2, 1.2, 1.2, 1.2, 1.2, 1.5, 1.2]),
      neckY: 1.45,
    });
    expect(Array.from(mask)).toEqual([0, 1, 0, 1, 1, 1, 0]);
  });

  it('covers the whole band 2.5 cm below the neck joint and skips the height rule without a neck joint', () => {
    const positions = pos([1.2, 1.2, 1.2, 1.2, 1.2, 1.426, 1.2]);
    const withNeck = protectedBodyVertices({
      boneNames,
      skinIndex,
      skinWeight,
      positions,
      neckY: 1.45,
    });
    expect(withNeck[5]).toBe(1);
    const without = protectedBodyVertices({ boneNames, skinIndex, skinWeight, positions });
    expect(without[5]).toBe(0);
  });

  it('extends protection to the UV-seam copies of a protected vertex', () => {
    const positions = pos([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]);
    // make vertex 6 coincide with vertex 1 so that the weld map groups them
    positions[6 * 3] = positions[1 * 3]!;
    positions[6 * 3 + 1] = positions[1 * 3 + 1]!;
    positions[6 * 3 + 2] = positions[1 * 3 + 2]!;
    const weld = buildWeldMap(positions);
    const mask = protectedBodyVertices({ boneNames, skinIndex, skinWeight, positions, weld });
    expect(mask[6]).toBe(1);
    expect(protectedBodyVertices({ boneNames, skinIndex, skinWeight, positions })[6]).toBe(0);
  });
});

describe('combinedHiddenMask', () => {
  it('keeps the protection guard after combining four garments', () => {
    const mask = combinedHiddenMask(
      [[0, 1], [1, 2], [3], [4, 5, 6]],
      [[0, 1, 0, 0, 0, 0, 0], null, [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]],
      7,
      undefined,
      [0, 1, 0, 0, 1, 0, 1],
    );
    expect(Array.from(mask)).toEqual([1, 0, 1, 1, 0, 1, 0]);
  });
  it('unions delete lists and footprint coverage, then clears the protected vertices', () => {
    const protectedMask = Uint8Array.from([0, 1, 0, 1, 0, 0, 0]);
    const coverage = Uint8Array.from([0, 0, 0, 1, 1, 0, 0]); // vertex 3 is protected, 4 is not
    const mask = combinedHiddenMask([[0, 1]], [coverage, null], 7, undefined, protectedMask);
    expect(Array.from(mask)).toEqual([1, 0, 0, 0, 1, 0, 0]);
  });

  it('protection beats both sources, including the seam copies of a deleted vertex', () => {
    const weld = buildWeldMap(positions);
    const protectedMask = new Uint8Array(7);
    protectedMask[1] = 1;
    protectedMask[6] = 1; // the guard marks every seam copy
    const mask = combinedHiddenMask(
      [[1, 2]],
      [Uint8Array.from([0, 1, 0, 0, 0, 0, 1])],
      7,
      weld,
      protectedMask,
    );
    expect(mask[1]).toBe(0);
    expect(mask[6]).toBe(0);
    expect(mask[2]).toBe(1);
    // triangles with a protected vertex always stay; (0, 2, 3) has none and goes
    const out = filterBodyIndex(
      indices,
      combinedHiddenMask([[0, 1, 2, 3]], [], 7, weld, protectedMask),
    );
    expect(Array.from(out)).toEqual([0, 1, 2, 6, 4, 5, 6, 5, 2]);
  });

  it('behaves like the plain union without a protection mask', () => {
    const mask = combinedHiddenMask([[0]], [Uint8Array.from([0, 0, 0, 0, 0, 1, 0])], 7);
    expect(Array.from(mask)).toEqual([1, 0, 0, 0, 0, 1, 0]);
  });
});

describe('clipDeleteVerts', () => {
  // Synthetic leg column:
  // 0: ankle at y = 0.10 m (original cuff)
  // 1: ankle at y = 0.14 m (4 cm below an 8 cm shortened cuff)
  // 2: ankle at y = 0.165 m (1.5 cm below the 0.18 m cuff -> more than 1 cm below)
  // 3: new cuff at y = 0.18 m (+8 cm from 0.10 m)
  // 4: lower calf at y = 0.22 m (above the new cuff, under the garment)
  // 5: mid calf at y = 0.45 m (well under the garment)
  // 6: protected hand/neck vertex at y = 1.48 m
  // 7: UV-seam copy of vertex 1 (y = 0.14 m, below the shortened cuff)
  // 8: UV-seam copy of vertex 4 (y = 0.22 m, above the shortened cuff)
  const legBody = Float32Array.from([
    0.08, 0.1, 0, 0.08, 0.14, 0, 0.08, 0.165, 0, 0.08, 0.18, 0, 0.08, 0.22, 0, 0.08, 0.45, 0, 0.08,
    1.48, 0, 0.08, 0.14, 0, 0.08, 0.22, 0,
  ]);
  const legCount = legBody.length / 3;
  const staticDelete = Uint32Array.from([0, 1, 2, 3, 4, 5, 6]);
  // Un-graded trouser quad spanning y = 0.10 m (ankle hem) to y = 0.80 m (crotch)
  const restTrouser = Float32Array.from([
    0.04, 0.1, 0.005, 0.12, 0.1, 0.005, 0.12, 0.8, 0.005, 0.04, 0.8, 0.005,
  ]);
  // Shortened trouser quad with hem pulled up by 8 cm to y = 0.18 m
  const shortenedTrouser = Float32Array.from([
    0.04, 0.18, 0.005, 0.12, 0.18, 0.005, 0.12, 0.8, 0.005, 0.04, 0.8, 0.005,
  ]);
  // Lengthened trouser quad with hem pushed down to y = 0.06 m
  const lengthenedTrouser = Float32Array.from([
    0.04, 0.06, 0.005, 0.12, 0.06, 0.005, 0.12, 0.8, 0.005, 0.04, 0.8, 0.005,
  ]);
  const trouserIndex = Uint32Array.from([0, 1, 2, 0, 2, 3]);

  it('keeps the exact same hidden vertex set when garment length does not change or increases', () => {
    // Why: When a garment is not shortened, the static delete list and footprint coverage must stay
    // bit-for-bit identical to the existing behaviour so normal fits never expose unexpected skin.
    const weld = buildWeldMap(legBody);
    const unchangedList = clipDeleteVerts(staticDelete, legBody, restTrouser, {
      rest: restTrouser,
      weld,
    });
    expect(unchangedList).toBe(staticDelete);
    const rawCoverage = coveredBodyVertices(legBody, legCount, restTrouser, trouserIndex);
    const clippedCoverage = clipCoverageToHem(Uint8Array.from(rawCoverage), legBody, restTrouser, {
      rest: restTrouser,
      weld,
    });
    expect(Array.from(clippedCoverage)).toEqual(Array.from(rawCoverage));
    const baselineMask = combinedHiddenMask([staticDelete], [rawCoverage], legCount, weld);
    const unchangedMask = combinedHiddenMask([unchangedList], [clippedCoverage], legCount, weld);
    expect(Array.from(unchangedMask)).toEqual(Array.from(baselineMask));

    const lengthenedList = clipDeleteVerts(staticDelete, legBody, lengthenedTrouser, {
      rest: restTrouser,
      weld,
    });
    expect(lengthenedList).toBe(staticDelete);
  });

  it('unhides ankle vertices more than 1 cm below an 8 cm shortened hem while keeping upper leg vertices hidden', () => {
    // Why: Pulling the trouser hem up by 8 cm exposes the ankle in open air; vertices > 1 cm below the
    // new hem (y < 0.17 m) must not be deleted, while vertices at/above the hem (y >= 0.18 m) stay hidden.
    const clippedDelete = clipDeleteVerts(staticDelete, legBody, shortenedTrouser, {
      rest: restTrouser,
    });
    const coverage = clipCoverageToHem(
      coveredBodyVertices(legBody, legCount, shortenedTrouser, trouserIndex),
      legBody,
      shortenedTrouser,
      { rest: restTrouser },
    );
    const mask = combinedHiddenMask([clippedDelete], [coverage], legCount);
    // More than 1 cm below the new 0.18 m hem (y = 0.10, 0.14, 0.165 m): visible (0)
    expect(mask[0]).toBe(0);
    expect(mask[1]).toBe(0);
    expect(mask[2]).toBe(0);
    // At and above the new 0.18 m hem under the garment (y = 0.18, 0.22, 0.45 m): hidden (1)
    expect(mask[3]).toBe(1);
    expect(mask[4]).toBe(1);
    expect(mask[5]).toBe(1);
  });

  it('never hides protected hand, finger, neck or head vertices even when above the hem', () => {
    // Why: Anatomical protection for hands, fingers, neck and head must override both the clipped
    // delete list and footprint coverage in every length-grading state.
    const weld = buildWeldMap(legBody);
    const protectedMask = new Uint8Array(legCount);
    protectedMask[6] = 1;
    const clippedDelete = clipDeleteVerts(staticDelete, legBody, shortenedTrouser, {
      rest: restTrouser,
      weld,
    });
    const coverage = clipCoverageToHem(
      coveredBodyVertices(legBody, legCount, shortenedTrouser, trouserIndex),
      legBody,
      shortenedTrouser,
      { rest: restTrouser, weld },
    );
    const mask = combinedHiddenMask([clippedDelete], [coverage], legCount, weld, protectedMask);
    expect(mask[6]).toBe(0);
    expect(mask[4]).toBe(1);
  });

  it('keeps UV-seam copies synchronized when ankle vertices are revealed and calf vertices stay hidden', () => {
    // Why: UV-split duplicate vertices occupy the same 3D point on a seam; revealing or hiding only one
    // copy would leave stray triangles or tear the seam at the ankle cuff.
    const weld = buildWeldMap(legBody);
    const clippedDelete = clipDeleteVerts(staticDelete, legBody, shortenedTrouser, {
      rest: restTrouser,
      weld,
    });
    const coverage = clipCoverageToHem(
      coveredBodyVertices(legBody, legCount, shortenedTrouser, trouserIndex),
      legBody,
      shortenedTrouser,
      { rest: restTrouser, weld },
    );
    const mask = combinedHiddenMask([clippedDelete], [coverage], legCount, weld);
    // Vertex 7 is the UV-seam copy of ankle vertex 1 (y = 0.14 m, below the 0.18 m cuff)
    expect(mask[1]).toBe(0);
    expect(mask[7]).toBe(0);
    // Vertex 8 is the UV-seam copy of calf vertex 4 (y = 0.22 m, above the 0.18 m cuff)
    expect(mask[4]).toBe(1);
    expect(mask[8]).toBe(1);
  });
});
