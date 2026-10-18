import { describe, expect, it } from 'vitest';
import { bridgeGarmentGaps } from './twinGapBridge';

/** A strip of `rows` vertex rows, `cols` vertices each, two triangles per quad; row r owns vertices r*cols..r*cols+cols-1. */
function strip(rows: number, cols: number): number[] {
  const out: number[] = [];
  for (let r = 0; r + 1 < rows; r++)
    for (let c = 0; c + 1 < cols; c++) {
      const a = r * cols + c;
      out.push(a, a + 1, a + cols, a + 1, a + cols + 1, a + cols);
    }
  return out;
}

function rowsMask(rows: number[], cols: number, total: number): Uint8Array {
  const mask = new Uint8Array(total * cols);
  for (const r of rows) mask.fill(1, r * cols, (r + 1) * cols);
  return mask;
}

describe('bridgeGarmentGaps', () => {
  const cols = 4;
  const rows = 8;
  const index = strip(rows, cols);

  it('hides the single vertex row between two footprints (low shoe collar under a trouser hem)', () => {
    // shoe covers rows 0-3, trousers rows 5-7: row 4 is the sliver of skin between them
    const shoe = rowsMask([0, 1, 2, 3], cols, rows);
    const trousers = rowsMask([5, 6, 7], cols, rows);
    const hidden = new Uint8Array(rows * cols);
    for (let v = 0; v < hidden.length; v++) hidden[v] = shoe[v]! | trousers[v]!;
    const added = bridgeGarmentGaps(index, [shoe, trousers], hidden);
    expect(added).toBe(cols);
    expect(Array.from(hidden.subarray(4 * cols, 5 * cols))).toEqual([1, 1, 1, 1]);
  });

  it('keeps a real bare-ankle gap of two or more rows visible (shortened trousers)', () => {
    const shoe = rowsMask([0, 1, 2], cols, rows);
    const trousers = rowsMask([6, 7], cols, rows);
    const hidden = new Uint8Array(rows * cols);
    for (let v = 0; v < hidden.length; v++) hidden[v] = shoe[v]! | trousers[v]!;
    const before = Array.from(hidden);
    expect(bridgeGarmentGaps(index, [shoe, trousers], hidden)).toBe(0);
    expect(Array.from(hidden)).toEqual(before);
  });

  it('does nothing with a single garment, with no garment, or when the two overlap without a gap', () => {
    const shoe = rowsMask([0, 1, 2, 3], cols, rows);
    const hidden = Uint8Array.from(shoe);
    expect(bridgeGarmentGaps(index, [shoe], hidden)).toBe(0);
    expect(bridgeGarmentGaps(index, [], hidden)).toBe(0);
    const trousers = rowsMask([2, 3, 4, 5], cols, rows);
    const merged = new Uint8Array(rows * cols);
    for (let v = 0; v < merged.length; v++) merged[v] = shoe[v]! | trousers[v]!;
    expect(bridgeGarmentGaps(index, [shoe, trousers], merged)).toBe(0);
  });

  it('never touches a vertex that is already hidden and does not cascade into the next row', () => {
    const a = rowsMask([0, 1], cols, rows);
    const b = rowsMask([3, 4], cols, rows);
    const hidden = new Uint8Array(rows * cols);
    for (let v = 0; v < hidden.length; v++) hidden[v] = a[v]! | b[v]!;
    bridgeGarmentGaps(index, [a, b], hidden);
    expect(Array.from(hidden.subarray(2 * cols, 3 * cols))).toEqual([1, 1, 1, 1]);
    expect(Array.from(hidden.subarray(5 * cols))).toEqual(new Array(3 * cols).fill(0));
  });
});
