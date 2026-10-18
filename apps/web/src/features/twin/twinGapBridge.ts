/**
 * Closes the one-row skin gap between two worn garments (pure, no three.js).
 *
 * The footprint pass hides the twin vertices that lie under each garment. Where two garments nearly meet (low shoe
 * collar under a trouser hem, a short top above a waistband) the row of twin vertices that lies in between belongs to
 * neither footprint. Its triangles survive (a triangle is only dropped when all three vertices are hidden), and the
 * hidden neighbours are pushed inwards, so a sliver of skin shows through the few millimetres between the fabrics.
 * A vertex that touches footprint vertices of two DIFFERENT garments is exactly such a row; hiding it removes the
 * sliver. A real gap (shortened trousers, bare ankles) is at least a row wide: its vertices touch one footprint only
 * and stay visible.
 */

/** Largest number of garment surfaces tracked (one bit each). */
const MAX_GARMENTS = 30;

/**
 * Marks (in `hidden`, in place) every visible vertex whose neighbours are covered by at least two different garments.
 * `coverages` holds one byte per vertex per garment surface (the footprint masks). Returns how many were added.
 */
export function bridgeGarmentGaps(
  index: ArrayLike<number>,
  coverages: readonly ArrayLike<number>[],
  hidden: Uint8Array,
): number {
  if (coverages.length < 2) return 0;
  const garments = Math.min(coverages.length, MAX_GARMENTS);
  const bits = new Uint32Array(hidden.length);
  for (let g = 0; g < garments; g++) {
    const coverage = coverages[g]!;
    for (let v = 0; v < hidden.length; v++) if (coverage[v]) bits[v] = bits[v]! | (1 << g);
  }
  const around = new Uint32Array(hidden.length);
  for (let t = 0; t + 2 < index.length; t += 3) {
    const a = index[t]!;
    const b = index[t + 1]!;
    const c = index[t + 2]!;
    around[a] = around[a]! | bits[b]! | bits[c]!;
    around[b] = around[b]! | bits[a]! | bits[c]!;
    around[c] = around[c]! | bits[a]! | bits[b]!;
  }
  let added = 0;
  for (let v = 0; v < hidden.length; v++) {
    if (hidden[v]) continue;
    const mask = around[v]!;
    if ((mask & (mask - 1)) === 0) continue; // zero or one garment around it
    hidden[v] = 1;
    added++;
  }
  return added;
}
