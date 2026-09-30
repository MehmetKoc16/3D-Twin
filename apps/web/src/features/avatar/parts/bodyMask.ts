/** Hiding body triangles behind the parts (eye-socket cavity, scalp under a hair cap); pure, no three.js. */

interface WeldLike {
  groupOf: ArrayLike<number>;
  start: ArrayLike<number>;
  members: ArrayLike<number>;
}

/**
 * Marks the deleted body render vertices and every UV-seam copy of them: the pipeline lists MakeHuman vertices, but
 * the render mesh holds one copy per UV island.
 */
export function hiddenVertexMask(
  deleteLists: readonly ArrayLike<number>[],
  vertexCount: number,
  weld?: WeldLike,
): Uint8Array {
  const mask = new Uint8Array(vertexCount);
  for (const list of deleteLists) {
    for (let i = 0; i < list.length; i++) {
      const v = list[i]!;
      if (v >= vertexCount) continue;
      if (weld) {
        const g = weld.groupOf[v]!;
        for (let m = weld.start[g]!; m < weld.start[g + 1]!; m++) mask[weld.members[m]!] = 1;
      } else {
        mask[v] = 1;
      }
    }
  }
  return mask;
}

/** The triangle list without the triangles whose three vertices are all hidden; `indices` itself if none. */
export function filterBodyIndex(indices: ArrayLike<number>, hidden: Uint8Array): ArrayLike<number> {
  let kept = 0;
  for (let t = 0; t < indices.length; t += 3) {
    if (!(hidden[indices[t]!] && hidden[indices[t + 1]!] && hidden[indices[t + 2]!])) kept += 3;
  }
  if (kept === indices.length) return indices;
  const out = new Uint32Array(kept);
  let o = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t]!;
    const b = indices[t + 1]!;
    const c = indices[t + 2]!;
    if (hidden[a] && hidden[b] && hidden[c]) continue;
    out[o++] = a;
    out[o++] = b;
    out[o++] = c;
  }
  return out;
}

/** Parses a `.delete.bin` (little-endian uint32 list). */
export function parseDeleteVerts(buffer: ArrayBuffer): Uint32Array {
  if (buffer.byteLength % 4 !== 0) throw new Error('delete list: byte length must be a multiple of 4');
  const view = new DataView(buffer);
  const out = new Uint32Array(buffer.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getUint32(i * 4, true);
  return out;
}
