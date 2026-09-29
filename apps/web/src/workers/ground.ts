/** Lowest y over the first `renderCount` vertices of a position array. */
export function minY(positions: ArrayLike<number>, renderCount: number): number {
  let m = Infinity;
  for (let v = 0; v < renderCount; v++) {
    const y = positions[v * 3 + 1]!;
    if (y < m) m = y;
  }
  return m;
}

/**
 * Copies the render vertices of `full` (render vertices ++ joint points) into a new array translated so the lowest
 * render vertex sits at y = 0. Leg-height morphs push the feet below y = 0, so the runtime must always re-ground.
 */
export function groundRenderPositions(
  full: ArrayLike<number>,
  renderCount: number,
): { positions: Float32Array; offsetY: number } {
  const offsetY = minY(full, renderCount);
  const positions = new Float32Array(renderCount * 3);
  for (let v = 0; v < renderCount; v++) {
    positions[v * 3] = full[v * 3]!;
    positions[v * 3 + 1] = full[v * 3 + 1]! - offsetY;
    positions[v * 3 + 2] = full[v * 3 + 2]!;
  }
  return { positions, offsetY };
}

/** Flattens joints (rig order) into 6 floats per bone, translated down by `offsetY`. */
export function flattenJoints(
  order: readonly string[],
  joints: ReadonlyMap<string, { head: readonly number[]; tail: readonly number[] }>,
  offsetY: number,
): Float32Array {
  const out = new Float32Array(order.length * 6);
  order.forEach((name, i) => {
    const j = joints.get(name);
    if (!j) throw new Error(`missing joint for bone ${name}`);
    out[i * 6] = j.head[0]!;
    out[i * 6 + 1] = j.head[1]! - offsetY;
    out[i * 6 + 2] = j.head[2]!;
    out[i * 6 + 3] = j.tail[0]!;
    out[i * 6 + 4] = j.tail[1]! - offsetY;
    out[i * 6 + 5] = j.tail[2]!;
  });
  return out;
}
