export const PUSH_MARGIN_M = 0.025;
export const MAX_PUSH_M = 0.008;

/** Spatially bounded distance to covered vertices; smoothstep gives zero slope at both ends of the margin. */
export function pushInWeights(positions: Float32Array, covered: Uint8Array): Float32Array {
  const weights = new Float32Array(covered.length);
  const grid = new Map<string, number[]>();
  const cell = (value: number): number => Math.floor(value / PUSH_MARGIN_M);
  const key = (x: number, y: number, z: number): string => `${x},${y},${z}`;
  for (let v = 0; v < covered.length; v++) {
    if (!covered[v]) continue;
    const k = key(
      cell(positions[v * 3]!),
      cell(positions[v * 3 + 1]!),
      cell(positions[v * 3 + 2]!),
    );
    const list = grid.get(k);
    if (list) list.push(v);
    else grid.set(k, [v]);
  }
  if (grid.size === 0) return weights;
  for (let v = 0; v < covered.length; v++) {
    if (covered[v]) {
      weights[v] = 1;
      continue;
    }
    const x = positions[v * 3]!,
      y = positions[v * 3 + 1]!,
      z = positions[v * 3 + 2]!;
    let distance = PUSH_MARGIN_M;
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++) {
          for (const other of grid.get(key(cell(x) + dx, cell(y) + dy, cell(z) + dz)) ?? [])
            distance = Math.min(
              distance,
              Math.hypot(
                x - positions[other * 3]!,
                y - positions[other * 3 + 1]!,
                z - positions[other * 3 + 2]!,
              ),
            );
        }
    const t = 1 - distance / PUSH_MARGIN_M;
    weights[v] = t * t * (3 - 2 * t);
  }
  return weights;
}

/** Always reads the unmodified rest mesh: repeated refreshes never accumulate displacement. */
export function pushInPositions(
  rest: Float32Array,
  normals: Float32Array,
  weights: Float32Array,
  out: Float32Array,
): void {
  for (let v = 0; v < weights.length; v++) {
    const p = v * 3;
    const length = Math.hypot(normals[p]!, normals[p + 1]!, normals[p + 2]!);
    const scale = length > 1e-8 ? (MAX_PUSH_M * Math.max(0, Math.min(1, weights[v]!))) / length : 0;
    for (let j = 0; j < 3; j++) out[p + j] = rest[p + j]! - normals[p + j]! * scale;
  }
}
