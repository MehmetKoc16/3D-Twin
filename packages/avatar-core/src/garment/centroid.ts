/** Approximate a horizontal body section by averaging body vertices in 1 cm height bins. */
export function sectionCentroids(
  bodyPositions: ArrayLike<number>,
): Map<number, [number, number, number]> {
  const bins = new Map<number, [number, number, number]>();
  for (let i = 0; i + 2 < bodyPositions.length; i += 3) {
    const key = Math.round(bodyPositions[i + 1]! * 100);
    const bin = bins.get(key) ?? [0, 0, 0];
    bin[0] += bodyPositions[i]!;
    bin[1] += bodyPositions[i + 2]!;
    bin[2]++;
    bins.set(key, bin);
  }
  for (const bin of bins.values()) {
    bin[0] /= bin[2];
    bin[1] /= bin[2];
  }
  return bins;
}

/** Nearest occupied section, accounting for gaps in sparse meshes. */
export function centroidAt(
  bins: Map<number, [number, number, number]>,
  y: number,
): [number, number] {
  const key = Math.round(y * 100);
  const direct = bins.get(key);
  if (direct) return [direct[0], direct[1]];
  let bestDistance = Infinity;
  let best: [number, number, number] | undefined;
  for (const [candidate, bin] of bins) {
    const distance = Math.abs(candidate - key);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = bin;
    }
  }
  return best ? [best[0], best[1]] : [0, 0];
}
