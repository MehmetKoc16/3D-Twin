import type { GarmentBinding } from './binding';

/** Blends the bound render vertices' bone influences and retains the four strongest. */
export function garmentSkinWeights(
  binding: GarmentBinding,
  bodySkinIndices: ArrayLike<number>,
  bodySkinWeights: ArrayLike<number>,
): { skinIndices: Uint16Array; skinWeights: Float32Array } {
  const skinIndices = new Uint16Array(binding.count * 4);
  const skinWeights = new Float32Array(binding.count * 4);
  for (let v = 0; v < binding.count; v++) {
    const influences = new Map<number, number>();
    for (let j = 0; j < 3; j++) {
      const p = v * 3 + j;
      const base = binding.indices[p]! * 4;
      if (base + 3 >= bodySkinIndices.length || base + 3 >= bodySkinWeights.length)
        throw new Error(`garment skin: bound vertex ${v} has no render skin weights`);
      for (let k = 0; k < 4; k++) {
        const bone = bodySkinIndices[base + k]!;
        const weight = binding.weights[p]! * bodySkinWeights[base + k]!;
        if (
          !Number.isInteger(bone) ||
          bone < 0 ||
          bone > 65535 ||
          !Number.isFinite(weight) ||
          weight < 0
        )
          throw new Error(`garment skin: invalid influence at vertex ${v}`);
        if (weight > 0) influences.set(bone, (influences.get(bone) ?? 0) + weight);
      }
    }
    const top = [...influences].sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, 4);
    const sum = top.reduce((total, [, weight]) => total + weight, 0);
    if (sum <= 0) throw new Error(`garment skin: no influences at vertex ${v}`);
    for (let k = 0; k < top.length; k++) {
      skinIndices[v * 4 + k] = top[k]![0];
      skinWeights[v * 4 + k] = top[k]![1] / sum;
    }
  }
  return { skinIndices, skinWeights };
}
