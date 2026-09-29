import type { GarmentScaleRefs } from '../garmentContracts';

export interface GarmentBinding {
  count: number;
  indices: Uint32Array;
  weights: Float32Array;
  offsets: Float32Array;
}

/** The exporter rounds float32 weights; sums within 0.05 of one are accepted without changing them. */
export function parseGarmentBinding(buf: ArrayBuffer, vertexCount?: number): GarmentBinding {
  if (buf.byteLength % 36 !== 0)
    throw new Error('garment binding: byte length must be a multiple of 36');
  if (vertexCount !== undefined && (!Number.isInteger(vertexCount) || vertexCount < 0))
    throw new Error('garment binding: invalid body vertex count');
  const count = buf.byteLength / 36;
  const indices = new Uint32Array(count * 3);
  const weights = new Float32Array(count * 3);
  const offsets = new Float32Array(count * 3);
  const view = new DataView(buf);
  for (let v = 0; v < count; v++) {
    const p = v * 36;
    let sum = 0;
    for (let j = 0; j < 3; j++) {
      const index = view.getUint32(p + j * 4, true);
      const weight = view.getFloat32(p + 12 + j * 4, true);
      const offset = view.getFloat32(p + 24 + j * 4, true);
      if (vertexCount !== undefined && index >= vertexCount)
        throw new Error(`garment binding: body index out of range at vertex ${v}`);
      if (!Number.isFinite(weight) || !Number.isFinite(offset) || weight < 0)
        throw new Error(`garment binding: non-finite or negative value at vertex ${v}`);
      indices[v * 3 + j] = index;
      weights[v * 3 + j] = weight;
      offsets[v * 3 + j] = offset;
      sum += weight;
    }
    if (Math.abs(sum - 1) > 0.05)
      throw new Error(`garment binding: weights do not sum to one at vertex ${v}`);
  }
  return { count, indices, weights, offsets };
}

/** MHCLO axis scale: absolute coordinate difference divided by the reference distance. */
function axisScale(
  ref: [number, number, number],
  axis: number,
  bodyPositions: ArrayLike<number>,
): number {
  const [a, b, refM] = ref;
  if (
    !Number.isInteger(a) ||
    !Number.isInteger(b) ||
    a < 0 ||
    b < 0 ||
    a * 3 + axis >= bodyPositions.length ||
    b * 3 + axis >= bodyPositions.length ||
    !Number.isFinite(refM) ||
    refM <= 0
  )
    throw new Error('garment scale: invalid reference');
  return Math.abs(bodyPositions[a * 3 + axis]! - bodyPositions[b * 3 + axis]!) / refM;
}

export function computeScale(
  scaleRefs: GarmentScaleRefs | undefined,
  bodyPositions: ArrayLike<number>,
): [number, number, number] {
  if (!scaleRefs) return [1, 1, 1];
  return [
    axisScale(scaleRefs.x, 0, bodyPositions),
    axisScale(scaleRefs.y, 1, bodyPositions),
    axisScale(scaleRefs.z, 2, bodyPositions),
  ];
}

/** Writes current garment rest positions into a caller-owned buffer without allocating. */
export function bindGarment(
  binding: GarmentBinding,
  scaleRefs: GarmentScaleRefs | undefined,
  bodyPositions: ArrayLike<number>,
  out: Float32Array,
): void {
  if (out.length < binding.count * 3) throw new Error('garment binding: output too short');
  const sx = scaleRefs ? axisScale(scaleRefs.x, 0, bodyPositions) : 1;
  const sy = scaleRefs ? axisScale(scaleRefs.y, 1, bodyPositions) : 1;
  const sz = scaleRefs ? axisScale(scaleRefs.z, 2, bodyPositions) : 1;
  for (let v = 0; v < binding.count; v++) {
    const o = v * 3;
    const a = binding.indices[o]! * 3;
    const b = binding.indices[o + 1]! * 3;
    const c = binding.indices[o + 2]! * 3;
    if (
      a + 2 >= bodyPositions.length ||
      b + 2 >= bodyPositions.length ||
      c + 2 >= bodyPositions.length
    )
      throw new Error(`garment binding: body index out of range at vertex ${v}`);
    const wa = binding.weights[o]!;
    const wb = binding.weights[o + 1]!;
    const wc = binding.weights[o + 2]!;
    out[o] =
      wa * bodyPositions[a]! +
      wb * bodyPositions[b]! +
      wc * bodyPositions[c]! +
      binding.offsets[o]! * sx;
    out[o + 1] =
      wa * bodyPositions[a + 1]! +
      wb * bodyPositions[b + 1]! +
      wc * bodyPositions[c + 1]! +
      binding.offsets[o + 1]! * sy;
    out[o + 2] =
      wa * bodyPositions[a + 2]! +
      wb * bodyPositions[b + 2]! +
      wc * bodyPositions[c + 2]! +
      binding.offsets[o + 2]! * sz;
  }
}
