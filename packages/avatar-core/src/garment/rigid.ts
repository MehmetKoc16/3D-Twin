import type { Vec3 } from '../contracts';
import type { GarmentRigidAttachments } from '../garmentContracts';

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Parse the importer sidecar without trusting JSON casts or diagnostic metadata. */
export function parseGarmentRigidAttachments(
  value: unknown,
  vertexCount: number,
): GarmentRigidAttachments {
  if (
    !Number.isInteger(vertexCount) ||
    vertexCount < 0 ||
    !record(value) ||
    value.version !== 1 ||
    typeof value.soleHeightMm !== 'number' ||
    !Number.isFinite(value.soleHeightMm) ||
    value.soleHeightMm <= 0 ||
    !Array.isArray(value.attachments)
  )
    throw new Error('garment rigid: invalid sidecar');
  const seen = new Set<number>();
  const attachments: GarmentRigidAttachments['attachments'] = [];
  for (const entry of value.attachments) {
    if (
      !record(entry) ||
      (entry.bone !== 'foot_l' &&
        entry.bone !== 'ball_l' &&
        entry.bone !== 'foot_r' &&
        entry.bone !== 'ball_r') ||
      !Array.isArray(entry.vertices) ||
      !Array.isArray(entry.localPositionsM) ||
      entry.vertices.length !== entry.localPositionsM.length ||
      entry.vertices.length === 0
    )
      throw new Error('garment rigid: invalid attachment');
    const vertices: number[] = [];
    const localPositionsM: Vec3[] = [];
    for (let i = 0; i < entry.vertices.length; i++) {
      const v: unknown = entry.vertices[i];
      const p: unknown = entry.localPositionsM[i];
      if (typeof v !== 'number' || !Number.isInteger(v) || v < 0 || v >= vertexCount || seen.has(v))
        throw new Error('garment rigid: duplicate or out-of-range vertex');
      if (
        !Array.isArray(p) ||
        p.length !== 3 ||
        !p.every((n: unknown) => typeof n === 'number' && Number.isFinite(n))
      )
        throw new Error('garment rigid: invalid local position');
      seen.add(v);
      vertices.push(v);
      localPositionsM.push([p[0] as number, p[1] as number, p[2] as number]);
    }
    attachments.push({ bone: entry.bone, vertices, localPositionsM });
  }
  return { version: 1, soleHeightMm: value.soleHeightMm, attachments };
}

export interface PreparedGarmentRigidAttachments {
  /** Skeleton index of each attached vertex; -1 for the unaffected upper. */
  bones: Int32Array;
  /** One for all authored sole vertices; smoothstep fade on the upper boundary band. */
  blend: Float32Array;
  localPositions: Float32Array;
}

/**
 * Extend attachments through a 10 mm geodesic band on the upper. The authored sole remains fully rigid.
 * Zero-length links weld UV copies, keeping the boundary continuous across texture seams.
 */
export function prepareGarmentRigidAttachments(
  sidecar: GarmentRigidAttachments,
  neutral: ArrayLike<number>,
  index: ArrayLike<number>,
  boneNames: readonly string[],
  bandM = 0.01,
): PreparedGarmentRigidAttachments {
  const count = neutral.length / 3;
  if (!Number.isInteger(count) || !Number.isFinite(bandM) || bandM <= 0 || index.length % 3 !== 0)
    throw new Error('garment rigid: invalid geometry or boundary band');
  const parsed = parseGarmentRigidAttachments(sidecar, count);
  const bones = new Int32Array(count).fill(-1);
  const blend = new Float32Array(count);
  const authored = new Uint8Array(count);
  const localPositions = new Float32Array(neutral.length);
  const heads = new Map<number, Vec3>();
  const distances = new Float64Array(count).fill(Infinity);
  const queue: number[] = [];
  for (const attachment of parsed.attachments) {
    const bone = boneNames.indexOf(attachment.bone);
    if (bone < 0 || bone > 65535) throw new Error(`garment rigid: missing bone ${attachment.bone}`);
    attachment.vertices.forEach((v, i) => {
      const p = attachment.localPositionsM[i]!;
      const head: Vec3 = [
        neutral[v * 3]! - p[0],
        neutral[v * 3 + 1]! - p[1],
        neutral[v * 3 + 2]! - p[2],
      ];
      const previous = heads.get(bone);
      if (
        head.some(
          (n, axis) => !Number.isFinite(n) || (previous && Math.abs(n - previous[axis]!) > 0.0001),
        )
      )
        throw new Error('garment rigid: local positions disagree with the neutral mesh');
      if (!previous) heads.set(bone, head);
      bones[v] = bone;
      blend[v] = 1;
      authored[v] = 1;
      distances[v] = 0;
      localPositions.set(p, v * 3);
      queue.push(v);
    });
  }
  const neighbors: number[][] = Array.from({ length: count }, () => []);
  const link = (a: number, b: number): void => {
    if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b < 0 || a >= count || b >= count)
      throw new Error('garment rigid: invalid triangle index');
    neighbors[a]!.push(b);
    neighbors[b]!.push(a);
  };
  for (let t = 0; t < index.length; t += 3) {
    link(index[t]!, index[t + 1]!);
    link(index[t + 1]!, index[t + 2]!);
    link(index[t + 2]!, index[t]!);
  }
  const weld = new Map<string, number>();
  for (let v = 0; v < count; v++) {
    const p = [neutral[v * 3]!, neutral[v * 3 + 1]!, neutral[v * 3 + 2]!];
    if (p.some((n) => !Number.isFinite(n)))
      throw new Error('garment rigid: non-finite neutral position');
    const key = p.map((n) => Math.round(n * 1e6)).join(',');
    const other = weld.get(key);
    if (other === undefined) weld.set(key, v);
    else link(v, other);
  }
  for (let q = 0; q < queue.length; q++) {
    const a = queue[q]!;
    for (const b of neighbors[a]!) {
      if (distances[b] === 0) continue; // Never soften or reassign an authored sole segment.
      const distance =
        distances[a]! +
        Math.hypot(
          neutral[a * 3]! - neutral[b * 3]!,
          neutral[a * 3 + 1]! - neutral[b * 3 + 1]!,
          neutral[a * 3 + 2]! - neutral[b * 3 + 2]!,
        );
      if (distance >= bandM || distance >= distances[b]! - 1e-12) continue;
      distances[b] = distance;
      bones[b] = bones[a]!;
      queue.push(b);
    }
  }
  for (let v = 0; v < count; v++) {
    if (bones[v]! < 0 || authored[v] === 1) continue;
    const head = heads.get(bones[v]!)!;
    const t = 1 - distances[v]! / bandM;
    blend[v] = t * t * (3 - 2 * t);
    for (let axis = 0; axis < 3; axis++)
      localPositions[v * 3 + axis] = neutral[v * 3 + axis]! - head[axis]!;
  }
  return { bones, blend, localPositions };
}

/** Replace sole rest geometry after grading; scale only once, before GPU rigid bone transforms. */
export function fitGarmentRigidAttachments(
  rigid: PreparedGarmentRigidAttachments,
  scale: Vec3,
  joint: (boneIndex: number) => Vec3 | undefined,
  positions: Float32Array,
): void {
  if (
    positions.length !== rigid.localPositions.length ||
    scale.some((n) => !Number.isFinite(n) || n <= 0)
  )
    throw new Error('garment rigid: invalid positions or body scale');
  for (let v = 0; v < rigid.bones.length; v++) {
    const alpha = rigid.blend[v]!;
    if (alpha === 0) continue;
    const head = joint(rigid.bones[v]!);
    if (!head || head.some((n) => !Number.isFinite(n)))
      throw new Error('garment rigid: missing rest joint');
    for (let axis = 0; axis < 3; axis++) {
      const p = v * 3 + axis;
      const target = head[axis]! + rigid.localPositions[p]! * scale[axis]!;
      positions[p] = positions[p]! * (1 - alpha) + target * alpha;
    }
  }
}

/** Preserve MHCLO weights outside the band, replace soles, and mix influences in the boundary ring. */
export function garmentRigidSkinWeights(
  rigid: PreparedGarmentRigidAttachments,
  baseIndices: Uint16Array,
  baseWeights: Float32Array,
): { skinIndices: Uint16Array; skinWeights: Float32Array } {
  if (baseIndices.length !== rigid.bones.length * 4 || baseWeights.length !== baseIndices.length)
    throw new Error('garment rigid: invalid skin buffers');
  const skinIndices = baseIndices.slice();
  const skinWeights = baseWeights.slice();
  for (let v = 0; v < rigid.bones.length; v++) {
    const alpha = rigid.blend[v]!;
    if (alpha === 0) continue;
    const influences = new Map<number, number>([[rigid.bones[v]!, alpha]]);
    for (let k = 0; k < 4; k++) {
      const p = v * 4 + k;
      const weight = baseWeights[p]! * (1 - alpha);
      if (weight > 0)
        influences.set(baseIndices[p]!, (influences.get(baseIndices[p]!) ?? 0) + weight);
    }
    const top = [...influences].sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, 4);
    const sum = top.reduce((total, [, weight]) => total + weight, 0);
    skinIndices.fill(0, v * 4, v * 4 + 4);
    skinWeights.fill(0, v * 4, v * 4 + 4);
    top.forEach(([bone, weight], k) => {
      skinIndices[v * 4 + k] = bone;
      skinWeights[v * 4 + k] = weight / sum;
    });
  }
  return { skinIndices, skinWeights };
}
