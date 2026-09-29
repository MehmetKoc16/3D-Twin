import type { BodyManifest } from './contracts';
import { AvatarCoreError } from './errors';

/** Size of one sparse morph entry in morphs.bin (uint32 vertexIndex, float32 dx, dy, dz; little-endian). */
export const MORPH_ENTRY_BYTES = 16;
const ENTRY_WORDS = MORPH_ENTRY_BYTES / 4;

const IS_LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/**
 * Parsed + validated morph data. Views alias the input buffer (no copy on little-endian hosts), so the buffer
 * must not be mutated afterwards. Target index = position in `manifest.targets`.
 */
export interface MorphSet {
  readonly vertexCount: number;
  /** Word view: entry k of a target starts at word wordOffsets[t] + 4k (word 0 = vertex index). */
  readonly words: Uint32Array;
  /** Same memory as `words`, read as float32 (words 1..3 of an entry = dx, dy, dz). */
  readonly floats: Float32Array;
  readonly targetIds: readonly string[];
  readonly targetIndex: ReadonlyMap<string, number>;
  readonly wordOffsets: Int32Array;
  readonly counts: Int32Array;
}

/** Structural checks of a manifest; throws AvatarCoreError with a precise message. */
export function validateManifest(m: BodyManifest): void {
  const fail = (msg: string): never => {
    throw new AvatarCoreError(`manifest: ${msg}`);
  };
  if (m.version !== 1) fail(`unsupported version ${String(m.version)}`);
  if (m.unit !== 'm') fail(`unsupported unit ${String(m.unit)}`);
  if (!Number.isInteger(m.renderVertexCount) || m.renderVertexCount <= 0)
    fail(`invalid renderVertexCount ${m.renderVertexCount}`);
  if (m.vertexCount !== m.renderVertexCount + m.jointPoints.length)
    fail(
      `vertexCount ${m.vertexCount} != renderVertexCount ${m.renderVertexCount} + jointPoints ${m.jointPoints.length}`,
    );

  const targetIds = new Set<string>();
  for (const t of m.targets) {
    if (targetIds.has(t.id)) fail(`duplicate target id "${t.id}"`);
    targetIds.add(t.id);
    if (!Number.isInteger(t.byteOffset) || t.byteOffset < 0)
      fail(`target "${t.id}": invalid byteOffset ${t.byteOffset}`);
    if (!Number.isInteger(t.count) || t.count < 0)
      fail(`target "${t.id}": invalid count ${t.count}`);
  }

  const varMap = new Map<string, Set<string>>();
  for (const v of m.macroVariables) {
    if (varMap.has(v.id)) fail(`duplicate macro variable "${v.id}"`);
    if (!(v.min <= v.max)) fail(`macro variable "${v.id}": min > max`);
    if (v.buckets.length === 0) fail(`macro variable "${v.id}": no buckets`);
    const names = new Set<string>();
    const ats = new Set<number>();
    for (const b of v.buckets) {
      if (!Number.isFinite(b.at))
        fail(`macro variable "${v.id}": bucket "${b.name}" has non-finite position`);
      if (names.has(b.name)) fail(`macro variable "${v.id}": duplicate bucket "${b.name}"`);
      if (ats.has(b.at)) fail(`macro variable "${v.id}": two buckets at ${b.at}`);
      names.add(b.name);
      ats.add(b.at);
    }
    varMap.set(v.id, names);
  }
  for (const t of m.targets) {
    for (const c of t.macroConditions ?? []) {
      const names = varMap.get(c.variable);
      if (!names) fail(`target "${t.id}": unknown macro variable "${c.variable}"`);
      else if (!names.has(c.bucket))
        fail(`target "${t.id}": unknown bucket "${c.bucket}" of variable "${c.variable}"`);
    }
  }

  const modIds = new Set<string>();
  for (const md of m.modifiers) {
    if (modIds.has(md.id)) fail(`duplicate modifier id "${md.id}"`);
    modIds.add(md.id);
    if (!(md.min <= md.max)) fail(`modifier "${md.id}": min > max`);
    for (const ref of [md.decrTarget, md.incrTarget]) {
      if (ref !== undefined && !targetIds.has(ref))
        fail(`modifier "${md.id}": unknown target "${ref}"`);
    }
  }
}

/** Builds the full base position array (length 3 * vertexCount): glTF render positions + joint points. */
export function buildBasePositions(
  gltfPositions: Float32Array,
  manifest: BodyManifest,
): Float32Array {
  if (gltfPositions.length !== manifest.renderVertexCount * 3)
    throw new AvatarCoreError(
      `glTF positions length ${gltfPositions.length} != 3 * renderVertexCount ${manifest.renderVertexCount}`,
    );
  if (manifest.vertexCount !== manifest.renderVertexCount + manifest.jointPoints.length)
    throw new AvatarCoreError('manifest: vertexCount != renderVertexCount + jointPoints.length');
  const out = new Float32Array(manifest.vertexCount * 3);
  out.set(gltfPositions);
  manifest.jointPoints.forEach((jp, i) => {
    const o = (manifest.renderVertexCount + i) * 3;
    out[o] = jp.position[0];
    out[o + 1] = jp.position[1];
    out[o + 2] = jp.position[2];
  });
  return out;
}

function toNativeEndian(buffer: ArrayBufferLike): ArrayBufferLike {
  if (IS_LITTLE_ENDIAN) return buffer;
  const src = new Uint8Array(buffer);
  const out = new Uint8Array(src.length);
  for (let i = 0; i + 3 < src.length; i += 4) {
    out[i] = src[i + 3]!;
    out[i + 1] = src[i + 2]!;
    out[i + 2] = src[i + 1]!;
    out[i + 3] = src[i]!;
  }
  return out.buffer;
}

/**
 * Parses morphs.bin against the manifest. Validates every target run (alignment, bounds), every vertex index
 * (< vertexCount) and every delta (finite), so the hot path needs no checks.
 */
export function parseMorphs(buffer: ArrayBufferLike, manifest: BodyManifest): MorphSet {
  validateManifest(manifest);
  const native = toNativeEndian(buffer);
  const wordCount = Math.floor(native.byteLength / 4);
  const words = new Uint32Array(native, 0, wordCount);
  const floats = new Float32Array(native, 0, wordCount);
  const n = manifest.targets.length;
  const wordOffsets = new Int32Array(n);
  const counts = new Int32Array(n);
  const targetIndex = new Map<string, number>();
  const targetIds: string[] = [];
  const vc = manifest.vertexCount;

  for (let t = 0; t < n; t++) {
    const def = manifest.targets[t]!;
    if (def.byteOffset % 4 !== 0)
      throw new AvatarCoreError(
        `morphs: target "${def.id}" byteOffset ${def.byteOffset} is not 4-byte aligned`,
      );
    const end = def.byteOffset + def.count * MORPH_ENTRY_BYTES;
    if (end > buffer.byteLength)
      throw new AvatarCoreError(
        `morphs: target "${def.id}" run [${def.byteOffset}, ${end}) exceeds morphs.bin size ${buffer.byteLength}`,
      );
    const w0 = def.byteOffset / 4;
    for (let k = 0, p = w0; k < def.count; k++, p += ENTRY_WORDS) {
      if (words[p]! >= vc)
        throw new AvatarCoreError(
          `morphs: target "${def.id}" entry ${k} has vertex index ${words[p]} >= vertexCount ${vc}`,
        );
      const s = floats[p + 1]! + floats[p + 2]! + floats[p + 3]!;
      if (!Number.isFinite(s))
        throw new AvatarCoreError(`morphs: target "${def.id}" entry ${k} has a non-finite delta`);
    }
    wordOffsets[t] = w0;
    counts[t] = def.count;
    targetIndex.set(def.id, t);
    targetIds.push(def.id);
  }
  return { vertexCount: vc, words, floats, targetIds, targetIndex, wordOffsets, counts };
}

function isMorphSet(x: ArrayBufferLike | MorphSet): x is MorphSet {
  return typeof x === 'object' && 'targetIndex' in x;
}

const parseCache = new WeakMap<object, WeakMap<BodyManifest, MorphSet>>();

/** Returns a parsed MorphSet, caching per (buffer, manifest) so repeated applyMorphs calls are cheap. */
export function getMorphSet(morphs: ArrayBufferLike | MorphSet, manifest: BodyManifest): MorphSet {
  if (isMorphSet(morphs)) {
    if (
      morphs.vertexCount !== manifest.vertexCount ||
      morphs.targetIds.length !== manifest.targets.length
    )
      throw new AvatarCoreError('morphs: MorphSet does not match manifest');
    return morphs;
  }
  let perManifest = parseCache.get(morphs);
  if (!perManifest) {
    perManifest = new WeakMap();
    parseCache.set(morphs, perManifest);
  }
  let set = perManifest.get(manifest);
  if (!set) {
    set = parseMorphs(morphs, manifest);
    perManifest.set(manifest, set);
  }
  return set;
}

/** out += weight * delta(target t). Hot path: no allocation, no bounds checks (validated at parse time). */
export function accumulateTarget(
  set: MorphSet,
  t: number,
  weight: number,
  out: Float32Array,
): void {
  if (weight === 0) return;
  const { words, floats } = set;
  let p = set.wordOffsets[t]!;
  const end = p + set.counts[t]! * ENTRY_WORDS;
  while (p < end) {
    const o = words[p]! * 3;
    out[o] = out[o]! + weight * floats[p + 1]!;
    out[o + 1] = out[o + 1]! + weight * floats[p + 2]!;
    out[o + 2] = out[o + 2]! + weight * floats[p + 3]!;
    p += ENTRY_WORDS;
  }
}

/** out = base + sum_t weights[t] * delta_t, with weights indexed by target index. */
export function applyWeightArray(
  set: MorphSet,
  base: Float32Array,
  weights: ArrayLike<number>,
  out: Float32Array,
): void {
  if (out !== base) out.set(base);
  for (let t = 0; t < set.counts.length; t++) {
    const w = weights[t]!;
    if (w !== 0) accumulateTarget(set, t, w, out);
  }
}

/**
 * out = base + sum_i weights[id_i] * delta_i (sparse accumulate). `out` may alias `base` (updated in place).
 * `morphs` is either the raw morphs.bin buffer (parsed once and cached) or a pre-parsed MorphSet.
 * Throws on unknown target ids, non-finite weights or size mismatches.
 */
export function applyMorphs(
  base: Float32Array,
  morphs: ArrayBufferLike | MorphSet,
  manifest: BodyManifest,
  weights: ReadonlyMap<string, number>,
  out: Float32Array,
): void {
  const set = getMorphSet(morphs, manifest);
  const len = set.vertexCount * 3;
  if (base.length !== len)
    throw new AvatarCoreError(`applyMorphs: base length ${base.length} != ${len}`);
  if (out.length !== len)
    throw new AvatarCoreError(`applyMorphs: out length ${out.length} != ${len}`);
  if (out !== base) out.set(base);
  weights.forEach((w, id) => {
    const t = set.targetIndex.get(id);
    if (t === undefined) throw new AvatarCoreError(`applyMorphs: unknown target "${id}"`);
    if (!Number.isFinite(w))
      throw new AvatarCoreError(`applyMorphs: non-finite weight for "${id}"`);
    accumulateTarget(set, t, w, out);
  });
}
