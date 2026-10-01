import type { Vec3 } from '@dt/avatar-core';
import { TwinFormatError } from './twinDef';

/**
 * Binding the twin mesh to the avatar's skeleton (pure maths, unit tested).
 *
 * rigged.glb follows the contract of base.glb: world-aligned bones (identity rest rotation) and inverse bind matrices
 * that are pure translations of -head. The avatar's skeleton (the single source of truth for poses, camera focus and
 * garments) has the same bones, but its heads come from the hidden MakeHuman body solved in the browser. The twin
 * therefore keeps its own vertices and skin weights, is re-indexed to the avatar's bone order, and is translated by
 * the (nearly constant) offset between the two sets of rest heads; the avatar's inverse bind matrices then make every
 * pose act on it exactly as it acts on the body.
 */

/** A 4x4 matrix in column-major order (three.js `Matrix4`). */
interface MatrixLike {
  elements: ArrayLike<number>;
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];

/**
 * Rest heads (3 floats per bone, in the skin's joint order) from inverse bind matrices that are translations of
 * -head. Throws `TwinFormatError('rest')` when a matrix has a rotation or scale (the pipeline contract is
 * world-aligned identity rest rotations).
 */
export function headsFromBoneInverses(
  boneInverses: readonly MatrixLike[],
  tolerance = 1e-4,
): Float32Array {
  const heads = new Float32Array(boneInverses.length * 3);
  boneInverses.forEach((m, i) => {
    const e = m.elements;
    for (let k = 0; k < IDENTITY.length; k++)
      if (Math.abs(e[k]! - IDENTITY[k]!) > tolerance)
        throw new TwinFormatError(
          'rest',
          `bone ${i}: the inverse bind matrix is not a pure translation`,
        );
    heads[i * 3] = -e[12]!;
    heads[i * 3 + 1] = -e[13]!;
    heads[i * 3 + 2] = -e[14]!;
  });
  return heads;
}

/**
 * `remap[t]` = index in the avatar's bone order of the twin's bone t (matched by name). The two skeletons must have
 * exactly the same bones; `expectedOrder` (twin.json `boneOrder`) must equal the twin's skin joint order.
 */
export function matchBones(
  twinNames: readonly string[],
  appNames: readonly string[],
  expectedOrder?: readonly string[],
): Int32Array {
  if (
    expectedOrder &&
    (expectedOrder.length !== twinNames.length || expectedOrder.some((n, i) => n !== twinNames[i]))
  )
    throw new TwinFormatError(
      'mismatch',
      'twin.json boneOrder differs from the joint order of rigged.glb',
    );
  if (twinNames.length !== appNames.length)
    throw new TwinFormatError(
      'bones',
      `the twin has ${twinNames.length} bones, the avatar ${appNames.length}`,
    );
  const appIndex = new Map(appNames.map((n, i) => [n, i]));
  const remap = new Int32Array(twinNames.length);
  twinNames.forEach((name, t) => {
    const i = appIndex.get(name);
    if (i === undefined)
      throw new TwinFormatError('bones', `bone "${name}" does not exist in the avatar skeleton`);
    remap[t] = i;
  });
  if (new Set(remap).size !== remap.length)
    throw new TwinFormatError('bones', 'duplicate bone names in the twin');
  return remap;
}

/** Skin indices of the twin (its joint order) -> the avatar's bone order. */
export function remapSkinIndices(skinIndex: ArrayLike<number>, remap: Int32Array): Uint16Array {
  const out = new Uint16Array(skinIndex.length);
  for (let i = 0; i < skinIndex.length; i++) {
    const mapped = remap[skinIndex[i]!];
    if (mapped === undefined)
      throw new TwinFormatError('noSkin', `skin index ${skinIndex[i]} is outside the skeleton`);
    out[i] = mapped;
  }
  return out;
}

/** Renormalises 4-influence weights in place; throws when a vertex has no usable weight. */
export function normalizeSkinWeights(weights: Float32Array): void {
  for (let v = 0; v < weights.length; v += 4) {
    let sum = 0;
    for (let k = 0; k < 4; k++) {
      const w = weights[v + k]!;
      if (!Number.isFinite(w) || w < 0)
        throw new TwinFormatError('noSkin', `vertex ${v / 4}: invalid skin weight`);
      sum += w;
    }
    if (sum < 1e-6) throw new TwinFormatError('noSkin', `vertex ${v / 4}: no skin weight`);
    if (Math.abs(sum - 1) > 1e-6)
      for (let k = 0; k < 4; k++) weights[v + k] = weights[v + k]! / sum;
  }
}

export interface RestAlignment {
  /** Translation that carries the twin's rest frame onto the avatar's (metres). */
  offset: Vec3;
  /** Largest remaining head distance after the translation (metres). */
  maxResidual: number;
}

/**
 * Translation between the twin's and the avatar's rest heads (mean over all bones) and what is left of the
 * difference. With a twin.json that matches its glb the residual is numerical noise; a large one means the two files
 * do not belong together.
 *
 * @param twinHeads 3 floats per bone in the twin's joint order
 * @param remap twin bone index -> avatar bone index
 * @param appHeads 3 floats per bone in the avatar's bone order
 */
export function restAlignment(
  twinHeads: ArrayLike<number>,
  remap: Int32Array,
  appHeads: ArrayLike<number>,
): RestAlignment {
  const n = remap.length;
  const offset: Vec3 = [0, 0, 0];
  for (let t = 0; t < n; t++) {
    const a = remap[t]! * 3;
    for (let k = 0; k < 3; k++)
      offset[k] = offset[k]! + (appHeads[a + k]! - twinHeads[t * 3 + k]!) / n;
  }
  let maxResidual = 0;
  for (let t = 0; t < n; t++) {
    const a = remap[t]! * 3;
    const dx = appHeads[a]! - twinHeads[t * 3]! - offset[0];
    const dy = appHeads[a + 1]! - twinHeads[t * 3 + 1]! - offset[1];
    const dz = appHeads[a + 2]! - twinHeads[t * 3 + 2]! - offset[2];
    maxResidual = Math.max(maxResidual, Math.hypot(dx, dy, dz));
  }
  return { offset, maxResidual };
}

/** out = base + offset for every vertex (3 floats per vertex). */
export function translatePositions(base: ArrayLike<number>, offset: Vec3, out: Float32Array): void {
  for (let i = 0; i < base.length; i += 3) {
    out[i] = base[i]! + offset[0];
    out[i + 1] = base[i + 1]! + offset[1];
    out[i + 2] = base[i + 2]! + offset[2];
  }
}
