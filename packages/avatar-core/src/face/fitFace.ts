import type { BodyManifest } from '../contracts';
import { AvatarCoreError } from '../errors';
import type { FaceMapDef } from '../faceContracts';
import { compileModifiers, mergeWeights, modifierWeights } from '../modifiers';
import { applyMorphs, getMorphSet, type MorphSet } from '../morphs';
import { solveLinear } from '../numeric';
import type { SolverData } from '../types';

/**
 * Face-shape fitting: finds values for the face modifiers listed in `FaceMapDef.fitModifiers` so that the frontal
 * (orthographic x, y) layout of the MakeHuman head matches the MediaPipe landmarks of the user's photo.
 *
 * Model. Every canonical landmark i is bound (face-map.json) to a point of the body surface: a triangle of render
 * vertices and barycentric weights. For modifier values `theta` the bound point is
 * `sum_k bary_k * (P_body[v_k] + sum_j delta_j(theta_j)[v_k])`, where `P_body` is the body that the caller has
 * already solved (`baseWeights`, e.g. `SolveResult.weights`) and `delta_j` the morph of face modifier j
 * (piecewise linear: `|theta_j| * decr` for theta_j < 0, `theta_j * incr` for theta_j > 0).
 *
 * Photo. Landmark i in pixels is `(x * imageWidth, -(y * imageHeight))`: MediaPipe's y axis points down, the
 * avatar's up, so y is flipped. The photo must be UN-mirrored (the subject's left is on the image right, which is
 * +X of the avatar). The landmark z is not used.
 *
 * Objective, per LM iteration:
 *   e(theta) = photo - Sim(model(theta))        Sim = the best 2-D similarity (translation, in-plane rotation,
 *                                               uniform scale), re-solved in closed form for every theta
 *   cost     = |e|^2 (in millimetres of the avatar head) + regularization * |theta|^2
 * i.e. head size, position and roll never leak into the shape values, and the values are pulled toward 0 (the
 * neutral head) by a Tikhonov term. Bounded Levenberg-Marquardt with a one-sided finite-difference Jacobian (the
 * morph model has a kink at 0, so the side follows the sign of each value).
 *
 * Landmark subset ("stable" landmarks). Everything that moves with expression or blinking is excluded: the eye
 * contours (blink, squint), the eyebrows (raise, frown) and the lips (open, smile, pucker) except the two outer lip
 * corners 61 and 291, which carry the mouth width. The jaw line and chin stay in (a closed mouth is assumed).
 * Eyebrow indices come from MediaPipe's face_mesh_connections (Apache-2.0); the other groups from
 * `FaceMapDef.regions`. The Python pipeline (`face_map.stable_landmarks`) uses the same rule.
 *
 * Head pose. A pitch of only 4 degrees (phone held low or high) already biases head-fat, cheek-volume and nose
 * height by 0.3 or more, and a yaw of 8 degrees biases the head width, so the out-of-plane head rotation is
 * estimated and removed. It comes from a 3-D similarity (Horn's method) between the photo landmarks, including the
 * MediaPipe depth z (same scale as x, smaller = closer), and the bound surface points; the model points are rotated
 * before the orthographic projection and the estimate is refreshed after the shape values changed (2 rounds). If the
 * photo carries no usable depth (all z equal), no rotation is applied and the photo is treated as frontal.
 */

/** MediaPipe FACEMESH_LEFT_EYEBROW / RIGHT_EYEBROW vertex indices (face_mesh_connections.py, Apache-2.0). */
export const MEDIAPIPE_EYEBROWS: readonly number[] = [
  276,
  283,
  282,
  295,
  285,
  300,
  293,
  334,
  296,
  336, // subject's left brow
  46,
  53,
  52,
  65,
  55,
  70,
  63,
  105,
  66,
  107, // subject's right brow
];
/** Outer lip corners kept although the lips are excluded (mouth width). */
export const LIP_CORNERS: readonly number[] = [61, 291];

/** Landmark indices used by the fit (ascending). */
export function stableLandmarkIndices(faceMap: FaceMapDef): number[] {
  const drop = new Set<number>([...faceMap.regions.leftEye, ...faceMap.regions.rightEye]);
  for (const i of MEDIAPIPE_EYEBROWS) drop.add(i);
  for (const i of faceMap.regions.lips) if (!LIP_CORNERS.includes(i)) drop.add(i);
  const out: number[] = [];
  for (let i = 0; i < 468; i++) if (!drop.has(i)) out.push(i);
  return out;
}

/** What the fit needs from the loaded assets (a `SolverData` fits). */
export type FaceFitData = Pick<SolverData, 'manifest' | 'base' | 'morphs'>;

export interface FitFaceOptions {
  /** Tikhonov weight pulling the values toward 0, in mm^2 per unit^2. Default 2. */
  regularization?: number;
  /** Max Levenberg-Marquardt iterations per round. Default 30. */
  maxIterations?: number;
  /**
   * 'auto' (default): estimate and remove the head pose when the photo landmarks carry depth; 'off': treat the
   * photo as frontal and ignore z.
   */
  pose?: 'auto' | 'off';
}

export interface FitFaceInput {
  data: FaceFitData;
  /** Merged target weights of the body the face is fitted onto (e.g. `SolveResult.weights`). */
  baseWeights: ReadonlyMap<string, number>;
  faceMap: FaceMapDef;
  /** MediaPipe normalized landmarks, x, y, z interleaved, at least 468 * 3 numbers (478 with irises is fine). */
  landmarks: ArrayLike<number>;
  imageWidth: number;
  imageHeight: number;
  options?: FitFaceOptions;
}

export interface FitFaceResult {
  /**
   * Values for every modifier in `faceMap.fitModifiers` (clamped to the manifest range). Apply them ON TOP of
   * `baseWeights`, e.g. `mergeWeights(baseWeights, modifierWeights(manifest, modifierValues))`.
   */
  modifierValues: Record<string, number>;
  /** RMS 2-D distance between photo and model landmarks after the similarity, in mm of the avatar head. */
  rmsResidual: number;
  /** Levenberg-Marquardt iterations performed (all rounds). */
  iterations: number;
  /**
   * Estimated head pose in degrees (null when the pose step was off or the photo has no depth). yaw: turn about the
   * vertical axis, positive = the nose points toward the image right (the subject's left); pitch: positive = the
   * nose points up; roll: in-plane rotation, positive = counter-clockwise as displayed. The caller can ask the user
   * to retake a photo whose |yaw| or |pitch| exceeds about 20 degrees.
   */
  pose: { yawDeg: number; pitchDeg: number; rollDeg: number } | null;
}

// ---------------------------------------------------------------------------------------------------------------
// Precomputed model (cached per face map + morph set)
// ---------------------------------------------------------------------------------------------------------------

interface Csr {
  start: Int32Array;
  slot: Int32Array;
  d: Float64Array;
}

interface FitModel {
  ids: string[];
  lo: Float64Array;
  hi: Float64Array;
  landmarks: Int32Array;
  /** Per used landmark: 3 slots and 3 weights. */
  lmSlots: Int32Array;
  lmBary: Float64Array;
  slotOf: Int32Array;
  slotVertex: Int32Array;
  inc: Csr;
  dec: Csr;
}

const modelCache = new WeakMap<FaceMapDef, WeakMap<MorphSet, FitModel>>();

function restrict(set: MorphSet, targets: number[], slotOf: Int32Array, slotCount: number): Csr {
  void slotCount;
  const slots: number[][] = [];
  const deltas: number[][] = [];
  for (const t of targets) {
    const s: number[] = [];
    const d: number[] = [];
    if (t >= 0) {
      const { words, floats } = set;
      let p = set.wordOffsets[t]!;
      const end = p + set.counts[t]! * 4;
      for (; p < end; p += 4) {
        const slot = slotOf[words[p]!]!;
        if (slot >= 0) {
          s.push(slot);
          d.push(floats[p + 1]!, floats[p + 2]!, floats[p + 3]!);
        }
      }
    }
    slots.push(s);
    deltas.push(d);
  }
  const start = new Int32Array(targets.length + 1);
  let total = 0;
  slots.forEach((s, i) => {
    start[i] = total;
    total += s.length;
  });
  start[targets.length] = total;
  const slot = new Int32Array(total);
  const d = new Float64Array(total * 3);
  slots.forEach((s, i) => {
    slot.set(s, start[i]!);
    d.set(deltas[i]!, start[i]! * 3);
  });
  return { start, slot, d };
}

function validateFaceMap(faceMap: FaceMapDef, manifest: BodyManifest): void {
  if (faceMap.version !== 1)
    throw new AvatarCoreError(`faceMap: unsupported version ${String(faceMap.version)}`);
  if (faceMap.landmarks.length !== 468)
    throw new AvatarCoreError(`faceMap: expected 468 landmarks, got ${faceMap.landmarks.length}`);
  faceMap.landmarks.forEach((l, i) => {
    if (l.index !== i) throw new AvatarCoreError(`faceMap: landmark ${i} has index ${l.index}`);
    for (const v of l.tri)
      if (!Number.isInteger(v) || v < 0 || v >= manifest.renderVertexCount)
        throw new AvatarCoreError(
          `faceMap: landmark ${i} references vertex ${v} outside the render vertices`,
        );
  });
}

function getModel(faceMap: FaceMapDef, manifest: BodyManifest, set: MorphSet): FitModel {
  let per = modelCache.get(faceMap);
  if (!per) {
    per = new WeakMap();
    modelCache.set(faceMap, per);
  }
  const cached = per.get(set);
  if (cached) return cached;

  validateFaceMap(faceMap, manifest);
  const cm = compileModifiers(manifest);
  const ids = [...faceMap.fitModifiers];
  const midx = ids.map((id) => {
    const i = cm.index.get(id);
    if (i === undefined)
      throw new AvatarCoreError(`faceMap: fit modifier "${id}" is not in the manifest`);
    return i;
  });
  const lo = Float64Array.from(midx.map((i) => cm.defs[i]!.min));
  const hi = Float64Array.from(midx.map((i) => cm.defs[i]!.max));

  const landmarks = Int32Array.from(stableLandmarkIndices(faceMap));
  const slotOf = new Int32Array(manifest.vertexCount).fill(-1);
  const slotVertex: number[] = [];
  const lmSlots = new Int32Array(landmarks.length * 3);
  const lmBary = new Float64Array(landmarks.length * 3);
  landmarks.forEach((li, n) => {
    const l = faceMap.landmarks[li]!;
    for (let k = 0; k < 3; k++) {
      const v = l.tri[k]!;
      let s = slotOf[v]!;
      if (s < 0) {
        s = slotVertex.length;
        slotVertex.push(v);
        slotOf[v] = s;
      }
      lmSlots[n * 3 + k] = s;
      lmBary[n * 3 + k] = l.bary[k]!;
    }
  });
  const inc = restrict(
    set,
    midx.map((i) => cm.incr[i]!),
    slotOf,
    slotVertex.length,
  );
  const dec = restrict(
    set,
    midx.map((i) => cm.decr[i]!),
    slotOf,
    slotVertex.length,
  );
  const model: FitModel = {
    ids,
    lo,
    hi,
    landmarks,
    lmSlots,
    lmBary,
    slotOf,
    slotVertex: Int32Array.from(slotVertex),
    inc,
    dec,
  };
  per.set(set, model);
  return model;
}

/** Body positions (base + baseWeights) at the slot vertices only. */
function baseAtSlots(
  data: FaceFitData,
  set: MorphSet,
  baseWeights: ReadonlyMap<string, number>,
  model: FitModel,
): Float64Array {
  const { slotVertex, slotOf } = model;
  const p0 = new Float64Array(slotVertex.length * 3);
  for (let s = 0; s < slotVertex.length; s++) {
    const o = slotVertex[s]! * 3;
    p0[s * 3] = data.base[o]!;
    p0[s * 3 + 1] = data.base[o + 1]!;
    p0[s * 3 + 2] = data.base[o + 2]!;
  }
  const { words, floats } = set;
  baseWeights.forEach((w, id) => {
    const t = set.targetIndex.get(id);
    if (t === undefined)
      throw new AvatarCoreError(`fitFace: unknown target "${id}" in baseWeights`);
    if (!Number.isFinite(w)) throw new AvatarCoreError(`fitFace: non-finite weight for "${id}"`);
    if (w === 0) return;
    let p = set.wordOffsets[t]!;
    const end = p + set.counts[t]! * 4;
    for (; p < end; p += 4) {
      const s = slotOf[words[p]!]!;
      if (s >= 0) {
        p0[s * 3] = p0[s * 3]! + w * floats[p + 1]!;
        p0[s * 3 + 1] = p0[s * 3 + 1]! + w * floats[p + 2]!;
        p0[s * 3 + 2] = p0[s * 3 + 2]! + w * floats[p + 3]!;
      }
    }
  });
  return p0;
}

// ---------------------------------------------------------------------------------------------------------------
// Residuals
// ---------------------------------------------------------------------------------------------------------------

/** Scratch buffers of one fit. */
interface Work {
  n: number;
  pcur: Float64Array;
  mx: Float64Array;
  my: Float64Array;
  qx: Float64Array;
  qy: Float64Array;
  /** Rows 0 and 1 of the head rotation applied before the orthographic projection (identity: frontal photo). */
  rot: Float64Array;
}

function assemble(
  model: FitModel,
  p0: Float64Array,
  theta: ArrayLike<number>,
  pcur: Float64Array,
): void {
  pcur.set(p0);
  for (let j = 0; j < model.ids.length; j++) {
    const v = theta[j]!;
    if (v === 0) continue;
    const csr = v > 0 ? model.inc : model.dec;
    const w = Math.abs(v);
    for (let k = csr.start[j]!, e = csr.start[j + 1]!; k < e; k++) {
      const o = csr.slot[k]! * 3;
      const q = k * 3;
      pcur[o] = pcur[o]! + w * csr.d[q]!;
      pcur[o + 1] = pcur[o + 1]! + w * csr.d[q + 1]!;
      pcur[o + 2] = pcur[o + 2]! + w * csr.d[q + 2]!;
    }
  }
}

/**
 * Fills `out` (2 n values, mm) with the residuals of the model at `theta` after the optimal 2-D similarity and
 * returns |k|, the pixels per meter of that similarity.
 */
function residuals(
  model: FitModel,
  p0: Float64Array,
  theta: ArrayLike<number>,
  w: Work,
  out: Float64Array,
): number {
  const { n, pcur, mx, my, qx, qy, rot } = w;
  assemble(model, p0, theta, pcur);
  const { lmSlots, lmBary } = model;
  let cmx = 0;
  let cmy = 0;
  let cqx = 0;
  let cqy = 0;
  for (let i = 0; i < n; i++) {
    const b = i * 3;
    const s0 = lmSlots[b]! * 3;
    const s1 = lmSlots[b + 1]! * 3;
    const s2 = lmSlots[b + 2]! * 3;
    const w0 = lmBary[b]!;
    const w1 = lmBary[b + 1]!;
    const w2 = lmBary[b + 2]!;
    const px = w0 * pcur[s0]! + w1 * pcur[s1]! + w2 * pcur[s2]!;
    const py = w0 * pcur[s0 + 1]! + w1 * pcur[s1 + 1]! + w2 * pcur[s2 + 1]!;
    const pz = w0 * pcur[s0 + 2]! + w1 * pcur[s1 + 2]! + w2 * pcur[s2 + 2]!;
    const x = rot[0]! * px + rot[1]! * py + rot[2]! * pz;
    const y = rot[3]! * px + rot[4]! * py + rot[5]! * pz;
    mx[i] = x;
    my[i] = y;
    cmx += x;
    cmy += y;
    cqx += qx[i]!;
    cqy += qy[i]!;
  }
  cmx /= n;
  cmy /= n;
  cqx /= n;
  cqy /= n;
  let den = 0;
  let re = 0;
  let im = 0;
  for (let i = 0; i < n; i++) {
    const ax = mx[i]! - cmx;
    const ay = my[i]! - cmy;
    const bx = qx[i]! - cqx;
    const by = qy[i]! - cqy;
    den += ax * ax + ay * ay;
    re += ax * bx + ay * by;
    im += ax * by - ay * bx;
  }
  const kr = re / den;
  const ki = im / den;
  const kAbs = Math.hypot(kr, ki);
  const toMm = 1000 / kAbs;
  for (let i = 0; i < n; i++) {
    const ax = mx[i]! - cmx;
    const ay = my[i]! - cmy;
    const bx = qx[i]! - cqx;
    const by = qy[i]! - cqy;
    out[2 * i] = (bx - (kr * ax - ki * ay)) * toMm;
    out[2 * i + 1] = (by - (kr * ay + ki * ax)) * toMm;
  }
  return kAbs;
}

function sumSquares(a: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * a[i]!;
  return s;
}

const FD_STEP = 1e-3;

// ---------------------------------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------------------------------

// --- head pose ----------------------------------------------------------------------------------------------------

/** Eigenvector of the largest eigenvalue of a symmetric 4 x 4 matrix (cyclic Jacobi). */
function largestEigenvector4(m: number[][]): number[] {
  const a = m.map((r) => r.slice());
  const v = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  for (let sweep = 0; sweep < 40; sweep++) {
    let off = 0;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 4; q++) off += a[p]![q]! * a[p]![q]!;
    if (off < 1e-24) break;
    for (let p = 0; p < 3; p++) {
      for (let q = p + 1; q < 4; q++) {
        const apq = a[p]![q]!;
        if (Math.abs(apq) < 1e-30) continue;
        const theta = (a[q]![q]! - a[p]![p]!) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 4; k++) {
          const akp = a[k]![p]!;
          const akq = a[k]![q]!;
          a[k]![p] = c * akp - s * akq;
          a[k]![q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k++) {
          const apk = a[p]![k]!;
          const aqk = a[q]![k]!;
          a[p]![k] = c * apk - s * aqk;
          a[q]![k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 4; k++) {
          const vkp = v[k]![p]!;
          const vkq = v[k]![q]!;
          v[k]![p] = c * vkp - s * vkq;
          v[k]![q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < 4; i++) if (a[i]![i]! > a[best]![best]!) best = i;
  return [v[0]![best]!, v[1]![best]!, v[2]![best]!, v[3]![best]!];
}

/**
 * Rotation (row-major 3 x 3 in `rot`) that best maps the model points (at `theta`) onto the photo points in the
 * least-squares sense (Horn's quaternion method; scale and translation drop out). `qz` holds the photo depth in
 * pixels (+ = toward the camera). Returns false when the photo has no usable depth.
 */
function estimatePose(
  model: FitModel,
  p0: Float64Array,
  theta: ArrayLike<number>,
  w: Work,
  qz: Float64Array,
  rot: Float64Array,
): boolean {
  const { n, pcur, qx, qy } = w;
  assemble(model, p0, theta, pcur);
  const { lmSlots, lmBary } = model;
  const mp = new Float64Array(n * 3);
  const c = [0, 0, 0, 0, 0, 0];
  for (let i = 0; i < n; i++) {
    const b = i * 3;
    for (let k = 0; k < 3; k++) {
      const v =
        lmBary[b]! * pcur[lmSlots[b]! * 3 + k]! +
        lmBary[b + 1]! * pcur[lmSlots[b + 1]! * 3 + k]! +
        lmBary[b + 2]! * pcur[lmSlots[b + 2]! * 3 + k]!;
      mp[b + k] = v;
      c[k] = c[k]! + v;
    }
    c[3] = c[3]! + qx[i]!;
    c[4] = c[4]! + qy[i]!;
    c[5] = c[5]! + qz[i]!;
  }
  for (let k = 0; k < 6; k++) c[k] = c[k]! / n;
  // depth spread of the photo landmarks (pixels): no depth -> no pose
  let zVar = 0;
  for (let i = 0; i < n; i++) zVar += (qz[i]! - c[5]!) ** 2;
  if (Math.sqrt(zVar / n) < 1e-3) return false;
  const S = new Float64Array(9); // S[a * 3 + b] = sum of M_a * Q_b (both centered)
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) {
      const ma = mp[i * 3 + a]! - c[a]!;
      S[a * 3] = S[a * 3]! + ma * (qx[i]! - c[3]!);
      S[a * 3 + 1] = S[a * 3 + 1]! + ma * (qy[i]! - c[4]!);
      S[a * 3 + 2] = S[a * 3 + 2]! + ma * (qz[i]! - c[5]!);
    }
  }
  const sxx = S[0]!;
  const sxy = S[1]!;
  const sxz = S[2]!;
  const syx = S[3]!;
  const syy = S[4]!;
  const syz = S[5]!;
  const szx = S[6]!;
  const szy = S[7]!;
  const szz = S[8]!;
  const N = [
    [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
    [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
    [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
    [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
  ];
  const q = largestEigenvector4(N);
  const norm = Math.hypot(q[0]!, q[1]!, q[2]!, q[3]!) || 1;
  const qw = q[0]! / norm;
  const qi = q[1]! / norm;
  const qj = q[2]! / norm;
  const qk = q[3]! / norm;
  rot.set([
    1 - 2 * (qj * qj + qk * qk),
    2 * (qi * qj - qw * qk),
    2 * (qi * qk + qw * qj),
    2 * (qi * qj + qw * qk),
    1 - 2 * (qi * qi + qk * qk),
    2 * (qj * qk - qw * qi),
    2 * (qi * qk - qw * qj),
    2 * (qj * qk + qw * qi),
    1 - 2 * (qi * qi + qj * qj),
  ]);
  return true;
}

/** Yaw, pitch, roll (degrees) of a model -> photo rotation; see FitFaceResult.pose for the sign conventions. */
function poseAngles(rot: Float64Array): { yawDeg: number; pitchDeg: number; rollDeg: number } {
  const deg = 180 / Math.PI;
  // the model's forward axis (0, 0, 1) becomes (r02, r12, r22); its right axis (1, 0, 0) becomes (r00, r10, r20)
  return {
    yawDeg: Math.atan2(rot[2]!, rot[8]!) * deg,
    pitchDeg: Math.asin(Math.max(-1, Math.min(1, rot[5]!))) * deg,
    rollDeg: Math.atan2(rot[3]!, rot[0]!) * deg,
  };
}

// --- bounded Levenberg-Marquardt ------------------------------------------------------------------------------------

/** LM on `theta` (in place) for the rotation currently in `w.rot`; returns the iterations used. */
function lmSolve(
  model: FitModel,
  p0: Float64Array,
  w: Work,
  theta: Float64Array,
  lambda: number,
  maxIter: number,
): number {
  const K = model.ids.length;
  const m = 2 * w.n;
  const r = new Float64Array(m);
  const rTry = new Float64Array(m);
  const rP = new Float64Array(m);
  const rM = new Float64Array(m);
  const thTry = new Float64Array(K);
  const thFd = new Float64Array(K);
  const jac = new Float64Array(m * K);
  const g = new Float64Array(K);
  const A = new Float64Array(K * K);
  const rhs = new Float64Array(K);
  const free: number[] = [];

  const costOf = (th: Float64Array, out: Float64Array): number => {
    residuals(model, p0, th, w, out);
    let reg = 0;
    for (let j = 0; j < K; j++) reg += th[j]! * th[j]!;
    return sumSquares(out) + lambda * reg;
  };

  let cost = costOf(theta, r);
  let mu = 1e-2;
  let iterations = 0;
  while (iterations < maxIter) {
    iterations++;
    // Jacobian: one-sided finite differences, the side follows the sign of theta_j (the morph model has a kink at 0);
    // at exactly 0 both sides are probed and the one that lowers the cost faster is used
    for (let j = 0; j < K; j++) {
      thFd.set(theta);
      let side = theta[j]! < 0 ? -1 : 1;
      thFd[j] = theta[j]! + side * FD_STEP;
      residuals(model, p0, thFd, w, rP);
      let col = rP;
      if (theta[j] === 0) {
        thFd[j] = -FD_STEP;
        residuals(model, p0, thFd, w, rM);
        let up = 0; // d cost / d theta along +theta
        let down = 0; // d cost / d |theta| along -theta
        for (let i = 0; i < m; i++) {
          up += r[i]! * ((rP[i]! - r[i]!) / FD_STEP);
          down += r[i]! * ((rM[i]! - r[i]!) / FD_STEP);
        }
        if (down < up && down < 0) {
          side = -1;
          col = rM;
        }
      }
      for (let i = 0; i < m; i++) jac[i * K + j] = (col[i]! - r[i]!) / (side * FD_STEP);
    }
    // gradient of 1/2 cost; variables pinned at a bound with the gradient pushing outward are frozen
    for (let j = 0; j < K; j++) {
      let s = 0;
      for (let i = 0; i < m; i++) s += jac[i * K + j]! * r[i]!;
      g[j] = s + lambda * theta[j]!;
    }
    free.length = 0;
    for (let j = 0; j < K; j++) {
      const atLo = theta[j]! <= model.lo[j]! + 1e-12 && g[j]! > 0;
      const atHi = theta[j]! >= model.hi[j]! - 1e-12 && g[j]! < 0;
      if (!atLo && !atHi) free.push(j);
    }
    if (free.length === 0) break;

    let accepted = false;
    let converged = false;
    for (let tries = 0; tries < 10; tries++) {
      const nf = free.length;
      for (let a = 0; a < nf; a++) {
        const ja = free[a]!;
        for (let b = 0; b < nf; b++) {
          const jb = free[b]!;
          let s = 0;
          for (let i = 0; i < m; i++) s += jac[i * K + ja]! * jac[i * K + jb]!;
          A[a * nf + b] = s + (a === b ? lambda : 0);
        }
        A[a * nf + a] = A[a * nf + a]! * (1 + mu) + 1e-9;
        rhs[a] = -g[ja]!;
      }
      const sol = rhs.subarray(0, nf);
      if (!solveLinear(A.subarray(0, nf * nf), sol, nf)) {
        mu *= 10;
        continue;
      }
      thTry.set(theta);
      for (let a = 0; a < nf; a++) {
        const ja = free[a]!;
        thTry[ja] = Math.min(model.hi[ja]!, Math.max(model.lo[ja]!, theta[ja]! + sol[a]!));
      }
      const c = costOf(thTry, rTry);
      if (c < cost) {
        let step = 0;
        for (let j = 0; j < K; j++) step = Math.max(step, Math.abs(thTry[j]! - theta[j]!));
        converged = step < 1e-6 || (cost - c) / Math.max(cost, 1e-12) < 1e-10;
        theta.set(thTry);
        r.set(rTry);
        cost = c;
        mu = Math.max(mu / 4, 1e-9);
        accepted = true;
        break;
      }
      mu *= 4;
    }
    if (!accepted || converged) break;
  }
  return iterations;
}

const POSE_ROUNDS = 2;

export function fitFaceModifiers(input: FitFaceInput): FitFaceResult {
  const { data, baseWeights, faceMap, landmarks, imageWidth, imageHeight } = input;
  if (!(imageWidth > 0) || !(imageHeight > 0))
    throw new AvatarCoreError(`fitFace: invalid image size ${imageWidth} x ${imageHeight}`);
  if (landmarks.length < 468 * 3)
    throw new AvatarCoreError(
      `fitFace: expected at least ${468 * 3} landmark values, got ${landmarks.length}`,
    );
  const lambda = input.options?.regularization ?? 2;
  const maxIter = input.options?.maxIterations ?? 30;
  const usePose = (input.options?.pose ?? 'auto') === 'auto';

  const set = getMorphSet(data.morphs, data.manifest);
  const model = getModel(faceMap, data.manifest, set);
  const K = model.ids.length;
  const n = model.landmarks.length;
  if (n < 20) throw new AvatarCoreError('fitFace: too few landmarks');
  const p0 = baseAtSlots(data, set, baseWeights, model);

  const work: Work = {
    n,
    pcur: new Float64Array(p0.length),
    mx: new Float64Array(n),
    my: new Float64Array(n),
    qx: new Float64Array(n),
    qy: new Float64Array(n),
    rot: Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]),
  };
  const qz = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const li = model.landmarks[i]!;
    const x = landmarks[li * 3]!;
    const y = landmarks[li * 3 + 1]!;
    const z = landmarks[li * 3 + 2]!;
    if (!Number.isFinite(x) || !Number.isFinite(y))
      throw new AvatarCoreError(`fitFace: landmark ${li} is not finite`);
    work.qx[i] = x * imageWidth;
    work.qy[i] = -(y * imageHeight);
    qz[i] = Number.isFinite(z) ? -(z * imageWidth) : 0; // MediaPipe: smaller z = closer; z has the scale of x
  }

  const theta = new Float64Array(K);
  const rot = new Float64Array(9);
  let iterations = 0;
  let pose: FitFaceResult['pose'] = null;
  if (usePose && estimatePose(model, p0, theta, work, qz, rot)) {
    work.rot.set(rot);
    for (let round = 0; round < POSE_ROUNDS; round++) {
      iterations += lmSolve(model, p0, work, theta, lambda, maxIter);
      if (!estimatePose(model, p0, theta, work, qz, rot)) break;
      let change = 0;
      for (let k = 0; k < 9; k++) change = Math.max(change, Math.abs(rot[k]! - work.rot[k]!));
      work.rot.set(rot);
      if (change < 2e-3) break;
    }
    iterations += lmSolve(model, p0, work, theta, lambda, maxIter);
    pose = poseAngles(work.rot);
  } else {
    iterations = lmSolve(model, p0, work, theta, lambda, maxIter);
  }

  // final residual (mm) without the regularization term
  const r = new Float64Array(2 * n);
  residuals(model, p0, theta, work, r);
  const modifierValues: Record<string, number> = {};
  model.ids.forEach((id, j) => {
    modifierValues[id] = theta[j]!;
  });
  return { modifierValues, rmsResidual: Math.sqrt(sumSquares(r) / n), iterations, pose };
}

/**
 * Frontal layout helper (tests, debug overlays): positions (x, y, z interleaved, meters, length 468 * 3) of every
 * bound landmark point on the body `baseWeights` plus the given face modifier values. Uses the full morph
 * application, independently of the fitter's restricted incremental model.
 */
export function faceLandmarkPositions(
  data: FaceFitData,
  baseWeights: ReadonlyMap<string, number>,
  faceMap: FaceMapDef,
  modifierValues: Readonly<Record<string, number>> = {},
): Float64Array {
  validateFaceMap(faceMap, data.manifest);
  const weights = mergeWeights(baseWeights, modifierWeights(data.manifest, modifierValues));
  const out = new Float32Array(data.base.length);
  applyMorphs(data.base, data.morphs, data.manifest, weights, out);
  const pts = new Float64Array(468 * 3);
  faceMap.landmarks.forEach((l, i) => {
    for (let c = 0; c < 3; c++) {
      pts[i * 3 + c] =
        l.bary[0] * out[l.tri[0] * 3 + c]! +
        l.bary[1] * out[l.tri[1] * 3 + c]! +
        l.bary[2] * out[l.tri[2] * 3 + c]!;
    }
  });
  return pts;
}

/** Convenience: the merged weights of a body plus fitted face modifier values. */
export function withFaceModifiers(
  manifest: BodyManifest,
  baseWeights: ReadonlyMap<string, number>,
  modifierValues: Readonly<Record<string, number>>,
): Map<string, number> {
  return mergeWeights(baseWeights, modifierWeights(manifest, modifierValues));
}
