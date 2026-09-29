import type { BodyManifest, BodyParams, MacroVar, MeasureDef, MeasureId } from './contracts';
import { AvatarCoreError } from './errors';
import { addMacroWeights, clampMacroValue, compileMacro, type CompiledMacro } from './macro';
import {
  bboxYRange,
  estimateMassKg,
  LoopWorkspace,
  measure,
  polylineLength,
  validateMeasure,
} from './measure';
import {
  addModifierWeight,
  clampModifier,
  compileModifiers,
  type CompiledModifiers,
} from './modifiers';
import {
  accumulateTarget,
  applyWeightArray,
  getMorphSet,
  validateManifest,
  type MorphSet,
} from './morphs';
import { solveLinear, solveScalar } from './numeric';
import { footLengthCmFromShoe } from './shoe';
import type { SolveOptions, SolveResult, SolverData } from './types';

const FD_STEP = 0.01;
const DEFAULT_BMI_KNOTS: ReadonlyArray<readonly [number, number]> = [
  [16, 0],
  [22, 0.5],
  [34, 1],
];
/** [measure id, BodyParams key]. `height` is handled by the macro pass, `footLength` comes from the shoe. */
const PARAM_KEYS: ReadonlyArray<readonly [MeasureId, keyof BodyParams]> = [
  ['shoulder', 'shoulderCm'],
  ['neck', 'neckCm'],
  ['chest', 'chestCm'],
  ['waist', 'waistCm'],
  ['hip', 'hipCm'],
  ['thigh', 'thighCm'],
  ['upperArm', 'upperArmCm'],
  ['armLength', 'armLengthCm'],
  ['inseam', 'inseamCm'],
];
const FLOOR_CANDIDATES = 32;
/** Residual (cm) above which a converged measure is treated as an outlier / unreachable and down-weighted. */
const OUTLIER_CM = 1;

/** Piecewise-linear BMI -> macro weight value, clamped to the end knots. */
export function bmiToWeightValue(
  bmi: number,
  knots: ReadonlyArray<readonly [number, number]> = DEFAULT_BMI_KNOTS,
): number {
  const first = knots[0]!;
  const last = knots[knots.length - 1]!;
  if (bmi <= first[0]) return first[1];
  if (bmi >= last[0]) return last[1];
  let i = 0;
  while (bmi > knots[i + 1]![0]) i++;
  const [b0, v0] = knots[i]!;
  const [b1, v1] = knots[i + 1]!;
  return v0 + ((v1 - v0) * (bmi - b0)) / (b1 - b0);
}

// ---------------------------------------------------------------------------------------------------------
// Local (measure-vertices-only) model
// ---------------------------------------------------------------------------------------------------------

const K_CIRC = 0;
const K_POLY = 1;
const K_DIST = 2;
const K_VHEIGHT = 3;

interface Csr {
  start: Int32Array;
  slot: Int32Array;
  d: Float64Array;
}

interface LocalTarget {
  id: MeasureId;
  def: MeasureDef;
  targetM: number;
  /** Indices into manifest.modifiers. */
  drivers: number[];
}

interface LocalModel {
  /** Solved variables = modifiers (index into manifest.modifiers). */
  n: number;
  mods: Int32Array;
  lo: Float64Array;
  hi: Float64Array;
  v0: Float64Array;
  m: number;
  kind: Uint8Array;
  axis: Int8Array;
  slotStart: Int32Array;
  slots: Int32Array;
  targetM: Float64Array;
  /** Per-measure residual weights (IRLS): 1 normally, < 1 for outliers so unreachable targets do not distort the rest. */
  omega: Float64Array;
  ws: (LoopWorkspace | null)[];
  floorSlots: Int32Array;
  vertexOfSlot: Int32Array;
  p0: Float64Array;
  pcur: Float64Array;
  inc: Csr;
  dec: Csr;
  /** Per variable: bitmask of measures whose vertices it can move. */
  depMask: Uint32Array;
}

function restrictTarget(
  set: MorphSet,
  t: number,
  slotOf: Int32Array,
): { slot: number[]; d: number[] } {
  const slot: number[] = [];
  const d: number[] = [];
  if (t < 0) return { slot, d };
  const { words, floats } = set;
  let p = set.wordOffsets[t]!;
  const end = p + set.counts[t]! * 4;
  for (; p < end; p += 4) {
    const s = slotOf[words[p]!]!;
    if (s >= 0) {
      slot.push(s);
      d.push(floats[p + 1]!, floats[p + 2]!, floats[p + 3]!);
    }
  }
  return { slot, d };
}

function toCsr(lists: { slot: number[]; d: number[] }[]): Csr {
  const start = new Int32Array(lists.length + 1);
  let total = 0;
  lists.forEach((l, i) => {
    start[i] = total;
    total += l.slot.length;
  });
  start[lists.length] = total;
  const slot = new Int32Array(total);
  const d = new Float64Array(total * 3);
  lists.forEach((l, i) => {
    slot.set(l.slot, start[i]!);
    d.set(l.d, start[i]! * 3);
  });
  return { start, slot, d };
}

function buildLocalModel(
  set: MorphSet,
  cmods: CompiledModifiers,
  targets: LocalTarget[],
  solved: number[],
  posNow: Float32Array,
  renderCount: number,
  vertexCount: number,
): LocalModel {
  const slotOf = new Int32Array(vertexCount).fill(-1);
  const vertexOfSlot: number[] = [];
  const getSlot = (v: number): number => {
    let s = slotOf[v]!;
    if (s < 0) {
      s = vertexOfSlot.length;
      vertexOfSlot.push(v);
      slotOf[v] = s;
    }
    return s;
  };

  const m = targets.length;
  const kind = new Uint8Array(m);
  const axis = new Int8Array(m).fill(-1);
  const slotStart = new Int32Array(m + 1);
  const slotList: number[] = [];
  const ws: (LoopWorkspace | null)[] = [];
  let vhMask = 0;
  targets.forEach((tg, i) => {
    slotStart[i] = slotList.length;
    const def = tg.def;
    ws.push(null);
    switch (def.type) {
      case 'circumference':
        kind[i] = K_CIRC;
        ws[i] = new LoopWorkspace(def.verts.length);
        for (const v of def.verts) slotList.push(getSlot(v));
        break;
      case 'polyline':
        kind[i] = K_POLY;
        for (const v of def.verts) slotList.push(getSlot(v));
        break;
      case 'distance':
        kind[i] = K_DIST;
        if (def.axis !== undefined) axis[i] = def.axis === 'x' ? 0 : def.axis === 'y' ? 1 : 2;
        slotList.push(getSlot(def.verts[0]), getSlot(def.verts[1]));
        break;
      case 'vertexHeight':
        kind[i] = K_VHEIGHT;
        vhMask |= 1 << i;
        slotList.push(getSlot(def.vert));
        break;
      case 'height':
        throw new AvatarCoreError('solver: height is solved by the macro pass, not locally');
    }
  });
  slotStart[m] = slotList.length;

  // floor candidates: the lowest render vertices at the current macro state (only needed for vertexHeight)
  const floorSlots: number[] = [];
  if (vhMask !== 0) {
    const best: { y: number; v: number }[] = [];
    for (let v = 0; v < renderCount; v++) {
      const y = posNow[v * 3 + 1]!;
      let k: number;
      if (best.length < FLOOR_CANDIDATES) {
        best.push({ y, v });
        k = best.length - 1;
      } else if (y < best[FLOOR_CANDIDATES - 1]!.y) {
        best[FLOOR_CANDIDATES - 1] = { y, v };
        k = FLOOR_CANDIDATES - 1;
      } else continue;
      while (k > 0 && best[k - 1]!.y > best[k]!.y) {
        const t = best[k - 1]!;
        best[k - 1] = best[k]!;
        best[k] = t;
        k--;
      }
    }
    for (const b of best) floorSlots.push(getSlot(b.v));
  }

  // per-slot dependency masks
  const slotMask = new Uint32Array(vertexOfSlot.length);
  for (let i = 0; i < m; i++)
    for (let k = slotStart[i]!; k < slotStart[i + 1]!; k++)
      slotMask[slotList[k]!] = slotMask[slotList[k]!]! | (1 << i);
  for (const s of floorSlots) slotMask[s] = slotMask[s]! | vhMask;

  // restricted target deltas per solved modifier
  const n = solved.length;
  const incLists: { slot: number[]; d: number[] }[] = [];
  const decLists: { slot: number[]; d: number[] }[] = [];
  const depMask = new Uint32Array(n);
  const lo = new Float64Array(n);
  const hi = new Float64Array(n);
  const v0 = new Float64Array(n);
  solved.forEach((mi, j) => {
    const def = cmods.defs[mi]!;
    lo[j] = def.min;
    hi[j] = def.max;
    v0[j] = clampModifier(def, def.default);
    const inc = restrictTarget(set, cmods.incr[mi]!, slotOf);
    const dec = restrictTarget(set, cmods.decr[mi]!, slotOf);
    incLists.push(inc);
    decLists.push(dec);
    let mask = 0;
    for (const s of inc.slot) mask |= slotMask[s]!;
    for (const s of dec.slot) mask |= slotMask[s]!;
    depMask[j] = mask;
  });

  const u = vertexOfSlot.length;
  return {
    n,
    mods: Int32Array.from(solved),
    lo,
    hi,
    v0,
    m,
    kind,
    axis,
    slotStart,
    slots: Int32Array.from(slotList),
    targetM: Float64Array.from(targets.map((t) => t.targetM)),
    omega: new Float64Array(m).fill(1),
    ws,
    floorSlots: Int32Array.from(floorSlots),
    vertexOfSlot: Int32Array.from(vertexOfSlot),
    p0: new Float64Array(u * 3),
    pcur: new Float64Array(u * 3),
    inc: toCsr(incLists),
    dec: toCsr(decLists),
    depMask,
  };
}

function loadP0(model: LocalModel, pos: Float32Array): void {
  const { vertexOfSlot, p0 } = model;
  for (let s = 0; s < vertexOfSlot.length; s++) {
    const o = vertexOfSlot[s]! * 3;
    p0[s * 3] = pos[o]!;
    p0[s * 3 + 1] = pos[o + 1]!;
    p0[s * 3 + 2] = pos[o + 2]!;
  }
}

/** pcur += sign * |value| * (delta of the side of modifier j selected by the sign of value). */
function addContribution(model: LocalModel, j: number, value: number, sign: number): void {
  const csr = value > 0 ? model.inc : value < 0 ? model.dec : null;
  if (!csr) return;
  const w = sign * Math.abs(value);
  const { pcur } = model;
  for (let k = csr.start[j]!, e = csr.start[j + 1]!; k < e; k++) {
    const o = csr.slot[k]! * 3;
    const q = k * 3;
    pcur[o] = pcur[o]! + w * csr.d[q]!;
    pcur[o + 1] = pcur[o + 1]! + w * csr.d[q + 1]!;
    pcur[o + 2] = pcur[o + 2]! + w * csr.d[q + 2]!;
  }
}

function assemble(model: LocalModel, v: Float64Array): void {
  model.pcur.set(model.p0);
  for (let j = 0; j < model.n; j++) addContribution(model, j, v[j]!, 1);
}

function evalMeasure(model: LocalModel, i: number): number {
  const { pcur, slots } = model;
  const a = model.slotStart[i]!;
  const b = model.slotStart[i + 1]!;
  switch (model.kind[i]) {
    case K_CIRC:
      return model.ws[i]!.hullPerimeter(pcur, slots, a, b);
    case K_POLY:
      return polylineLength(pcur, slots, a, b, false);
    case K_DIST: {
      const s0 = slots[a]! * 3;
      const s1 = slots[a + 1]! * 3;
      const ax = model.axis[i]!;
      if (ax >= 0) return Math.abs(pcur[s0 + ax]! - pcur[s1 + ax]!);
      return Math.hypot(
        pcur[s0]! - pcur[s1]!,
        pcur[s0 + 1]! - pcur[s1 + 1]!,
        pcur[s0 + 2]! - pcur[s1 + 2]!,
      );
    }
    default: {
      let floor = Infinity;
      for (let k = 0; k < model.floorSlots.length; k++) {
        const y = pcur[model.floorSlots[k]! * 3 + 1]!;
        if (y < floor) floor = y;
      }
      return pcur[slots[a]! * 3 + 1]! - floor;
    }
  }
}

function evalAll(model: LocalModel, out: Float64Array): void {
  for (let i = 0; i < model.m; i++) out[i] = evalMeasure(model, i);
}

/** Unweighted residuals (measured - target, cm) at v. */
function localResidualsCm(model: LocalModel, v: Float64Array, out: Float64Array): void {
  assemble(model, v);
  evalAll(model, out);
  for (let i = 0; i < model.m; i++) out[i] = (out[i]! - model.targetM[i]!) * 100;
}

/**
 * Bounded Levenberg-Marquardt on the modifier values `v` (in place) minimizing
 *   1/2 sum_i (100 (m_i - t_i))^2 + 1/2 mu |v - v0|^2
 * (residuals scaled by model.omega) with a one-sided finite-difference Jacobian restricted to the measures each modifier can move. Because the
 * morph model is piecewise linear in the modifier value (kink at 0), the side used for the difference follows
 * the current sign of the value and, at exactly 0, the side whose derivative reduces the cost.
 * Returns the number of iterations.
 */
function lmSolve(model: LocalModel, v: Float64Array, maxIter: number, mu: number): number {
  const { n, m } = model;
  if (n === 0 || m === 0) return 0;
  const r = new Float64Array(m);
  const rNew = new Float64Array(m);
  const mval = new Float64Array(m);
  const mtmp = new Float64Array(m);
  const colP = new Float64Array(m);
  const colM = new Float64Array(m);
  const jac = new Float64Array(m * n);
  const g = new Float64Array(n);
  const vNew = new Float64Array(n);
  const free: number[] = [];
  const A = new Float64Array(n * n);
  const rhs = new Float64Array(n);

  const costAt = (vt: Float64Array, rOut: Float64Array): number => {
    assemble(model, vt);
    evalAll(model, mtmp);
    let c = 0;
    for (let i = 0; i < m; i++) {
      const ri = (mtmp[i]! - model.targetM[i]!) * 100 * model.omega[i]!;
      rOut[i] = ri;
      c += ri * ri;
    }
    let reg = 0;
    for (let j = 0; j < n; j++) {
      const dv = vt[j]! - model.v0[j]!;
      reg += dv * dv;
    }
    return 0.5 * c + 0.5 * mu * reg;
  };

  /** Measures at v with variable j replaced by nv, written into `out` (only dependent measures recomputed). */
  const perturbed = (j: number, nv: number, out: Float64Array): void => {
    const vj = v[j]!;
    addContribution(model, j, vj, -1);
    addContribution(model, j, nv, 1);
    const mask = model.depMask[j]!;
    for (let i = 0; i < m; i++) out[i] = mask & (1 << i) ? evalMeasure(model, i) : mval[i]!;
    addContribution(model, j, nv, -1);
    addContribution(model, j, vj, 1);
  };
  const column = (j: number, nv: number, out: Float64Array): void => {
    perturbed(j, nv, out);
    const dv = nv - v[j]!;
    for (let i = 0; i < m; i++) out[i] = (((out[i]! - mval[i]!) * 100) / dv) * model.omega[i]!;
  };

  let cost = costAt(v, r);
  let lambda = 1e-4; // Levenberg (identity) damping: all variables share the [-1, 1] scale, and this keeps the ridge uniform
  let iters = 0;
  while (iters < maxIter) {
    let maxRes = 0;
    for (let i = 0; i < m; i++) maxRes = Math.max(maxRes, Math.abs(r[i]!));
    if (maxRes < 5e-4) break;
    iters++;

    // Jacobian (cm per unit) at v
    assemble(model, v);
    evalAll(model, mval);
    for (let j = 0; j < n; j++) {
      const vj = v[j]!;
      const lo = model.lo[j]!;
      const hi = model.hi[j]!;
      if (Math.abs(vj) >= 1e-9) {
        let nv: number;
        if (vj > 0) nv = vj + FD_STEP <= hi ? vj + FD_STEP : vj - Math.min(FD_STEP, vj);
        else nv = vj - FD_STEP >= lo ? vj - FD_STEP : vj + Math.min(FD_STEP, -vj);
        column(j, nv, colP);
        for (let i = 0; i < m; i++) jac[i * n + j] = colP[i]!;
      } else {
        const plusOk = hi > 0;
        const minusOk = lo < 0;
        let gP = 0;
        let gM = 0;
        if (plusOk) {
          column(j, Math.min(FD_STEP, hi), colP);
          for (let i = 0; i < m; i++) gP += colP[i]! * r[i]!;
        }
        if (minusOk) {
          column(j, -Math.min(FD_STEP, -lo), colM);
          for (let i = 0; i < m; i++) gM += colM[i]! * r[i]!;
        }
        let usePlus: boolean;
        if (plusOk && minusOk) usePlus = (gP < 0 ? -gP : 0) >= (gM > 0 ? gM : 0);
        else usePlus = plusOk;
        const src = usePlus ? colP : colM;
        for (let i = 0; i < m; i++) jac[i * n + j] = plusOk || minusOk ? src[i]! : 0;
      }
    }
    // gradient of the regularized cost
    for (let j = 0; j < n; j++) {
      let s = mu * (v[j]! - model.v0[j]!);
      for (let i = 0; i < m; i++) s += jac[i * n + j]! * r[i]!;
      g[j] = s;
    }
    // active set: variables at a bound whose gradient pushes outward stay fixed
    free.length = 0;
    for (let j = 0; j < n; j++) {
      const atLo = v[j]! <= model.lo[j]! + 1e-12 && g[j]! > 0;
      const atHi = v[j]! >= model.hi[j]! - 1e-12 && g[j]! < 0;
      if (!atLo && !atHi) free.push(j);
    }
    const nf = free.length;
    if (nf === 0) break;

    let accepted = false;
    let costNew = cost;
    for (let attempt = 0; attempt < 12 && !accepted; attempt++) {
      for (let a = 0; a < nf; a++) {
        const ja = free[a]!;
        for (let b = 0; b < nf; b++) {
          const jb = free[b]!;
          let s = 0;
          for (let i = 0; i < m; i++) s += jac[i * n + ja]! * jac[i * n + jb]!;
          A[a * nf + b] = s + (a === b ? mu : 0);
        }
        const d = A[a * nf + a]!;
        A[a * nf + a] = d + lambda;
        rhs[a] = -g[ja]!;
      }
      if (!solveLinear(A, rhs, nf)) {
        lambda *= 10;
        continue;
      }
      vNew.set(v);
      for (let a = 0; a < nf; a++) {
        const ja = free[a]!;
        vNew[ja] = Math.min(model.hi[ja]!, Math.max(model.lo[ja]!, v[ja]! + rhs[a]!));
      }
      costNew = costAt(vNew, rNew);
      if (costNew < cost) {
        accepted = true;
        lambda = Math.max(lambda / 4, 1e-10);
      } else {
        lambda *= 6;
      }
    }
    if (!accepted) break;
    const decrease = cost - costNew;
    v.set(vNew);
    r.set(rNew);
    cost = costNew;
    if (decrease <= 1e-12 * Math.max(1, cost)) break;
  }
  return iters;
}

// ---------------------------------------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------------------------------------

function assertPositive(x: unknown, what: string): number {
  if (typeof x !== 'number' || !Number.isFinite(x) || x <= 0)
    throw new AvatarCoreError(
      `solveBody: ${what} must be a positive finite number (got ${String(x)})`,
    );
  return x;
}

/**
 * Prepared solver for one dataset: parses/validates morphs and measures once and keeps scratch buffers, so
 * repeated solves (slider drags) are cheap. Not re-entrant.
 */
export class BodySolver {
  private readonly manifest: BodyManifest;
  private readonly set: MorphSet;
  private readonly macro: CompiledMacro;
  private readonly cmods: CompiledModifiers;
  private readonly base: Float32Array;
  private readonly measures: MeasureDef[];
  private readonly indices: ArrayLike<number> | undefined;
  private readonly renderCount: number;
  private readonly W: Float64Array;
  private readonly posP0: Float32Array;
  private readonly pos: Float32Array;

  constructor(data: SolverData) {
    const m = data.manifest;
    validateManifest(m);
    this.manifest = m;
    this.set = getMorphSet(data.morphs, m);
    this.macro = compileMacro(m);
    this.cmods = compileModifiers(m);
    if (data.base.length !== m.vertexCount * 3)
      throw new AvatarCoreError(
        `solveBody: base length ${data.base.length} != 3 * vertexCount ${m.vertexCount}`,
      );
    this.base = data.base;
    this.indices = data.indices;
    this.renderCount = m.renderVertexCount;
    this.measures = data.measures.measures;
    for (const d of this.measures) validateMeasure(d, m.vertexCount);
    if (data.indices) {
      if (data.indices.length % 3 !== 0)
        throw new AvatarCoreError('solveBody: index count is not a multiple of 3');
      for (let i = 0; i < data.indices.length; i++) {
        const v = data.indices[i]!;
        if (!(v >= 0 && v < this.renderCount))
          throw new AvatarCoreError(`solveBody: index ${v} outside the render vertices`);
      }
    }
    this.W = new Float64Array(m.targets.length);
    this.posP0 = new Float32Array(data.base.length);
    this.pos = new Float32Array(data.base.length);
  }

  solve(params: BodyParams, opts: SolveOptions = {}): SolveResult {
    const { manifest, set, macro: cm, cmods, base, W, posP0, pos, renderCount } = this;
    if (!Number.isFinite(params.gender))
      throw new AvatarCoreError('solveBody: gender must be a finite number');
    const heightCm = assertPositive(params.heightCm, 'heightCm');
    const weightKg = assertPositive(params.weightKg, 'weightKg');
    const tolCm = opts.toleranceCm ?? 0.2;
    const maxIter = opts.maxIterations ?? 30;
    const mu = opts.regularization ?? 1e-3;
    const density = opts.densityKgPerL ?? 1.01;
    const canMass = (opts.matchMass ?? true) && this.indices !== undefined;
    const targetH = heightCm / 100;

    // ---- state: macro variable values + modifier values
    const macroVals = new Float64Array(cm.vars.length);
    cm.vars.forEach((v, i) => {
      const o = opts.macroOverrides?.[v.id];
      macroVals[i] = clampMacroValue(v, o === undefined ? v.default : o);
    });
    const hIdx = cm.varIndex.get('height') ?? -1;
    const wIdx = cm.varIndex.get('weight') ?? -1;
    const gIdx = cm.varIndex.get('gender') ?? -1;
    if (gIdx >= 0) macroVals[gIdx] = clampMacroValue(cm.vars[gIdx]!, params.gender);
    if (wIdx >= 0) {
      const bmi = weightKg / (targetH * targetH);
      macroVals[wIdx] = clampMacroValue(
        cm.vars[wIdx]!,
        bmiToWeightValue(bmi, opts.bmiKnots ?? DEFAULT_BMI_KNOTS),
      );
    }
    const modVals = new Float64Array(cmods.defs.length);
    cmods.defs.forEach((d, i) => {
      modVals[i] = clampModifier(d, d.default);
    });

    // ---- local targets
    const defById = new Map<MeasureId, MeasureDef>();
    for (const d of this.measures) defById.set(d.id, d);
    const targetsCm = new Map<MeasureId, number>([['height', heightCm]]);
    const unreachable = new Set<MeasureId>();
    const local: LocalTarget[] = [];
    const wanted: [MeasureId, number][] = [];
    for (const [id, key] of PARAM_KEYS) {
      const val = params[key] as number | undefined;
      if (val !== undefined) wanted.push([id, assertPositive(val, key)]);
    }
    if (defById.has('footLength') && params.shoe)
      wanted.push(['footLength', footLengthCmFromShoe(params.shoe)]);
    for (const [id, cmVal] of wanted) {
      targetsCm.set(id, cmVal);
      const def = defById.get(id);
      const drivers = def
        ? [
            ...new Set(def.drivers.map((d) => cmods.index.get(d)).filter((x) => x !== undefined)),
          ].filter((i) => cmods.incr[i]! >= 0 || cmods.decr[i]! >= 0)
        : [];
      if (!def || def.type === 'height' || drivers.length === 0) unreachable.add(id);
      else local.push({ id, def, targetM: cmVal / 100, drivers });
    }
    const solved: number[] = [...new Set(local.flatMap((t) => t.drivers))].sort((a, b) => a - b);
    const solvedMask = new Uint8Array(cmods.defs.length);
    for (const j of solved) solvedMask[j] = 1;

    // ---- evaluation helpers (positions = base + macros + fixed modifiers [+ solved modifiers])
    const buildWeights = (includeSolved: boolean): void => {
      W.fill(0);
      addMacroWeights(cm, macroVals, W);
      for (let j = 0; j < modVals.length; j++)
        if (includeSolved || !solvedMask[j]) addModifierWeight(cmods, j, modVals[j]!, W);
    };
    const evalP0 = (): void => {
      buildWeights(false);
      applyWeightArray(set, base, W, posP0);
    };
    const evalFull = (): void => {
      evalP0();
      pos.set(posP0);
      for (const j of solved) {
        const v = modVals[j]!;
        if (v > 0) accumulateTarget(set, cmods.incr[j]!, v, pos);
        else if (v < 0) accumulateTarget(set, cmods.decr[j]!, -v, pos);
      }
    };
    const heightOf = (): number => {
      const r = bboxYRange(pos, renderCount);
      return r.max - r.min;
    };
    const massOf = (): number => estimateMassKg(pos, this.indices!, density, renderCount);

    let iterations = 0;

    // ---- (a) height by macro variable; (b) weight (BMI seed, refined by mesh mass)
    const solveHeightPlain = (): void => {
      if (hIdx < 0) return;
      const v = cm.vars[hIdx]!;
      const sol = solveScalar(
        (x) => {
          macroVals[hIdx] = x;
          evalFull();
          return heightOf() - targetH;
        },
        macroVals[hIdx]!,
        v.min,
        v.max,
        2e-5,
        40,
      );
      macroVals[hIdx] = sol.x;
    };
    solveHeightPlain();
    if (canMass && wIdx >= 0) {
      const v = cm.vars[wIdx]!;
      for (let pass = 0; pass < 3; pass++) {
        const sol = solveScalar(
          (x) => {
            macroVals[wIdx] = x;
            evalFull();
            return massOf() - weightKg;
          },
          macroVals[wIdx]!,
          v.min,
          v.max,
          0.02,
          40,
        );
        macroVals[wIdx] = sol.x;
        solveHeightPlain();
        evalFull();
        if (Math.abs(massOf() - weightKg) <= 0.05) break;
      }
    }

    // ---- (c) local measures: LM on the driver modifiers, nested in a height correction on the macro variable
    if (local.length > 0) {
      evalFull();
      const model = buildLocalModel(
        set,
        cmods,
        local,
        solved,
        pos,
        renderCount,
        manifest.vertexCount,
      );
      const v = new Float64Array(model.n);
      const rr = new Float64Array(model.m);
      const localFit = (): void => {
        for (let j = 0; j < model.n; j++) v[j] = modVals[model.mods[j]!]!;
        model.omega.fill(1);
        iterations += lmSolve(model, v, maxIter, mu);
        // Robustness: a target that is far off after convergence is unreachable; down-weight it (IRLS) and refit
        // so it cannot drag the reachable measures away from their targets.
        // Only the worst offender is down-weighted per round: a far-off target poisons the residuals of the
        // measures coupled to it, and those must not be demoted with it.
        for (let round = 0; round < Math.min(model.m, 4); round++) {
          localResidualsCm(model, v, rr);
          let worst = -1;
          let worstAbs = OUTLIER_CM;
          for (let i = 0; i < model.m; i++) {
            const a = Math.abs(rr[i]!);
            if (a > worstAbs && model.omega[i] === 1) {
              worst = i;
              worstAbs = a;
            }
          }
          if (worst < 0) break;
          model.omega[worst] = (0.3 * OUTLIER_CM) / worstAbs;
          iterations += lmSolve(model, v, maxIter, mu);
        }
        for (let j = 0; j < model.n; j++) modVals[model.mods[j]!] = v[j]!;
      };
      let lastX = NaN;
      const heightAfterLocal = (x: number): number => {
        if (hIdx >= 0) macroVals[hIdx] = x;
        lastX = x;
        evalP0();
        loadP0(model, posP0);
        localFit();
        evalFull();
        return heightOf() - targetH;
      };
      if (hIdx >= 0) {
        const hv = cm.vars[hIdx]!;
        const sol = solveScalar(heightAfterLocal, macroVals[hIdx]!, hv.min, hv.max, 1e-4, 16);
        if (lastX !== sol.x) heightAfterLocal(sol.x);
        macroVals[hIdx] = sol.x;
      } else {
        heightAfterLocal(0);
      }
    }

    // ---- final state
    evalFull();
    buildWeights(true);
    const weights = new Map<string, number>();
    manifest.targets.forEach((t, i) => {
      if (W[i] !== 0) weights.set(t.id, W[i]!);
    });
    const positions = pos.slice();

    const achievedCm: Partial<Record<MeasureId, number>> = {};
    for (const d of this.measures) achievedCm[d.id] = measure(d, positions, renderCount) * 100;
    if (achievedCm.height === undefined) {
      const r = bboxYRange(positions, renderCount);
      achievedCm.height = (r.max - r.min) * 100;
    }
    const residualsCm: Partial<Record<MeasureId, number>> = {};
    targetsCm.forEach((target, id) => {
      const a = achievedCm[id];
      if (a === undefined) return;
      const res = a - target;
      residualsCm[id] = res;
      if (Math.abs(res) > tolCm) unreachable.add(id);
    });
    if (hIdx < 0) unreachable.add('height');

    const modifierValues: Record<string, number> = {};
    cmods.defs.forEach((d, i) => {
      if (modVals[i] !== 0) modifierValues[d.id] = modVals[i]!;
    });
    const macroOut: Partial<Record<MacroVar, number>> = {};
    cm.vars.forEach((vr, i) => {
      macroOut[vr.id] = macroVals[i]!;
    });

    return {
      weights,
      modifierValues,
      macroVars: macroOut,
      positions,
      achievedCm,
      residualsCm,
      unreachable: [...unreachable],
      estimatedMassKg: this.indices
        ? estimateMassKg(positions, this.indices, density, renderCount)
        : NaN,
      iterations,
    };
  }
}

const solverCache = new WeakMap<BodyManifest, { data: SolverData; solver: BodySolver }[]>();

/** Returns a prepared solver for the dataset (cached per manifest + morphs + base + measures + indices). */
export function createBodySolver(data: SolverData): BodySolver {
  let entries = solverCache.get(data.manifest);
  if (!entries) {
    entries = [];
    solverCache.set(data.manifest, entries);
  }
  const hit = entries.find(
    (e) =>
      e.data.morphs === data.morphs &&
      e.data.base === data.base &&
      e.data.measures === data.measures &&
      e.data.indices === data.indices,
  );
  if (hit) return hit.solver;
  const solver = new BodySolver(data);
  entries.push({ data, solver });
  return solver;
}

/**
 * Solves body parameters into morph weights and positions.
 * 1. Height: 1-D root find on the macro `height` variable.
 * 2. Weight: macro `weight` seeded from BMI (bmiToWeightValue), refined so the mesh mass matches weightKg.
 * 3. Provided local measures: bounded LM on the measures' driver modifiers (finite-difference Jacobian on the
 *    measure vertices only), nested inside a height root find so height stays exact when the modifiers or
 *    the macros couple circumference and height. Unreachable targets are reported, never thrown.
 */
export function solveBody(data: SolverData, params: BodyParams, opts?: SolveOptions): SolveResult {
  return createBodySolver(data).solve(params, opts);
}
