import type { BodyManifest, MacroVar } from './contracts';
import { AvatarCoreError } from './errors';
import { validateManifest } from './morphs';

export interface CompiledMacroVar {
  id: MacroVar;
  min: number;
  max: number;
  default: number;
  /** Bucket positions, ascending. */
  ats: Float64Array;
  /** Bucket names in the same (ascending) order as `ats`. */
  names: string[];
}

export interface CompiledMacroTarget {
  id: string;
  /** Index into manifest.targets. */
  targetIndex: number;
  /** Per condition: index into CompiledMacro.vars and index into that variable's sorted buckets. */
  varIdx: Int32Array;
  bucketIdx: Int32Array;
}

/** Manifest macro data pre-resolved to indices for allocation-free evaluation. */
export interface CompiledMacro {
  vars: CompiledMacroVar[];
  varIndex: ReadonlyMap<string, number>;
  targets: CompiledMacroTarget[];
  /** Scratch: per-variable bucket weights. */
  bucketScratch: Float64Array[];
}

const compileCache = new WeakMap<BodyManifest, CompiledMacro>();

export function compileMacro(manifest: BodyManifest): CompiledMacro {
  const cached = compileCache.get(manifest);
  if (cached) return cached;
  validateManifest(manifest);

  const vars: CompiledMacroVar[] = manifest.macroVariables.map((v) => {
    const sorted = [...v.buckets].sort((a, b) => a.at - b.at);
    return {
      id: v.id,
      min: v.min,
      max: v.max,
      default: v.default,
      ats: Float64Array.from(sorted.map((b) => b.at)),
      names: sorted.map((b) => b.name),
    };
  });
  const varIndex = new Map(vars.map((v, i) => [v.id as string, i]));

  const targets: CompiledMacroTarget[] = [];
  manifest.targets.forEach((t, targetIndex) => {
    if (!t.macroConditions || t.macroConditions.length === 0) return;
    const varIdx = new Int32Array(t.macroConditions.length);
    const bucketIdx = new Int32Array(t.macroConditions.length);
    t.macroConditions.forEach((c, i) => {
      const vi = varIndex.get(c.variable);
      if (vi === undefined) throw new AvatarCoreError(`macro: unknown variable "${c.variable}"`);
      const bi = vars[vi]!.names.indexOf(c.bucket);
      if (bi < 0)
        throw new AvatarCoreError(`macro: unknown bucket "${c.bucket}" of "${c.variable}"`);
      varIdx[i] = vi;
      bucketIdx[i] = bi;
    });
    targets.push({ id: t.id, targetIndex, varIdx, bucketIdx });
  });

  const compiled: CompiledMacro = {
    vars,
    varIndex,
    targets,
    bucketScratch: vars.map((v) => new Float64Array(v.ats.length)),
  };
  compileCache.set(manifest, compiled);
  return compiled;
}

/**
 * Tent (piecewise-linear) weights of the buckets of one variable at `value` (already clamped by the caller).
 * Weights sum to 1; below the first / above the last bucket the end bucket has weight 1.
 */
export function bucketWeightsInto(v: CompiledMacroVar, value: number, out: Float64Array): void {
  const n = v.ats.length;
  out.fill(0, 0, n);
  const ats = v.ats;
  if (n === 1 || value <= ats[0]!) {
    out[0] = 1;
  } else if (value >= ats[n - 1]!) {
    out[n - 1] = 1;
  } else {
    let i = 0;
    while (value >= ats[i + 1]!) i++;
    const t = (value - ats[i]!) / (ats[i + 1]! - ats[i]!);
    out[i] = 1 - t;
    out[i + 1] = t;
  }
}

/** Clamps a variable value into its [min, max]; NaN falls back to the default. */
export function clampMacroValue(v: CompiledMacroVar, value: number): number {
  if (Number.isNaN(value)) return v.default;
  return Math.min(v.max, Math.max(v.min, value));
}

/**
 * Adds the macro target weights (product of bucket tent weights over the macroConditions) into `out`, indexed
 * by manifest target index. `values[i]` is the (already clamped) value of `cm.vars[i]`. Does not zero `out`.
 */
export function addMacroWeights(
  cm: CompiledMacro,
  values: ArrayLike<number>,
  out: Float64Array,
): void {
  for (let i = 0; i < cm.vars.length; i++)
    bucketWeightsInto(cm.vars[i]!, values[i]!, cm.bucketScratch[i]!);
  for (const t of cm.targets) {
    let w = 1;
    for (let c = 0; c < t.varIdx.length && w !== 0; c++)
      w *= cm.bucketScratch[t.varIdx[c]!]![t.bucketIdx[c]!]!;
    if (w !== 0) out[t.targetIndex] = out[t.targetIndex]! + w;
  }
}

/** Resolves partial variable input into a value per compiled variable (defaults for missing, clamped). */
export function macroValues(
  cm: CompiledMacro,
  vars: Partial<Record<MacroVar, number>>,
): Float64Array {
  const out = new Float64Array(cm.vars.length);
  cm.vars.forEach((v, i) => {
    const x = vars[v.id];
    out[i] = clampMacroValue(v, x === undefined ? v.default : x);
  });
  return out;
}

/**
 * Macro target weights: for every target with macroConditions, the product of the tent weights of the buckets
 * it names. Inputs are clamped to each variable's range; missing variables use their default. Targets with
 * zero weight are omitted from the map.
 */
export function macroWeights(
  manifest: BodyManifest,
  vars: Partial<Record<MacroVar, number>>,
): Map<string, number> {
  const cm = compileMacro(manifest);
  const values = macroValues(cm, vars);
  const w = new Float64Array(manifest.targets.length);
  addMacroWeights(cm, values, w);
  const out = new Map<string, number>();
  for (const t of cm.targets) {
    const x = w[t.targetIndex]!;
    if (x !== 0) out.set(t.id, x);
  }
  return out;
}
