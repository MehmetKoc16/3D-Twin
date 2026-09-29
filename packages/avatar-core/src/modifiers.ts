import type { BodyManifest, MacroVar, ModifierDef } from './contracts';
import { AvatarCoreError } from './errors';
import { macroWeights } from './macro';
import { validateManifest } from './morphs';

/** Manifest modifiers pre-resolved to target indices (index into manifest.targets, -1 = none). */
export interface CompiledModifiers {
  defs: ModifierDef[];
  index: ReadonlyMap<string, number>;
  decr: Int32Array;
  incr: Int32Array;
}

const cache = new WeakMap<BodyManifest, CompiledModifiers>();

export function compileModifiers(manifest: BodyManifest): CompiledModifiers {
  const cached = cache.get(manifest);
  if (cached) return cached;
  validateManifest(manifest);
  const targetIdx = new Map(manifest.targets.map((t, i) => [t.id, i]));
  const defs = manifest.modifiers;
  const decr = new Int32Array(defs.length).fill(-1);
  const incr = new Int32Array(defs.length).fill(-1);
  defs.forEach((d, i) => {
    if (d.decrTarget !== undefined) decr[i] = targetIdx.get(d.decrTarget)!;
    if (d.incrTarget !== undefined) incr[i] = targetIdx.get(d.incrTarget)!;
  });
  const compiled: CompiledModifiers = {
    defs,
    index: new Map(defs.map((d, i) => [d.id, i])),
    decr,
    incr,
  };
  cache.set(manifest, compiled);
  return compiled;
}

export function clampModifier(def: ModifierDef, value: number): number {
  return Math.min(def.max, Math.max(def.min, value));
}

/**
 * Adds one modifier's contribution to `out` (indexed by manifest target index):
 * value < 0 -> |value| on the decr target, value > 0 -> value on the incr target. Value must be pre-clamped.
 */
export function addModifierWeight(
  cm: CompiledModifiers,
  modifier: number,
  value: number,
  out: Float64Array,
): void {
  if (value > 0) {
    const t = cm.incr[modifier]!;
    if (t >= 0) out[t] = out[t]! + value;
  } else if (value < 0) {
    const t = cm.decr[modifier]!;
    if (t >= 0) out[t] = out[t]! - value;
  }
}

/**
 * Target weights of a set of modifier values (each clamped to the modifier's [min, max]). Modifiers missing
 * from `values` use their default. Throws on unknown ids and non-finite values. Zero weights are omitted.
 */
export function modifierWeights(
  manifest: BodyManifest,
  values: Readonly<Record<string, number>>,
): Map<string, number> {
  const cm = compileModifiers(manifest);
  for (const id of Object.keys(values)) {
    if (!cm.index.has(id)) throw new AvatarCoreError(`modifiers: unknown modifier "${id}"`);
    if (!Number.isFinite(values[id]))
      throw new AvatarCoreError(`modifiers: non-finite value for "${id}"`);
  }
  const w = new Float64Array(manifest.targets.length);
  cm.defs.forEach((d, i) => {
    const raw = values[d.id];
    addModifierWeight(cm, i, clampModifier(d, raw === undefined ? d.default : raw), w);
  });
  const out = new Map<string, number>();
  manifest.targets.forEach((t, i) => {
    if (w[i] !== 0) out.set(t.id, w[i]!);
  });
  return out;
}

/** Sums several weight maps (a target used by more than one source gets the sum). */
export function mergeWeights(...maps: ReadonlyMap<string, number>[]): Map<string, number> {
  const out = new Map<string, number>();
  for (const m of maps) m.forEach((w, id) => out.set(id, (out.get(id) ?? 0) + w));
  return out;
}

/** Macro weights + modifier weights merged into the single map consumed by applyMorphs. */
export function combineWeights(
  manifest: BodyManifest,
  macroVars: Partial<Record<MacroVar, number>>,
  modifierValues: Readonly<Record<string, number>> = {},
): Map<string, number> {
  return mergeWeights(macroWeights(manifest, macroVars), modifierWeights(manifest, modifierValues));
}
