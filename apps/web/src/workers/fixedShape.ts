import {
  applyMorphs,
  combineWeights,
  estimateMassKg,
  measure,
  type BodyManifest,
  type MacroVar,
  type MeasureId,
  type MeasuresDef,
  type MorphSet,
} from '@dt/avatar-core';

/**
 * A body given directly by macro variables and modifier values (the realistic twin's fitted MakeHuman body) instead of
 * being solved from measurements. Pure: shared by the avatar worker and the tests.
 */
export interface FixedShape {
  macros: Partial<Record<MacroVar, number>>;
  /** Net modifier values by modifier id (each is clamped to the modifier's range by avatar-core). */
  modifiers: Record<string, number>;
}

export interface FixedShapeData {
  manifest: BodyManifest;
  base: Float32Array;
  morphs: ArrayBufferLike | MorphSet;
  measures: MeasuresDef;
  indices: ArrayLike<number>;
}

export interface FixedShapeResult {
  /** Render vertices ++ joint points, not grounded. */
  positions: Float32Array;
  achievedCm: Partial<Record<MeasureId, number>>;
  estimatedMassKg: number;
}

/** Target weights of a fixed shape. Throws (AvatarCoreError) on unknown modifier ids or non-finite values. */
export function fixedShapeWeights(manifest: BodyManifest, shape: FixedShape): Map<string, number> {
  return combineWeights(manifest, shape.macros, shape.modifiers);
}

export function solveFixedShape(data: FixedShapeData, shape: FixedShape): FixedShapeResult {
  const { manifest, base, morphs, measures, indices } = data;
  const weights = fixedShapeWeights(manifest, shape);
  const positions = new Float32Array(base.length);
  applyMorphs(base, morphs, manifest, weights, positions);
  const renderCount = manifest.renderVertexCount;
  const achievedCm: Partial<Record<MeasureId, number>> = {};
  for (const def of measures.measures)
    achievedCm[def.id] = measure(def, positions, renderCount) * 100;
  return {
    positions,
    achievedCm,
    estimatedMassKg: estimateMassKg(positions, indices, 1.01, renderCount),
  };
}
