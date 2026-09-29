import type { BodyManifest, MacroVar, MeasureId, MeasuresDef } from './contracts';
import type { MorphSet } from './morphs';

/** Everything the solver needs, loaded from the asset files. */
export interface SolverData {
  manifest: BodyManifest;
  /** Full base positions, length 3 * manifest.vertexCount (render vertices + joint points; see buildBasePositions). */
  base: Float32Array;
  /** Raw morphs.bin or a pre-parsed MorphSet. */
  morphs: ArrayBufferLike | MorphSet;
  measures: MeasuresDef;
  /**
   * Triangle indices (3 per triangle) into the RENDER vertices, used only for the mass estimate. Without them
   * `estimatedMassKg` is NaN and the weight is set from BMI alone.
   */
  indices?: ArrayLike<number>;
}

export interface SolveOptions {
  /** Max Levenberg-Marquardt iterations per local solve. Default 30. */
  maxIterations?: number;
  /** A measure whose |achieved - target| exceeds this (cm) is reported as unreachable. Default 0.2. */
  toleranceCm?: number;
  /** Refine the macro weight variable so the mesh mass matches weightKg (needs `indices`). Default true. */
  matchMass?: boolean;
  /** Tikhonov weight pulling modifiers toward their default (cm^2 per unit^2). Default 1e-3. */
  regularization?: number;
  /** Macro variables other than gender/height/weight (age, muscle, ...). Missing = manifest default. */
  macroOverrides?: Partial<Record<MacroVar, number>>;
  /** BMI -> macro `weight` value knots (ascending BMI). Default [[16, 0], [22, 0.5], [34, 1]]. */
  bmiKnots?: ReadonlyArray<readonly [number, number]>;
  /** Tissue density for the mass estimate in kg/L. Default 1.01. */
  densityKgPerL?: number;
}

export interface SolveResult {
  /** Merged macro + modifier target weights; feed to applyMorphs. */
  weights: Map<string, number>;
  /** Final modifier values (only non-zero ones). */
  modifierValues: Record<string, number>;
  /** Final macro variable values. */
  macroVars: Partial<Record<MacroVar, number>>;
  /** base + weights applied (length 3 * vertexCount). */
  positions: Float32Array;
  /** Measured values in cm for every measure defined in measures.json (plus height). */
  achievedCm: Partial<Record<MeasureId, number>>;
  /** achieved - target in cm, only for measures that had a target. */
  residualsCm: Partial<Record<MeasureId, number>>;
  /** Targeted measures that could not be met within tolerance (or have no definition / no driver). */
  unreachable: MeasureId[];
  /** Mesh volume x density; NaN when no indices were supplied. */
  estimatedMassKg: number;
  /** Total Levenberg-Marquardt iterations over all local solves. */
  iterations: number;
}
