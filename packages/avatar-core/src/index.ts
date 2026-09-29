export * from './contracts';
export * from './faceContracts';
export * from './types';
export { AvatarCoreError } from './errors';
export {
  MORPH_ENTRY_BYTES,
  validateManifest,
  buildBasePositions,
  parseMorphs,
  getMorphSet,
  applyMorphs,
  type MorphSet,
} from './morphs';
export { macroWeights } from './macro';
export { modifierWeights, mergeWeights, combineWeights } from './modifiers';
export {
  measure,
  estimateMassKg,
  meshVolumeM3,
  validateMeasure,
  bboxYRange,
  LoopWorkspace,
} from './measure';
export { solveBody, createBodySolver, BodySolver, bmiToWeightValue } from './solver';
export { computeJoints, validateRig, boneOrder, type JointPair } from './joints';
export { footLengthCmFromShoe, shoeFromFootLengthCm, convertShoeSize, roundShoeSize } from './shoe';
