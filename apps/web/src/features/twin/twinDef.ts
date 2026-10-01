import type { MacroVar, MeasureId } from '@dt/avatar-core';

/**
 * `twin.json` written by `tools/twin-lab/rig/rig_scan.py` (twin_export.py) next to `rigged.glb`.
 * Everything in it describes the MakeHuman body that was fitted to the scan; the browser rebuilds that body with the
 * same avatar-core morph model. Parsing is strict: a malformed file is rejected before anything is solved or bound.
 */
export interface TwinDef {
  version: 1;
  /** The 53 bone names in rig.json order (= the glTF skin joint order of rigged.glb). */
  boneOrder: string[];
  fittedMacros: Partial<Record<MacroVar, number>>;
  /** Net modifier values (incr - decr) by modifier id. */
  fittedModifiers: Record<string, number>;
  /** Measurements of the person (fitted body minus the clothing allowance), cm. */
  measurementsCm: Partial<Record<MeasureId, number>>;
  /** Raw measurements of the fitted body and the allowance that was subtracted (informational). */
  measurementsRawCm?: Partial<Record<MeasureId, number>>;
  clothingAllowanceCm?: Partial<Record<MeasureId, number>>;
  /** Rest head of every bone in the frame of rigged.glb (metres); informational, the skeleton is read from the glb. */
  restHeadsM?: Record<string, [number, number, number]>;
  /** Twin vertex -> nearest MakeHuman render vertex file (optional; without it no twin triangle is hidden). */
  mapping?: { file: string; twinVertexCount: number; renderVertexCount: number };
  fit?: { twinHeightM?: number };
}

export type TwinErrorCode =
  | 'json'
  | 'version'
  | 'bones'
  | 'macros'
  | 'modifiers'
  | 'measurements'
  | 'mapping'
  | 'glb'
  | 'noSkin'
  | 'rest'
  | 'mismatch'
  | 'shape';

/** Validation / binding failure with a stable code; the UI shows `twin.errors.<code>`. */
export class TwinFormatError extends Error {
  constructor(
    readonly code: TwinErrorCode,
    detail: string,
  ) {
    super(`twin: ${detail}`);
    this.name = 'TwinFormatError';
  }
}

export const MACRO_IDS: readonly MacroVar[] = [
  'gender',
  'age',
  'muscle',
  'weight',
  'height',
  'proportions',
  'cupsize',
  'firmness',
];

/** Measure ids in the order the body panel lists them, with the matching `measure.<field>` label key. */
export const TWIN_MEASURES: readonly { id: MeasureId; field: string; unit: 'cm' }[] = [
  { id: 'height', field: 'heightCm', unit: 'cm' },
  { id: 'shoulder', field: 'shoulderCm', unit: 'cm' },
  { id: 'neck', field: 'neckCm', unit: 'cm' },
  { id: 'chest', field: 'chestCm', unit: 'cm' },
  { id: 'waist', field: 'waistCm', unit: 'cm' },
  { id: 'hip', field: 'hipCm', unit: 'cm' },
  { id: 'thigh', field: 'thighCm', unit: 'cm' },
  { id: 'upperArm', field: 'upperArmCm', unit: 'cm' },
  { id: 'armLength', field: 'armLengthCm', unit: 'cm' },
  { id: 'inseam', field: 'inseamCm', unit: 'cm' },
];

const MEASURE_IDS = new Set<string>([...TWIN_MEASURES.map((m) => m.id), 'footLength']);
const MACRO_SET = new Set<string>(MACRO_IDS);
/** The widest modifier range in the body manifest is +-1.5 (measure modifiers, ADR 0006). */
const MODIFIER_LIMIT = 1.5;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

function numberRecord(
  value: unknown,
  code: TwinErrorCode,
  what: string,
  accept: (key: string, v: number) => boolean,
): Record<string, number> {
  if (!isRecord(value)) throw new TwinFormatError(code, `${what} must be an object`);
  const out: Record<string, number> = {};
  for (const [key, v] of Object.entries(value)) {
    if (!isFiniteNumber(v) || !accept(key, v))
      throw new TwinFormatError(code, `${what}: invalid entry "${key}"`);
    out[key] = v;
  }
  return out;
}

/** Parses and validates a decoded `twin.json`. Throws `TwinFormatError`. */
export function parseTwinDef(value: unknown): TwinDef {
  if (!isRecord(value)) throw new TwinFormatError('json', 'not a JSON object');
  if (value.version !== 1)
    throw new TwinFormatError('version', `unsupported version ${String(value.version)}`);
  const boneOrder = value.boneOrder;
  if (
    !Array.isArray(boneOrder) ||
    boneOrder.length === 0 ||
    !boneOrder.every((n): n is string => typeof n === 'string' && n.length > 0) ||
    new Set(boneOrder).size !== boneOrder.length
  )
    throw new TwinFormatError('bones', 'boneOrder must be a list of unique bone names');
  const fittedMacros = numberRecord(
    value.fittedMacros,
    'macros',
    'fittedMacros',
    (key, v) => MACRO_SET.has(key) && v >= 0 && v <= 1,
  );
  const fittedModifiers = numberRecord(
    value.fittedModifiers,
    'modifiers',
    'fittedModifiers',
    (_key, v) => Math.abs(v) <= MODIFIER_LIMIT,
  );
  const measures = (raw: unknown, what: string): Record<string, number> =>
    numberRecord(raw, 'measurements', what, (key, v) => MEASURE_IDS.has(key) && v >= 0 && v < 400);
  const measurementsCm = measures(value.measurementsCm, 'measurementsCm');
  if (measurementsCm.height === undefined)
    throw new TwinFormatError('measurements', 'measurementsCm needs a height');
  const def: TwinDef = {
    version: 1,
    boneOrder,
    fittedMacros: fittedMacros as TwinDef['fittedMacros'],
    fittedModifiers,
    measurementsCm: measurementsCm as TwinDef['measurementsCm'],
  };
  if (value.measurementsRawCm !== undefined)
    def.measurementsRawCm = measures(
      value.measurementsRawCm,
      'measurementsRawCm',
    ) as TwinDef['measurementsRawCm'];
  if (value.clothingAllowanceCm !== undefined)
    def.clothingAllowanceCm = numberRecord(
      value.clothingAllowanceCm,
      'measurements',
      'clothingAllowanceCm',
      (key, v) => MEASURE_IDS.has(key) && v >= 0 && v < 50,
    ) as TwinDef['clothingAllowanceCm'];
  const mapping = value.mapping;
  if (mapping !== undefined) {
    if (
      !isRecord(mapping) ||
      typeof mapping.file !== 'string' ||
      !Number.isInteger(mapping.twinVertexCount) ||
      (mapping.twinVertexCount as number) <= 0 ||
      !Number.isInteger(mapping.renderVertexCount) ||
      (mapping.renderVertexCount as number) <= 0
    )
      throw new TwinFormatError('mapping', 'invalid mapping block');
    def.mapping = {
      file: mapping.file,
      twinVertexCount: mapping.twinVertexCount as number,
      renderVertexCount: mapping.renderVertexCount as number,
    };
  }
  if (value.restHeadsM !== undefined) {
    if (!isRecord(value.restHeadsM))
      throw new TwinFormatError('bones', 'restHeadsM must be an object');
    const heads: Record<string, [number, number, number]> = {};
    for (const [name, head] of Object.entries(value.restHeadsM)) {
      if (!Array.isArray(head) || head.length !== 3 || !head.every(isFiniteNumber))
        throw new TwinFormatError('bones', `restHeadsM: invalid head for "${name}"`);
      heads[name] = [head[0] as number, head[1] as number, head[2] as number];
    }
    def.restHeadsM = heads;
  }
  if (isRecord(value.fit) && isFiniteNumber(value.fit.twinHeightM))
    def.fit = { twinHeightM: value.fit.twinHeightM };
  return def;
}

/** Parses the text of a twin.json file. */
export function parseTwinJson(text: string): TwinDef {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new TwinFormatError('json', 'not valid JSON');
  }
  return parseTwinDef(raw);
}

/** The macro variables handed to the solver: the fitted ones, everything else stays at the manifest default. */
export function twinMacros(def: TwinDef): Partial<Record<MacroVar, number>> {
  return { ...def.fittedMacros };
}
