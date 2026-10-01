import { describe, expect, it } from 'vitest';
import {
  parseTwinDef,
  parseTwinJson,
  TwinFormatError,
  twinMacros,
  type TwinErrorCode,
} from './twinDef';

const valid = {
  version: 1,
  generator: 'dijital-ikiz twin-lab rig',
  glb: 'rigged.glb',
  boneOrder: ['Root', 'pelvis', 'spine_01'],
  fittedMacros: { gender: 0.6, muscle: 0.5, weight: 0.55, height: 0.52 },
  fittedModifiers: { 'measure/measure-waist-circ': 0.4, 'torso/torso-scale-horiz': -0.2 },
  measurementsRawCm: { height: 172.5, chest: 96.2, waist: 80.1 },
  clothingAllowanceCm: { height: 3, chest: 3, waist: 3 },
  measurementsCm: { height: 169.5, chest: 93.2, waist: 77.1 },
  mapping: {
    file: 'mh2twin.bin',
    format: 'uint32le',
    twinVertexCount: 3,
    renderVertexCount: 14517,
  },
  fit: { twinHeightM: 1.72, bodyLowestYM: 0.001 },
};

function code(mutate: (v: Record<string, unknown>) => void): TwinErrorCode | undefined {
  const value = structuredClone(valid) as Record<string, unknown>;
  mutate(value);
  try {
    parseTwinDef(value);
  } catch (error) {
    return error instanceof TwinFormatError ? error.code : undefined;
  }
  return undefined;
}

describe('parseTwinDef', () => {
  it('accepts what twin_export.py writes and keeps the fields the app uses', () => {
    const def = parseTwinDef(valid);
    expect(def.boneOrder).toEqual(['Root', 'pelvis', 'spine_01']);
    expect(def.fittedMacros.gender).toBe(0.6);
    expect(def.fittedModifiers['torso/torso-scale-horiz']).toBe(-0.2);
    expect(def.measurementsCm.chest).toBe(93.2);
    expect(def.measurementsRawCm?.height).toBe(172.5);
    expect(def.clothingAllowanceCm?.waist).toBe(3);
    expect(def.mapping).toEqual({
      file: 'mh2twin.bin',
      twinVertexCount: 3,
      renderVertexCount: 14517,
    });
    expect(def.fit?.twinHeightM).toBe(1.72);
  });

  it('twinMacros hands the fitted macros to the solver as they are', () => {
    expect(twinMacros(parseTwinDef(valid))).toEqual(valid.fittedMacros);
  });

  it('mapping and the raw / allowance blocks are optional', () => {
    const value = structuredClone(valid) as Record<string, unknown>;
    delete value.mapping;
    delete value.measurementsRawCm;
    delete value.clothingAllowanceCm;
    const def = parseTwinDef(value);
    expect(def.mapping).toBeUndefined();
    expect(def.measurementsRawCm).toBeUndefined();
  });

  it('rejects malformed documents with a stable error code', () => {
    expect(() => parseTwinDef([])).toThrow(TwinFormatError);
    expect(code((v) => (v.version = 2))).toBe('version');
    expect(code((v) => delete v.version)).toBe('version');
    expect(code((v) => (v.boneOrder = []))).toBe('bones');
    expect(code((v) => (v.boneOrder = ['a', 'a']))).toBe('bones');
    expect(code((v) => (v.boneOrder = ['a', 3]))).toBe('bones');
    expect(code((v) => (v.fittedMacros = { gender: 1.4 }))).toBe('macros');
    expect(code((v) => (v.fittedMacros = { colour: 0.5 }))).toBe('macros');
    expect(code((v) => (v.fittedMacros = { gender: 'high' }))).toBe('macros');
    expect(code((v) => (v.fittedModifiers = { 'a/b': 4 }))).toBe('modifiers');
    expect(code((v) => (v.fittedModifiers = { 'a/b': Number.NaN }))).toBe('modifiers');
    expect(code((v) => (v.fittedModifiers = null))).toBe('modifiers');
    expect(code((v) => (v.measurementsCm = { chest: 90 }))).toBe('measurements'); // no height
    expect(code((v) => (v.measurementsCm = { height: 170, sleeve: 20 }))).toBe('measurements'); // unknown measure
    expect(code((v) => (v.measurementsCm = { height: -3 }))).toBe('measurements');
    expect(
      code((v) => (v.mapping = { file: 'x.bin', twinVertexCount: 0, renderVertexCount: 10 })),
    ).toBe('mapping');
    expect(code((v) => (v.mapping = 'mh2twin.bin'))).toBe('mapping');
  });

  it('parseTwinJson reports invalid JSON', () => {
    expect(() => parseTwinJson('{nope')).toThrow(/valid JSON/);
    try {
      parseTwinJson('{nope');
    } catch (error) {
      expect((error as TwinFormatError).code).toBe('json');
    }
    expect(parseTwinJson(JSON.stringify(valid)).version).toBe(1);
  });
});
