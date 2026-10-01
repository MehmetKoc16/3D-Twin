// @ts-expect-error Node types are not in the browser app's tsconfig; Vitest supplies this module.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  applyMorphs,
  buildBasePositions,
  combineWeights,
  type BodyManifest,
  type MeasuresDef,
} from '@dt/avatar-core';
import { bytes, parseGlb, primitive, root } from '../features/wardrobe/realAssets.testkit';
import { fixedShapeWeights, solveFixedShape, type FixedShape } from './fixedShape';

/** The realistic twin's fixed body on the real MakeHuman assets (skipped when they are not built). */
describe.skipIf(!root)('fixed shape (twin body)', () => {
  const dir = root ?? '';
  const manifest = JSON.parse(readFileSync(`${dir}/body/manifest.json`, 'utf8')) as BodyManifest;
  const measures = JSON.parse(readFileSync(`${dir}/body/measures.json`, 'utf8')) as MeasuresDef;
  const baseGlb = primitive(parseGlb(readFileSync(`${dir}/body/base.glb`)));
  const base = buildBasePositions(baseGlb.position, manifest);
  const morphs = bytes(`${dir}/body/morphs.bin`);
  const data = { manifest, base, morphs, measures, indices: baseGlb.indices };

  it('without macros and modifiers it is the neutral body (about 1.66 m, plausible mass)', () => {
    const result = solveFixedShape(data, { macros: {}, modifiers: {} });
    expect(result.achievedCm.height).toBeCloseTo(165.9, 0);
    expect(result.estimatedMassKg).toBeGreaterThan(50);
    expect(result.estimatedMassKg).toBeLessThan(90);
    expect(result.positions.length).toBe(manifest.vertexCount * 3);
  });

  it('applies macro variables and net modifier values exactly like avatar-core does', () => {
    const shape: FixedShape = {
      macros: { gender: 0.8, weight: 0.7, height: 0.6 },
      modifiers: { 'measure/measure-waist-circ': 0.4, 'torso/torso-scale-horiz': -0.25 },
    };
    const result = solveFixedShape(data, shape);
    const expected = new Float32Array(base.length);
    applyMorphs(
      base,
      morphs,
      manifest,
      combineWeights(manifest, shape.macros, shape.modifiers),
      expected,
    );
    expect(Array.from(result.positions.slice(0, 3000))).toEqual(
      Array.from(expected.slice(0, 3000)),
    );
    const neutral = solveFixedShape(data, { macros: {}, modifiers: {} });
    expect(result.achievedCm.height!).toBeGreaterThan(neutral.achievedCm.height!); // taller macro
    expect(result.achievedCm.waist!).not.toBeCloseTo(neutral.achievedCm.waist!, 0);
  });

  it('measures every measures.json entry in cm', () => {
    const result = solveFixedShape(data, { macros: {}, modifiers: {} });
    for (const def of measures.measures)
      expect(result.achievedCm[def.id], def.id).toBeGreaterThan(0);
    expect(result.achievedCm.chest!).toBeGreaterThan(60);
    expect(result.achievedCm.chest!).toBeLessThan(140);
  });

  it('rejects unknown modifier ids and non-finite values before anything is applied', () => {
    expect(() =>
      fixedShapeWeights(manifest, { macros: {}, modifiers: { 'no/such-modifier': 0.2 } }),
    ).toThrow(/unknown modifier/);
    expect(() =>
      fixedShapeWeights(manifest, {
        macros: {},
        modifiers: { 'torso/torso-scale-horiz': Number.NaN },
      }),
    ).toThrow(/non-finite/);
  });
});
