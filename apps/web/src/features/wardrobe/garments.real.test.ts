// @ts-expect-error Node types are not in the browser app's tsconfig; Vitest supplies this module.
import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  applyMorphs,
  bindGarment,
  buildBasePositions,
  createBodySolver,
  createGarmentSections,
  garmentClearance,
  gradeGarment,
  gradeGarmentLength,
  garmentSkinWeights,
  macroWeights,
  parseGarmentBinding,
  type BodyManifest,
  type BodyParams,
  type GarmentTemplateDef,
  type MeasuresDef,
  type StoreItemDef,
} from '@dt/avatar-core';
import { groundRenderPositions } from '../../workers/ground';
import { hiddenVertexMask, filterBodyIndex, parseDeleteVerts } from './bodyHide';
import { analyzeItemFit } from './fitAnalysis';
import { bodyCmFromSolve, bodyPlanes, buildGradeRings, buildLengthGrades } from './garmentGrading';
import { buildWeldMap } from '../avatar/meshMath';
import { bytes, parseGlb, primitive, root } from './realAssets.testkit';

/** Integration check against the real body and garment assets (skipped when they are not built). */

const chartFor = (t: GarmentTemplateDef, offset: number): StoreItemDef => {
  const chart: StoreItemDef['chart'] = {};
  for (const [id, native] of Object.entries(t.nativeMeasures)) {
    chart[id as keyof StoreItemDef['chart']] = [
      native - 6 + offset,
      native + offset,
      native + 8 + offset,
    ];
  }
  return {
    id: t.id,
    name: t.id,
    templateId: t.id,
    color: '#3366cc',
    sizes: ['S', 'M', 'L'],
    chart,
    selectedSize: 'M',
  };
};

const params = (
  gender: number,
  heightCm: number,
  weightKg: number,
  chestCm: number,
  waistCm: number,
  hipCm: number,
): BodyParams => ({
  gender,
  heightCm,
  weightKg,
  chestCm,
  waistCm,
  hipCm,
  shoe: { system: 'EU', size: 42 },
});

describe.skipIf(!root)('wardrobe on the real assets', () => {
  const dir = root ?? '';
  const manifest = JSON.parse(readFileSync(`${dir}/body/manifest.json`, 'utf8')) as BodyManifest;
  const measures = JSON.parse(readFileSync(`${dir}/body/measures.json`, 'utf8')) as MeasuresDef;
  const baseGlb = primitive(parseGlb(readFileSync(`${dir}/body/base.glb`)));
  const base = buildBasePositions(baseGlb.position, manifest);
  const morphs = bytes(`${dir}/body/morphs.bin`);
  const renderCount = manifest.renderVertexCount;
  const catalogue = JSON.parse(readFileSync(`${dir}/garments/index.json`, 'utf8')) as {
    garments: GarmentTemplateDef[];
  };
  const solver = createBodySolver({ manifest, base, morphs, measures, indices: baseGlb.indices });
  const weld = buildWeldMap(baseGlb.position);

  const neutral = (): Float32Array => {
    const out = new Float32Array(base.length);
    applyMorphs(base, morphs, manifest, macroWeights(manifest, {}), out);
    return groundRenderPositions(out, renderCount).positions;
  };

  it('lists templates whose files exist and whose binding matches the mesh', () => {
    expect(catalogue.garments.length).toBeGreaterThanOrEqual(4);
    for (const t of catalogue.garments) {
      expect(existsSync(`${dir}/garments/${t.mesh}`), t.id).toBe(true);
      expect(t.binding && existsSync(`${dir}/garments/${t.binding}`), t.id).toBeTruthy();
      const mesh = primitive(parseGlb(readFileSync(`${dir}/garments/${t.mesh}`)));
      const binding = parseGarmentBinding(bytes(`${dir}/garments/${t.binding!}`), renderCount);
      expect(binding.count, t.id).toBe(mesh.position.length / 3);
    }
  });

  it('re-binding the neutral body reproduces the stored garment positions', () => {
    const body = neutral();
    for (const t of catalogue.garments) {
      const mesh = primitive(parseGlb(readFileSync(`${dir}/garments/${t.mesh}`)));
      const binding = parseGarmentBinding(bytes(`${dir}/garments/${t.binding!}`), renderCount);
      const out = new Float32Array(mesh.position.length);
      bindGarment(binding, t.scaleRefs, body, out);
      let sum = 0;
      let max = 0;
      for (let v = 0; v < binding.count; v++) {
        const d = Math.hypot(
          out[v * 3]! - mesh.position[v * 3]!,
          out[v * 3 + 1]! - mesh.position[v * 3 + 1]!,
          out[v * 3 + 2]! - mesh.position[v * 3 + 2]!,
        );
        sum += d;
        max = Math.max(max, d);
      }
      console.log(
        `[real] ${t.id}: neutral rebind mean ${((sum / binding.count) * 1000).toFixed(2)} mm, max ${(max * 1000).toFixed(1)} mm`,
      );
      expect(sum / binding.count, t.id).toBeLessThan(0.004);
    }
  });

  it('skin weights of every template are normalised and use existing bones', () => {
    for (const t of catalogue.garments) {
      const binding = parseGarmentBinding(bytes(`${dir}/garments/${t.binding!}`), renderCount);
      const { skinIndices, skinWeights } = garmentSkinWeights(
        binding,
        baseGlb.joints!,
        baseGlb.weights!,
      );
      for (let v = 0; v < binding.count; v++) {
        const sum =
          skinWeights[v * 4]! +
          skinWeights[v * 4 + 1]! +
          skinWeights[v * 4 + 2]! +
          skinWeights[v * 4 + 3]!;
        expect(sum, `${t.id} ${v}`).toBeCloseTo(1, 4);
      }
      expect(Math.max(...skinIndices), t.id).toBeLessThan(53);
    }
  });

  const bodies: [string, BodyParams][] = [
    ['female 160/52', params(0, 160, 52, 84, 66, 92)],
    ['male 182/95', params(1, 182, 95, 106, 94, 104)],
  ];

  for (const [name, p] of bodies) {
    it(`fits every template to ${name} without penetrating and reports a sensible fit`, () => {
      const res = solver.solve(p);
      const grounded = groundRenderPositions(res.positions, renderCount).positions;
      const planes = bodyPlanes(measures, grounded);
      const sectionStart = performance.now();
      const sections = createGarmentSections(grounded);
      const sectionMs = performance.now() - sectionStart;
      console.log(`[real] ${name}: sections ${sectionMs.toFixed(1)} ms`);
      expect(planes.crotchY).toBeGreaterThan(0.6);
      expect(planes.planeY.chest).toBeGreaterThan(1.0);
      for (const t of catalogue.garments) {
        const binding = parseGarmentBinding(bytes(`${dir}/garments/${t.binding!}`), renderCount);
        const item = chartFor(t, 4);
        const rest = new Float32Array(binding.count * 3);
        const graded = new Float32Array(binding.count * 3);
        const clearance = new Float32Array(binding.count);
        const t0 = performance.now();
        bindGarment(binding, t.scaleRefs, grounded, rest);
        const rings = buildGradeRings(item, t, bodyCmFromSolve(res.achievedCm), planes);
        gradeGarment(rest, binding, grounded, rings, graded, sections);
        for (const spec of buildLengthGrades(
          item,
          t,
          res.achievedCm,
          planes,
          rest,
          () => undefined,
        ))
          gradeGarmentLength(graded, binding, grounded, spec, graded);
        garmentClearance(graded, binding, grounded, clearance, sections);
        const ms = performance.now() - t0;
        const sorted = Float32Array.from(clearance).sort();
        const p02 = sorted[Math.floor(sorted.length * 0.02)]!;
        console.log(
          `[real] ${name} ${t.id}: ${binding.count} verts, ${ms.toFixed(1)} ms, rings ${rings.length}, clearance p2 ${(p02 * 1000).toFixed(1)} mm, median ${(sorted[sorted.length >> 1]! * 1000).toFixed(1)} mm`,
        );
        expect(Number.isFinite(p02)).toBe(true);
        expect(p02, `${t.id} penetrates`).toBeGreaterThan(-0.004);
        expect(ms, `${t.id} update time`).toBeLessThan(200);
        const report = analyzeItemFit(item, t, res.achievedCm);
        expect(report, t.id).toBeDefined();
      }
    });
  }

  it('a 3 cm longer jeans inseam lowers its hem and keeps 99% of leg vertices outside', () => {
    const jeans = catalogue.garments.find((t) => t.id === 'jeans')!;
    const res = solver.solve(params(1, 182, 95, 106, 94, 104));
    const body = groundRenderPositions(res.positions, renderCount).positions;
    const planes = bodyPlanes(measures, body);
    const sections = createGarmentSections(body);
    const binding = parseGarmentBinding(bytes(`${dir}/garments/${jeans.binding!}`), renderCount);
    const rest = new Float32Array(binding.count * 3);
    const graded = new Float32Array(rest.length);
    bindGarment(binding, jeans.scaleRefs, body, rest);
    const item = chartFor(jeans, 0);
    item.chart.inseam = [0, res.achievedCm.inseam! + (jeans.defaultEase.inseam ?? 0) + 3, 0];
    const rings = buildGradeRings(item, jeans, bodyCmFromSolve(res.achievedCm), planes);
    gradeGarment(rest, binding, body, rings, graded, sections);
    const before = Math.min(...Array.from(graded).filter((_, i) => i % 3 === 1));
    for (const spec of buildLengthGrades(
      item,
      jeans,
      res.achievedCm,
      planes,
      rest,
      () => undefined,
    ))
      gradeGarmentLength(graded, binding, body, spec, graded);
    const after = Math.min(...Array.from(graded).filter((_, i) => i % 3 === 1));
    expect(before - after).toBeCloseTo(0.03, 2);
    const clearance = new Float32Array(binding.count);
    garmentClearance(graded, binding, body, clearance, sections);
    const sorted = Float32Array.from(clearance).sort();
    expect(sorted[Math.floor(sorted.length * 0.01)]!).toBeGreaterThanOrEqual(-0.001);
  });

  it('hides body triangles under the garments that ship a delete list', () => {
    const withDelete = catalogue.garments.filter((t) => t.deleteVerts);
    expect(withDelete.length).toBeGreaterThan(0);
    const lists = withDelete.map((t) =>
      parseDeleteVerts(bytes(`${dir}/garments/${t.deleteVerts!}`)),
    );
    const mask = hiddenVertexMask(lists, renderCount, weld);
    const filtered = filterBodyIndex(baseGlb.indices, mask);
    const hiddenTriangles = (baseGlb.indices.length - filtered.length) / 3;
    console.log(
      `[real] delete lists hide ${hiddenTriangles} of ${baseGlb.indices.length / 3} body triangles`,
    );
    expect(hiddenTriangles).toBeGreaterThan(200);
    expect(hiddenTriangles).toBeLessThan(baseGlb.indices.length / 3 / 2);
  });
});
