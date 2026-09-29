// @ts-expect-error Node types are not in the browser app's tsconfig; Vitest supplies this module.
import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  applyMorphs,
  bindGarment,
  buildBasePositions,
  createBodySolver,
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
import { bodyCmFromSolve, bodyPlanes, buildGradeRings, clearanceLegAware, gradeGarmentLegAware, SplitScratch } from './garmentGrading';
import { buildWeldMap } from '../avatar/meshMath';

/** Integration check against the real body and garment assets (skipped when they are not built). */

const root = ['public/assets', 'apps/web/public/assets'].find((d) => existsSync(`${d}/garments/index.json`) && existsSync(`${d}/body/manifest.json`));

interface Glb {
  json: {
    meshes: { primitives: { attributes: Record<string, number>; indices?: number }[] }[];
    accessors: { bufferView: number; byteOffset?: number; componentType: number; count: number; type: string }[];
    bufferViews: { byteOffset?: number; byteLength: number; byteStride?: number }[];
  };
  bin: DataView;
}

function parseGlb(buffer: Uint8Array): Glb {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const jsonLength = view.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(buffer.subarray(20, 20 + jsonLength))) as Glb['json'];
  const binStart = 20 + jsonLength + 8;
  return { json, bin: new DataView(buffer.buffer, buffer.byteOffset + binStart, buffer.byteLength - binStart) };
}

const COMPONENTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

function readAccessor(glb: Glb, index: number): Float64Array {
  const accessor = glb.json.accessors[index]!;
  const view = glb.json.bufferViews[accessor.bufferView]!;
  const n = COMPONENTS[accessor.type]!;
  const size = accessor.componentType === 5126 || accessor.componentType === 5125 ? 4 : 2;
  const stride = view.byteStride ?? size * n;
  const out = new Float64Array(accessor.count * n);
  for (let i = 0; i < accessor.count; i++) {
    for (let k = 0; k < n; k++) {
      const offset = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0) + i * stride + k * size;
      out[i * n + k] =
        accessor.componentType === 5126
          ? glb.bin.getFloat32(offset, true)
          : accessor.componentType === 5125
            ? glb.bin.getUint32(offset, true)
            : glb.bin.getUint16(offset, true);
    }
  }
  return out;
}

function primitive(glb: Glb): { position: Float32Array; indices: Uint32Array; joints?: Float64Array; weights?: Float64Array } {
  const attributes = glb.json.meshes[0]!.primitives[0]!.attributes;
  const p = glb.json.meshes[0]!.primitives[0]!;
  return {
    position: Float32Array.from(readAccessor(glb, attributes.POSITION!)),
    indices: Uint32Array.from(readAccessor(glb, p.indices!)),
    ...(attributes.JOINTS_0 !== undefined ? { joints: readAccessor(glb, attributes.JOINTS_0) } : {}),
    ...(attributes.WEIGHTS_0 !== undefined ? { weights: readAccessor(glb, attributes.WEIGHTS_0) } : {}),
  };
}

function bytes(path: string): ArrayBuffer {
  const b = readFileSync(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

const chartFor = (t: GarmentTemplateDef, offset: number): StoreItemDef => {
  const chart: StoreItemDef['chart'] = {};
  for (const [id, native] of Object.entries(t.nativeMeasures)) {
    chart[id as keyof StoreItemDef['chart']] = [native - 6 + offset, native + offset, native + 8 + offset];
  }
  return { id: t.id, name: t.id, templateId: t.id, color: '#3366cc', sizes: ['S', 'M', 'L'], chart, selectedSize: 'M' };
};

const params = (gender: number, heightCm: number, weightKg: number, chestCm: number, waistCm: number, hipCm: number): BodyParams => ({
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
  const catalogue = JSON.parse(readFileSync(`${dir}/garments/index.json`, 'utf8')) as { garments: GarmentTemplateDef[] };
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
        const d = Math.hypot(out[v * 3]! - mesh.position[v * 3]!, out[v * 3 + 1]! - mesh.position[v * 3 + 1]!, out[v * 3 + 2]! - mesh.position[v * 3 + 2]!);
        sum += d;
        max = Math.max(max, d);
      }
      console.log(`[real] ${t.id}: neutral rebind mean ${((sum / binding.count) * 1000).toFixed(2)} mm, max ${(max * 1000).toFixed(1)} mm`);
      expect(sum / binding.count, t.id).toBeLessThan(0.004);
    }
  });

  it('skin weights of every template are normalised and use existing bones', () => {
    for (const t of catalogue.garments) {
      const binding = parseGarmentBinding(bytes(`${dir}/garments/${t.binding!}`), renderCount);
      const { skinIndices, skinWeights } = garmentSkinWeights(binding, baseGlb.joints!, baseGlb.weights!);
      for (let v = 0; v < binding.count; v++) {
        const sum = skinWeights[v * 4]! + skinWeights[v * 4 + 1]! + skinWeights[v * 4 + 2]! + skinWeights[v * 4 + 3]!;
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
      expect(planes.crotchY).toBeGreaterThan(0.6);
      expect(planes.planeY.chest).toBeGreaterThan(1.0);
      for (const t of catalogue.garments) {
        const binding = parseGarmentBinding(bytes(`${dir}/garments/${t.binding!}`), renderCount);
        const item = chartFor(t, 4);
        const rest = new Float32Array(binding.count * 3);
        const graded = new Float32Array(binding.count * 3);
        const clearance = new Float32Array(binding.count);
        const scratch = new SplitScratch();
        const t0 = performance.now();
        bindGarment(binding, t.scaleRefs, grounded, rest);
        const rings = buildGradeRings(item, t, bodyCmFromSolve(res.achievedCm), planes);
        const crotch = t.category === 'top' ? undefined : planes.crotchY;
        gradeGarmentLegAware(rest, binding, grounded, rings, crotch, graded, scratch);
        clearanceLegAware(graded, binding, grounded, crotch, clearance, scratch);
        const ms = performance.now() - t0;
        const sorted = Float32Array.from(clearance).sort();
        const p02 = sorted[Math.floor(sorted.length * 0.02)]!;
        console.log(`[real] ${name} ${t.id}: ${binding.count} verts, ${ms.toFixed(1)} ms, rings ${rings.length}, clearance p2 ${(p02 * 1000).toFixed(1)} mm, median ${(sorted[sorted.length >> 1]! * 1000).toFixed(1)} mm`);
        expect(Number.isFinite(p02)).toBe(true);
        expect(p02, `${t.id} penetrates`).toBeGreaterThan(-0.004);
        expect(ms, `${t.id} update time`).toBeLessThan(200);
        const report = analyzeItemFit(item, t, res.achievedCm);
        expect(report, t.id).toBeDefined();
      }
    });
  }

  it('hides body triangles under the garments that ship a delete list', () => {
    const withDelete = catalogue.garments.filter((t) => t.deleteVerts);
    expect(withDelete.length).toBeGreaterThan(0);
    const lists = withDelete.map((t) => parseDeleteVerts(bytes(`${dir}/garments/${t.deleteVerts!}`)));
    const mask = hiddenVertexMask(lists, renderCount, weld);
    const filtered = filterBodyIndex(baseGlb.indices, mask);
    const hiddenTriangles = (baseGlb.indices.length - filtered.length) / 3;
    console.log(`[real] delete lists hide ${hiddenTriangles} of ${baseGlb.indices.length / 3} body triangles`);
    expect(hiddenTriangles).toBeGreaterThan(200);
    expect(hiddenTriangles).toBeLessThan(baseGlb.indices.length / 3 / 2);
  });
});
