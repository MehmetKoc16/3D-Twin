import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { GarmentTemplateDef, StoreItemDef } from '../garmentContracts';
import { createSyntheticBody } from '../testing/syntheticBody';
import { bindGarment, computeScale, parseGarmentBinding, type GarmentBinding } from './binding';
import { gradeGarment } from './grade';
import { garmentClearance, clearanceToColor } from './heatmap';
import { garmentSkinWeights } from './skin';
import { analyzeFit, bodyMeasuresForGarment, recommendSize, verdictForEase } from './fit';

function encode(indices: number[], weights: number[], offsets: number[]): ArrayBuffer {
  const buffer = new ArrayBuffer((indices.length / 3) * 36);
  const view = new DataView(buffer);
  for (let v = 0; v < indices.length / 3; v++) {
    for (let j = 0; j < 3; j++) {
      view.setUint32(v * 36 + j * 4, indices[v * 3 + j]!, true);
      view.setFloat32(v * 36 + 12 + j * 4, weights[v * 3 + j]!, true);
      view.setFloat32(v * 36 + 24 + j * 4, offsets[v * 3 + j]!, true);
    }
  }
  return buffer;
}

function ringFixture(): {
  body: ReturnType<typeof createSyntheticBody>;
  binding: GarmentBinding;
  y: number;
} {
  const body = createSyntheticBody({ rings: 24, segments: 32, decoyModifiers: 0 });
  const ring = body.ringAt(0.72);
  const indices: number[] = [],
    weights: number[] = [],
    offsets: number[] = [];
  for (let s = 0; s < body.segments; s++) {
    const vertex = ring * body.segments + s;
    const p = vertex * 3;
    const radius = Math.hypot(body.base[p]!, body.base[p + 2]!);
    indices.push(vertex, vertex, vertex);
    weights.push(1, 0, 0);
    offsets.push((body.base[p]! / radius) * 0.02, 0, (body.base[p + 2]! / radius) * 0.02);
  }
  return {
    body,
    binding: parseGarmentBinding(encode(indices, weights, offsets), body.manifest.vertexCount),
    y: body.base[ring * body.segments * 3 + 1]!,
  };
}

describe('garment binding', () => {
  it('reads little-endian entries and rejects malformed data', () => {
    const bytes = encode([0, 1, 2], [0.2, 0.3, 0.5], [0.01, 0.02, 0.03]);
    expect(parseGarmentBinding(bytes, 3).weights[2]).toBeCloseTo(0.5);
    expect(() => parseGarmentBinding(new ArrayBuffer(35))).toThrow(/multiple of 36/);
    expect(() => parseGarmentBinding(bytes, 2)).toThrow(/out of range/);
    const bad = bytes.slice(0);
    new DataView(bad).setFloat32(12, NaN, true);
    expect(() => parseGarmentBinding(bad)).toThrow(/non-finite/);
    new DataView(bad).setFloat32(12, 0.1, true);
    expect(() => parseGarmentBinding(bad)).toThrow(/sum to one/);
    const tolerant = bytes.slice(0);
    new DataView(tolerant).setFloat32(12, 0.22, true);
    expect(parseGarmentBinding(tolerant).count).toBe(1);
  });

  it('tracks a changed body and scales offsets by axis references', () => {
    const { body, binding } = ringFixture();
    const out = new Float32Array(binding.count * 3);
    bindGarment(binding, undefined, body.base, out);
    expect(out[0]).toBeCloseTo(body.base[binding.indices[0]! * 3]! + 0.02, 5);
    const changed = body.base.slice();
    for (let v = 0; v < body.manifest.vertexCount; v++) {
      changed[v * 3] = changed[v * 3]! * 1.2;
      changed[v * 3 + 2] = changed[v * 3 + 2]! * 1.2;
    }
    bindGarment(binding, undefined, changed, out);
    expect(out[0]).toBeCloseTo(body.base[binding.indices[0]! * 3]! * 1.2 + 0.02, 5);
    const a = binding.indices[0]!;
    const opposite = a + body.segments / 2;
    const top = (body.rings - 1) * body.segments;
    const refs = {
      x: [a, opposite, Math.abs(body.base[a * 3]! - body.base[opposite * 3]!)] as [
        number,
        number,
        number,
      ],
      y: [0, top, body.base[top * 3 + 1]!] as [number, number, number],
      z: [a + 8, a + 24, Math.abs(body.base[(a + 8) * 3 + 2]! - body.base[(a + 24) * 3 + 2]!)] as [
        number,
        number,
        number,
      ],
    };
    expect(computeScale(refs, changed)[0]).toBeCloseTo(1.2);
    bindGarment(binding, refs, changed, out);
    expect(out[0]).toBeCloseTo(body.base[a * 3]! * 1.2 + 0.024, 5);
  });

  it('blends duplicate bones, keeps top four, and normalizes', () => {
    const binding = parseGarmentBinding(encode([0, 1, 2], [0.5, 0.3, 0.2], [0, 0, 0]));
    const indices = [0, 1, 2, 3, 1, 4, 5, 6, 2, 7, 8, 9];
    const weights = [0.4, 0.3, 0.2, 0.1, 0.5, 0.3, 0.2, 0, 0.5, 0.3, 0.2, 0];
    const skin = garmentSkinWeights(binding, indices, weights);
    expect(Array.from(skin.skinIndices)).toEqual([1, 2, 0, 4]);
    expect(skin.skinWeights[0]).toBeCloseTo(0.3 / 0.79);
    expect(Array.from(skin.skinWeights).reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });
});

describe('garment geometry', () => {
  it('grades a ring by the requested circumference and enforces 2 mm clearance', () => {
    const { body, binding, y } = ringFixture();
    const rest = new Float32Array(binding.count * 3);
    bindGarment(binding, undefined, body.base, rest);
    const graded = new Float32Array(rest.length);
    gradeGarment(rest, binding, body.base, [{ planeY: y, deltaCircumferenceM: 0.12 }], graded);
    const perimeter = (positions: Float32Array): number => {
      let result = 0;
      for (let s = 0; s < binding.count; s++) {
        const a = s * 3,
          b = ((s + 1) % binding.count) * 3;
        result += Math.hypot(positions[a]! - positions[b]!, positions[a + 2]! - positions[b + 2]!);
      }
      return result;
    };
    expect(Math.abs((perimeter(graded) - perimeter(rest) - 0.12) / 0.12)).toBeLessThan(0.02);
    const inset = rest.map((value, i) => (i % 3 === 0 || i % 3 === 2 ? value * 0.8 : value));
    gradeGarment(inset, binding, body.base, [], graded);
    const clearance = new Float32Array(binding.count);
    garmentClearance(graded, binding, body.base, clearance);
    expect(Math.min(...clearance)).toBeGreaterThan(0.0019);
  });

  it('reports signed radial clearance and maps heatmap stops', () => {
    const { body, binding } = ringFixture();
    const rest = new Float32Array(binding.count * 3);
    bindGarment(binding, undefined, body.base, rest);
    const clearance = new Float32Array(binding.count);
    garmentClearance(rest, binding, body.base, clearance);
    expect(clearance[0]).toBeCloseTo(0.02, 5);
    rest[0] = body.base[binding.indices[0]! * 3]! - 0.01;
    garmentClearance(rest, binding, body.base, clearance);
    expect(clearance[0]).toBeCloseTo(-0.01, 5);
    const color = new Float32Array(3);
    clearanceToColor(-0.01, color);
    expect(Array.from(color)).toEqual([1, 0, 0]);
    clearanceToColor(0.02, color);
    expect(Array.from(color)).toEqual([0, 1, 0]);
    clearanceToColor(0.06, color);
    expect(Array.from(color)).toEqual([0, 0, 1]);
  });
});

const template: GarmentTemplateDef = {
  id: 'shirt',
  kind: 'tshirt',
  category: 'top',
  label: { tr: 'Tişört', en: 'T-shirt' },
  license: 'CC0-1.0',
  mesh: 'shirt.glb',
  nativeMeasures: {},
  defaultEase: { chest: 8, waist: 6 },
  layer: 1,
  baseColor: '#ffffff',
};
const item: StoreItemDef = {
  id: 'store-shirt',
  name: 'Shirt',
  templateId: 'shirt',
  color: '#fff',
  sizes: ['S', 'M', 'L'],
  chart: { chest: [98, 108, 118], waist: [82, 90, 100], sleeve: [58, 60, 62] },
  selectedSize: 'M',
};

describe('fit assessment', () => {
  it('uses documented threshold boundaries', () => {
    expect(verdictForEase('chest', -0.01)).toBe('tight');
    expect(verdictForEase('chest', 0)).toBe('snug');
    expect(verdictForEase('chest', 4)).toBe('regular');
    expect(verdictForEase('chest', 12)).toBe('regular');
    expect(verdictForEase('chest', 20)).toBe('loose');
    expect(verdictForEase('chest', 20.01)).toBe('oversized');
    expect(verdictForEase('footLength', 0.49)).toBe('tight');
    expect(verdictForEase('footLength', 1)).toBe('regular');
    expect(verdictForEase('footLength', 1.8)).toBe('regular');
    expect(verdictForEase('footLength', 2.51)).toBe('oversized');
    expect(verdictForEase('sleeve', -2.01)).toBe('tight');
    expect(verdictForEase('sleeve', 2)).toBe('regular');
  });

  it('reports girth verdicts and recommends the closest non-tight size', () => {
    const body = { chest: 100, waist: 88, sleeve: 59 };
    const report = analyzeFit(item, template, body);
    expect(report.overall).toBe('snug');
    expect(report.recommendedSize).toBe('M');
    expect(report.regions.find((r) => r.id === 'chest')?.easeCm).toBe(8);
    expect(recommendSize(item, template, { chest: 110, waist: 94 })).toBe('L');
    expect(recommendSize(item, template, { chest: 130, waist: 110 })).toBeUndefined();
    expect(
      bodyMeasuresForGarment({ chest: 100, armLength: 62, footLength: 26, height: 180 }),
    ).toEqual({ chest: 100, sleeve: 62, footLength: 26 });
  });
});

const garmentIndexPaths = [
  'apps/web/public/assets/garments/index.json',
  '../../apps/web/public/assets/garments/index.json',
  '../../../../apps/web/public/assets/garments/index.json',
];
const garmentIndex = garmentIndexPaths.find((path) => existsSync(path));
describe.skipIf(!garmentIndex)('real garment assets', () => {
  it('has parseable bindings for available templates', () => {
    if (!garmentIndex) return;
    const indexBytes = readFileSync(garmentIndex);
    const index = JSON.parse(
      new TextDecoder().decode(
        indexBytes.buffer.slice(
          indexBytes.byteOffset,
          indexBytes.byteOffset + indexBytes.byteLength,
        ),
      ),
    ) as {
      garments: GarmentTemplateDef[];
    };
    const directory = garmentIndex.slice(0, garmentIndex.lastIndexOf('/'));
    for (const garment of index.garments) {
      if (!garment.binding) continue;
      const path = `${directory}/${garment.binding}`;
      if (!existsSync(path)) continue;
      const bytes = readFileSync(path);
      expect(
        parseGarmentBinding(
          bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
        ).count,
      ).toBeGreaterThan(0);
    }
  });
});
