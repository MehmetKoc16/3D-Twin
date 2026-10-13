import { describe, expect, it } from 'vitest';
import { garmentSlot, type GarmentTemplateDef } from '../garmentContracts';
import { garmentLayerOrder, garmentNormalOffset, offsetAlongNormals } from './layers';

const top: GarmentTemplateDef = {
  id: 'test',
  kind: 'tshirt',
  category: 'top',
  label: { tr: 'Test', en: 'Test' },
  license: 'CC0-1.0',
  mesh: 'test.glb',
  layer: 2,
  baseColor: '#888888',
  nativeMeasures: {},
  defaultEase: {},
};

describe('garment layer ordering', () => {
  it('defaults only high-layer tops to outer and respects explicit slots', () => {
    expect(garmentSlot(top)).toBe('top');
    expect(garmentSlot({ ...top, layer: 3 })).toBe('outer');
    expect(garmentSlot({ ...top, layer: 5, category: 'bottom' })).toBe('bottom');
    expect(garmentSlot({ ...top, layer: 4, slot: 'top' })).toBe('top');
    expect(garmentSlot({ ...top, slot: 'outer' })).toBe('outer');
  });

  it('places outer normal offsets above inner garments and gives zero offset alone', () => {
    expect(garmentNormalOffset(top)).toBe(0);
    expect(garmentNormalOffset(top, [])).toBe(0);
    expect(garmentNormalOffset({ ...top, layer: 3, slot: 'outer' })).toBe(0);
    expect(garmentNormalOffset({ ...top, category: 'shoes' })).toBe(0);
    const inner = 0.001;
    const outer = { ...top, layer: 1, slot: 'outer' as const };
    expect(garmentNormalOffset(outer, [inner, 0.001])).toBeCloseTo(inner + 0.002);
    expect(garmentLayerOrder({ ...top, layer: 3 })).toBeGreaterThan(garmentLayerOrder(top));
  });

  it('expands along surface normals without altering tangent coordinates', () => {
    const positions = new Float32Array([1, 2, 3, 4, 5, 6]);
    offsetAlongNormals(positions, [0, 0, 1, 1, 0, 0], 0.004);
    expect(positions[2]).toBeCloseTo(3.004);
    expect(positions[3]).toBeCloseTo(4.004);
    expect(positions[0]).toBe(1);
    expect(positions[1]).toBe(2);
    expect(positions[4]).toBe(5);
    expect(positions[5]).toBe(6);
  });

  it('does not split a thin two-sided flap', () => {
    // Front face at z = 0.10 with normal [0, 0, 1]; back face at z = 0.099 with normal [0, 0, -1]
    const positions = new Float32Array([0, 1, 0.1, 0, 1, 0.099]);
    const normals = [0, 0, 1, 0, 0, -1];
    offsetAlongNormals(positions, normals, 0.003);
    expect(positions[2]! - positions[5]!).toBeCloseTo(0.001);
  });

  it('skips offset for vertices with degenerate normals', () => {
    const positions = new Float32Array([1, 2, 3]);
    const normals = [0, 0, 0];
    offsetAlongNormals(positions, normals, 0.004);
    expect(positions[0]).toBe(1);
    expect(positions[1]).toBe(2);
    expect(positions[2]).toBe(3);
  });
});
