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

  it('places outer normal offsets above the top and bottom even with unusual layer metadata', () => {
    const inner = garmentNormalOffset({ ...top, layer: 9, slot: 'top' });
    const outer = { ...top, layer: 1, slot: 'outer' as const };
    expect(garmentNormalOffset(outer, [inner, 0.001])).toBeCloseTo(inner + 0.002);
    expect(garmentLayerOrder({ ...top, layer: 3 })).toBeGreaterThan(garmentLayerOrder(top));
    expect(garmentNormalOffset({ ...top, category: 'shoes' })).toBe(0);
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
});
