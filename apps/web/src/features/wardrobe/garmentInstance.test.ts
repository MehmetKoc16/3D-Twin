import { describe, expect, it } from 'vitest';
import { Bone, Matrix4, MeshStandardMaterial, Skeleton } from 'three';
import type { GarmentTemplateDef, StoreItemDef } from '@dt/avatar-core';
import { buildWeldMap } from '../avatar/meshMath';
import { GarmentInstance, recolorFactor } from './garmentInstance';
import type { TemplateRuntime } from './templateLoader';

describe('recolorFactor', () => {
  it('is 1 at the template base colour (the texture is shown unchanged)', () => {
    const [r, g, b] = recolorFactor('#838383', '#838383');
    expect(r).toBeCloseTo(1, 5);
    expect(g).toBeCloseTo(1, 5);
    expect(b).toBeCloseTo(1, 5);
  });

  it('scales the grey texture up to a brighter or more saturated item colour, per linear channel', () => {
    const [r, g, b] = recolorFactor('#ff0000', '#838383');
    expect(r).toBeGreaterThan(1);
    expect(g).toBeLessThan(0.05);
    expect(b).toBeLessThan(0.05);
  });

  it('caps the factor for near-black textures', () => {
    const [r] = recolorFactor('#ffffff', '#101010');
    expect(r).toBeLessThanOrEqual(12);
  });
});

describe('GarmentInstance.hideUnder', () => {
  it('hides triangles flagged in extraTriangleMask and restores full index on null', () => {
    const def: GarmentTemplateDef = {
      id: 'test-shoes',
      category: 'shoes',
      kind: 'sneakers',
      layer: 1,
      label: { tr: 'ayakkabı', en: 'shoes' },
      mesh: 'test.glb',
      license: 'CC0-1.0',
      nativeMeasures: {},
      defaultEase: {},
      baseColor: '#888888',
    };
    const positions = Float32Array.from([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      1, 1, 0,
    ]);
    const indices = Uint32Array.from([0, 1, 2, 1, 3, 2]);
    const runtime: TemplateRuntime = {
      def,
      vertexCount: 4,
      index: indices,
      uv: new Float32Array(8),
      neutralPositions: positions,
      binding: {
        count: 4,
        indices: new Uint32Array(12),
        weights: Float32Array.from({ length: 12 }, (_, i) => (i % 3 === 0 ? 1 : 0)),
        offsets: positions.slice(),
      },
      skinIndices: new Uint16Array(16),
      skinWeights: Float32Array.from({ length: 16 }, (_, i) => (i % 4 === 0 ? 1 : 0)),
      weld: buildWeldMap(positions),
      deleteVerts: new Uint32Array(0),
      rigidAttachments: null,
      material: new MeshStandardMaterial(),
    };
    const item: StoreItemDef = {
      id: 'test-item',
      name: 'test-item',
      templateId: def.id,
      color: def.baseColor,
      sizes: ['M'],
      selectedSize: 'M',
      chart: {},
    };
    const skeleton = new Skeleton([new Bone()]);
    const bindMatrix = new Matrix4();
    const instance = new GarmentInstance(runtime, item, skeleton, bindMatrix);

    // a fresh instance draws its whole index (the draw range starts unbounded)
    expect(instance.mesh.geometry.getIndex()!.count).toBe(6);

    // Hide triangle 0 via extraTriangleMask = [1, 0].
    instance.hideUnder(null, { extraTriangleMask: Uint8Array.from([1, 0]) });
    expect(instance.mesh.geometry.drawRange.count).toBe(3);
    const remainingIndices = Array.from(
      (instance.mesh.geometry.getIndex()!.array as Uint32Array).subarray(0, 3),
    );
    expect(remainingIndices).toEqual([1, 3, 2]);

    // Passing null with no masks restores full draw range and indices.
    instance.hideUnder(null);
    expect(instance.mesh.geometry.drawRange.count).toBe(6);
    const restoredIndices = Array.from(
      (instance.mesh.geometry.getIndex()!.array as Uint32Array).subarray(0, 6),
    );
    expect(restoredIndices).toEqual([0, 1, 2, 1, 3, 2]);

    instance.dispose();
  });
});
