import { describe, expect, it } from 'vitest';
import {
  BackSide,
  DataTexture,
  DoubleSide,
  FrontSide,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  SRGBColorSpace,
} from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { BLEND_AS_MASK_CUTOFF, twinMaterial } from './twinModel';

const texture = () => new DataTexture(new Uint8Array([255, 0, 0, 0]), 1, 1);

/** One-triangle GLB with the given glTF material, embedded buffer (no network, no images). */
function triangleGlb(material: Record<string, unknown>): ArrayBuffer {
  const json = new TextEncoder().encode(
    JSON.stringify({
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [material],
      accessors: [
        {
          bufferView: 0,
          componentType: 5126,
          count: 3,
          type: 'VEC3',
          min: [0, 0, 0],
          max: [1, 1, 0],
        },
      ],
      bufferViews: [{ buffer: 0, byteLength: 36 }],
      buffers: [{ byteLength: 36 }],
    }),
  );
  const padded = new Uint8Array(Math.ceil(json.length / 4) * 4).fill(32);
  padded.set(json);
  const out = new ArrayBuffer(12 + 8 + padded.length + 8 + 36);
  const view = new DataView(out);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, out.byteLength, true);
  view.setUint32(12, padded.length, true);
  view.setUint32(16, 0x4e4f534a, true);
  new Uint8Array(out, 20, padded.length).set(padded);
  view.setUint32(20 + padded.length, 36, true);
  view.setUint32(24 + padded.length, 0x004e4942, true);
  new Float32Array(out, 28 + padded.length, 9).set([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  return out;
}

describe('twinMaterial', () => {
  it('keeps opaque materials exactly as before', () => {
    const map = texture();
    const material = twinMaterial(new MeshStandardMaterial({ map }));
    expect(material.map).toBe(map);
    expect(map.colorSpace).toBe(SRGBColorSpace);
    expect(material.alphaTest).toBe(0);
    expect(material.transparent).toBe(false);
    expect(material.side).toBe(FrontSide);
    expect(material.roughness).toBe(0.88);
    expect(material.metalness).toBe(0);
    expect(twinMaterial(undefined).alphaTest).toBe(0);
    expect(twinMaterial(new MeshBasicMaterial()).side).toBe(FrontSide);
  });

  it('honours MASK: alphaTest is the cutoff and the texture alpha is used', () => {
    const map = texture();
    const material = twinMaterial(new MeshStandardMaterial({ map, alphaTest: 0.3 }));
    expect(material.alphaTest).toBe(0.3);
    expect(material.map).toBe(map);
    expect(material.transparent).toBe(false);
  });

  it('honours doubleSided, and only DoubleSide', () => {
    expect(twinMaterial(new MeshStandardMaterial({ side: DoubleSide })).side).toBe(DoubleSide);
    expect(twinMaterial(new MeshStandardMaterial({ side: BackSide })).side).toBe(FrontSide);
  });

  it('treats BLEND as MASK at 0.5 (no transparent sorting)', () => {
    const material = twinMaterial(new MeshStandardMaterial({ map: texture(), transparent: true }));
    expect(material.alphaTest).toBe(BLEND_AS_MASK_CUTOFF);
    expect(material.transparent).toBe(false);
  });

  it('needs no custom depth material: three applies alphaTest with the live map in the shadow pass', () => {
    const material = twinMaterial(new MeshStandardMaterial({ map: texture(), alphaTest: 0.5 }));
    const mesh = new Mesh(undefined, material);
    expect(mesh.customDepthMaterial).toBeUndefined();
    expect(material.alphaTest).toBeGreaterThan(0);
    expect(material.map).not.toBeNull(); // three's depth variant requires map && alphaTest > 0
  });

  it.each([
    ['MASK', { alphaMode: 'MASK', alphaCutoff: 0.25 }, 0.25],
    ['MASK default cutoff', { alphaMode: 'MASK' }, 0.5],
    ['BLEND', { alphaMode: 'BLEND' }, 0.5],
    ['OPAQUE', {}, 0],
  ])('maps a parsed glTF %s material', async (_name, alpha, cutoff) => {
    const loaded = await new GLTFLoader().parseAsync(
      triangleGlb({ doubleSided: true, ...alpha }),
      '',
    );
    let source: MeshStandardMaterial | undefined;
    loaded.scene.traverse((o) => {
      if (o instanceof Mesh) source = o.material as MeshStandardMaterial;
    });
    const material = twinMaterial(source);
    expect(material.alphaTest).toBe(cutoff);
    expect(material.side).toBe(DoubleSide);
    expect(material.transparent).toBe(false);
  });
});
