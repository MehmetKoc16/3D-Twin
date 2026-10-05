import { afterEach, expect, it, vi } from 'vitest';
import { useEffect } from 'react';
import { SoftShadows } from '@react-three/drei/core/softShadows.js';
import {
  DataTexture,
  Mesh,
  MeshStandardMaterial,
  NoColorSpace,
  ShaderLib,
  type WebGLRenderer,
} from 'three';
import { twinMaterial } from '../twin/twinModel';

const runtime = vi.hoisted(() => ({
  gl: {
    properties: { remove: vi.fn() },
    shadowMap: { type: 1 },
    info: { programs: [] },
    compile: vi.fn(),
  },
  scene: { traverse: vi.fn() },
  camera: {},
}));

vi.mock('@react-three/fiber', () => ({
  useThree: (select: (state: typeof runtime) => unknown) => select(runtime),
}));
vi.mock('react', async (importOriginal) => ({
  ...(await importOriginal<typeof import('react')>()),
  useEffect: vi.fn(),
}));

afterEach(() => vi.clearAllMocks());

it('retains twin normal details through the actual SoftShadows resets used by High and Performance', () => {
  const normalMap = new DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1);
  const source = new MeshStandardMaterial({ normalMap });
  source.normalScale.set(0.35, -0.6);
  const material = twinMaterial(source);
  const mesh = new Mesh(undefined, material);
  const normalDisposed = vi.fn();
  normalMap.addEventListener('dispose', normalDisposed);
  runtime.scene.traverse.mockImplementation((visit: (mesh: Mesh) => void) => visit(mesh));
  let cleanup: (() => void) | undefined;
  vi.mocked(useEffect).mockImplementation((effect) => {
    cleanup = effect() || undefined;
  });
  try {
    for (const samples of [12, 6, 12]) {
      // Run the installed drei effect, including material.dispose / needsUpdate and shader cache removal.
      SoftShadows({ size: 35, samples });
      expect(runtime.gl.compile).toHaveBeenCalled();
      expect(runtime.gl.properties.remove).toHaveBeenCalledWith(material);
      expect(material.normalMap).toBe(normalMap);
      expect(material.normalScale.toArray()).toEqual([0.35, -0.6]);
      expect(normalMap.colorSpace).toBe(NoColorSpace);
      const shader = { ...ShaderLib.physical, uniforms: { ...ShaderLib.physical.uniforms } };
      material.onBeforeCompile(
        shader as Parameters<typeof material.onBeforeCompile>[0],
        {} as WebGLRenderer,
      );
      expect(shader.fragmentShader).toContain('#include <normal_fragment_maps>');
      expect(shader.fragmentShader).toContain('uniform float dtSkinScatter;');
      cleanup?.();
      cleanup = undefined;
    }
    expect(normalDisposed).not.toHaveBeenCalled();
  } finally {
    cleanup?.();
    material.dispose();
    source.dispose();
    mesh.geometry.dispose();
    normalMap.dispose();
  }
});
