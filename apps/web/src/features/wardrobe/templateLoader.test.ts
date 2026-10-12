import { afterEach, describe, expect, it, vi } from 'vitest';
import { BufferGeometry, Float32BufferAttribute, Group, Mesh, MeshStandardMaterial } from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { loadRigidAttachments, loadTemplateRuntime } from './templateLoader';
import {
  shoeBinding,
  shoeBodySkin,
  shoeDef,
  shoeIndex,
  shoeNeutral,
  shoeSidecar,
} from './rigid.testkit';

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function bindingBuffer(): ArrayBuffer {
  const binding = shoeBinding();
  const buffer = new ArrayBuffer(binding.count * 36);
  const view = new DataView(buffer);
  for (let v = 0; v < binding.count; v++)
    for (let k = 0; k < 3; k++) {
      view.setUint32(v * 36 + k * 4, binding.indices[v * 3 + k]!, true);
      view.setFloat32(v * 36 + 12 + k * 4, binding.weights[v * 3 + k]!, true);
      view.setFloat32(v * 36 + 24 + k * 4, 0, true);
    }
  return buffer;
}

function mockMesh() {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(shoeNeutral, 3));
  geometry.setIndex(Array.from(shoeIndex));
  const material = new MeshStandardMaterial();
  const scene = new Group();
  scene.add(new Mesh(geometry, material));
  vi.spyOn(GLTFLoader.prototype, 'loadAsync').mockResolvedValue({ scene } as unknown as GLTF);
  return () => {
    geometry.dispose();
    material.dispose();
  };
}

describe('shoe sidecar loading', () => {
  it('loads the existing importer filename without a catalogue reference', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(shoeSidecar)));
    vi.stubGlobal('fetch', fetchMock);
    expect(await loadRigidAttachments(shoeDef, '/garments/')).toEqual({ data: shoeSidecar });
    expect(fetchMock).toHaveBeenCalledWith('/garments/synthetic-shoe.rigid.json');
  });
  it('honours an explicit path', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(shoeSidecar)));
    vi.stubGlobal('fetch', fetchMock);
    await loadRigidAttachments(
      { ...shoeDef, rigidAttachments: 'attachments/custom.json' },
      '/garments/',
    );
    expect(fetchMock).toHaveBeenCalledWith('/garments/attachments/custom.json');
  });
  it('allows only implicit 404 fallback and never probes ordinary clothing', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 404 }));
    vi.stubGlobal('fetch', fetchMock);
    expect(await loadRigidAttachments(shoeDef, '/')).toBeNull();
    await expect(
      loadRigidAttachments({ ...shoeDef, rigidAttachments: 'required.json' }, '/'),
    ).rejects.toThrow(/404/);
    fetchMock.mockClear();
    expect(await loadRigidAttachments({ ...shoeDef, category: 'top' }, '/')).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
    await expect(
      loadRigidAttachments({ ...shoeDef, category: 'top', rigidAttachments: 'bad.json' }, '/'),
    ).rejects.toThrow(/require shoes/);
  });
  it('rejects failed requests, invalid paths and invalid JSON', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(null, { status: 500 })));
    await expect(loadRigidAttachments(shoeDef, '/')).rejects.toThrow(/500/);
    await expect(loadRigidAttachments({ ...shoeDef, rigidAttachments: '' }, '/')).rejects.toThrow(
      /invalid/,
    );
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>missing file</html>')));
    await expect(loadRigidAttachments(shoeDef, '/')).rejects.toThrow();
  });
  it('validates sidecar vertex indices against the mesh and carries neutral geometry into the runtime', async () => {
    const dispose = mockMesh();
    try {
      const value = structuredClone(shoeSidecar);
      vi.stubGlobal(
        'fetch',
        vi.fn(async (url: string) =>
          url.endsWith('.json')
            ? new Response(JSON.stringify(value))
            : new Response(bindingBuffer()),
        ),
      );
      const runtime = await loadTemplateRuntime(shoeDef, shoeBodySkin(), '/');
      expect(runtime.rigidAttachments?.attachments).toEqual(shoeSidecar.attachments);
      expect(runtime.neutralPositions).toEqual(shoeNeutral);
      value.attachments[0]!.vertices[0] = 999;
      await expect(loadTemplateRuntime(shoeDef, shoeBodySkin(), '/')).rejects.toThrow(
        /out-of-range/,
      );
      vi.stubGlobal(
        'fetch',
        vi.fn(async (url: string) =>
          url.endsWith('.json') ? new Response('null') : new Response(bindingBuffer()),
        ),
      );
      await expect(loadTemplateRuntime(shoeDef, shoeBodySkin(), '/')).rejects.toThrow(
        /invalid sidecar/,
      );
    } finally {
      dispose();
    }
  });
  it('accepts CC-BY-3.0 exactly and rejects missing attribution before fetching', async () => {
    const dispose = mockMesh();
    try {
      const fetchMock = vi.fn(async () => new Response(bindingBuffer()));
      vi.stubGlobal('fetch', fetchMock);
      const def = {
        ...shoeDef,
        category: 'top' as const,
        kind: 'longsleeve' as const,
        license: 'CC-BY-3.0' as const,
        attribution: 'Synthetic jacket by Author, source URL, CC-BY-3.0. Converted.',
      };
      const runtime = await loadTemplateRuntime(def, shoeBodySkin(), '/');
      expect(runtime.def.license).toBe('CC-BY-3.0');
      expect(runtime.def.attribution).toBe(def.attribution);
      fetchMock.mockClear();
      await expect(
        loadTemplateRuntime({ ...def, attribution: '' }, shoeBodySkin(), '/'),
      ).rejects.toThrow(/attribution/);
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      dispose();
    }
  });
});
