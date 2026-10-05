import { afterEach, describe, expect, it, vi } from 'vitest';
import { Bone, Mesh, MeshStandardMaterial, Skeleton, Vector3 } from 'three';
import { loadTwinBundle } from './twinBundle';
import { bundleBytes } from './twinBundleTestkit';
import { accessoryBytes, accessoryParams, appendTestAccessory } from './twinAccessoryTestkit';
import {
  bundledAccessoryDefs,
  disposeAccessory,
  loadAccessory,
  parseAccessoryDefs,
  TwinAccessories,
  useTwinAccessoryStore,
} from './twinAccessories';

vi.hoisted(() => {
  const items = new Map<string, string>();
  vi.stubGlobal('localStorage', {
    getItem: (key: string) => items.get(key) ?? null,
    setItem: (key: string, value: string) => items.set(key, value),
    removeItem: (key: string) => items.delete(key),
  });
});

afterEach(() => {
  useTwinAccessoryStore.getState().setGlasses(true);
  vi.restoreAllMocks();
});

describe('head accessories', () => {
  it('parses the optional embedded GLB contract without changing the mapping', async () => {
    const buffer = appendTestAccessory(bundleBytes());
    expect(bundledAccessoryDefs(buffer)).toEqual([
      { id: 'glasses', bone: 'head', mesh: { bufferView: 1 }, params: accessoryParams },
    ]);
    const bundle = await loadTwinBundle(buffer);
    expect(bundle?.mapping).toEqual(Uint32Array.from([1, 3]));
    expect(bundledAccessoryDefs(bundleBytes())).toEqual([]);
  });

  it.each([
    {},
    { id: 'glasses', bone: 'neck', mesh: { bufferView: 0 }, params: {} },
    { id: 'glasses', bone: 'head', mesh: { bufferView: -1 }, params: {} },
    { id: 'glasses', bone: 'head', mesh: { bufferView: 1 }, params: {} },
    { id: 'glasses', bone: 'head', mesh: { bufferView: 0.5 }, params: {} },
  ])('rejects invalid accessory metadata %#', (entry) => {
    expect(() => parseAccessoryDefs({ accessories: [entry] }, 1)).toThrow(/accessory reference/);
  });

  it('rejects URLs before parsing and rejects nonrigid weights', async () => {
    await expect(loadAccessory(accessoryBytes({ external: true }))).rejects.toThrow(/embedded/);
    await expect(
      loadTwinBundle(appendTestAccessory(bundleBytes(), accessoryBytes({ external: true }))),
    ).rejects.toThrow(/embedded/);
    await expect(loadAccessory(accessoryBytes({ invalidWeights: true }))).rejects.toThrow(
      /weights/,
    );
  });

  it('keeps metal and subtle transparent lens materials', async () => {
    const group = await loadAccessory(accessoryBytes({ lenses: true }));
    expect(group.children).toHaveLength(2);
    const metal = (group.children[0] as Mesh).material as MeshStandardMaterial;
    const lens = (group.children[1] as Mesh).material as MeshStandardMaterial;
    expect(metal.metalness).toBe(1);
    expect(metal.roughness).toBe(0.4);
    expect(metal.envMapIntensity).toBe(0.65);
    expect(lens.envMapIntensity).toBe(0.8);
    expect(lens.transparent).toBe(true);
    expect(lens.depthWrite).toBe(false);
    expect((group.children[1] as Mesh).castShadow).toBe(false);
    expect(lens.opacity).toBeCloseTo(0.035);
    disposeAccessory(group);
  });

  it('follows a shared head pose and rest translation, persists its toggle, and detaches on disposal', async () => {
    const group = await loadAccessory(accessoryBytes());
    const head = new Bone();
    head.name = 'head';
    head.position.set(0, 1.6, 0);
    const skeleton = new Skeleton([head]);
    const rig = new TwinAccessories([group], skeleton);
    expect(group.parent).toBe(head);
    expect(group.visible).toBe(false);
    rig.show();
    expect(group.visible).toBe(true);
    const before = group.localToWorld(new Vector3(0.02, 0.13, 0.11));
    head.rotation.y = Math.PI / 2;
    head.updateMatrixWorld(true);
    const after = group.localToWorld(new Vector3(0.02, 0.13, 0.11));
    expect(after.x).toBeCloseTo(0.11);
    expect(after.y).toBeCloseTo(1.73);
    expect(after.z).toBeCloseTo(-0.02);
    expect(after.distanceTo(before)).toBeGreaterThan(0.1);
    head.position.y += 0.02;
    head.updateMatrixWorld(true);
    expect(group.localToWorld(new Vector3(0, 0.13, 0)).y).toBeCloseTo(1.75);
    useTwinAccessoryStore.getState().setGlasses(false);
    expect(rig.visible).toBe(false);
    const saved = localStorage.getItem('dt:twin:accessories')!;
    const stored = JSON.parse(saved) as {
      state: { glasses: boolean };
    };
    expect(stored.state.glasses).toBe(false);
    useTwinAccessoryStore.setState({ glasses: true });
    localStorage.setItem('dt:twin:accessories', saved);
    await useTwinAccessoryStore.persist.rehydrate();
    expect(group.visible).toBe(false);
    rig.dispose();
    expect(group.parent).toBeNull();
    const material = (group.children[0] as Mesh).material as MeshStandardMaterial;
    const disposed = vi.spyOn(material, 'dispose');
    disposeAccessory(group);
    expect(disposed).toHaveBeenCalledOnce();
  });
});
