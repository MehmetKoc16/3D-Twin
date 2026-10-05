import { Group, Mesh, MeshStandardMaterial, SkinnedMesh, type Object3D, type Skeleton } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { TwinFormatError } from './twinDef';

export interface TwinAccessoryDef {
  id: 'glasses';
  bone: 'head';
  /** A complete, head-local GLB inside the bundle BIN, not a URL or a body mesh index. */
  mesh: { bufferView: number };
  params: Record<string, unknown>;
}

export interface AccessoryParser {
  json: { asset?: { extras?: { dtTwin?: unknown } }; bufferViews?: unknown[] };
  getDependency(type: 'bufferView', index: number): Promise<ArrayBuffer>;
}

const record = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

export const useTwinAccessoryStore = create<{ glasses: boolean; setGlasses(on: boolean): void }>()(
  persist((set) => ({ glasses: true, setGlasses: (glasses) => set({ glasses }) }), {
    name: 'dt:twin:accessories',
    storage: createJSONStorage(() => localStorage),
    partialize: ({ glasses }) => ({ glasses }),
  }),
);

/** Legacy bundles have no accessories. Fail closed on malformed or ambiguous references. */
export function parseAccessoryDefs(raw: unknown, viewCount: number): TwinAccessoryDef[] {
  if (!record(raw) || raw.accessories === undefined) return [];
  if (!Array.isArray(raw.accessories) || raw.accessories.length > 1)
    throw new TwinFormatError('json', 'invalid dtTwin.accessories');
  return raw.accessories.map((entry: unknown) => {
    if (
      !record(entry) ||
      entry.id !== 'glasses' ||
      entry.bone !== 'head' ||
      !record(entry.mesh) ||
      !Number.isSafeInteger(entry.mesh.bufferView) ||
      (entry.mesh.bufferView as number) < 0 ||
      (entry.mesh.bufferView as number) >= viewCount ||
      !record(entry.params)
    )
      throw new TwinFormatError('json', 'invalid head accessory reference');
    return {
      id: 'glasses',
      bone: 'head',
      mesh: { bufferView: entry.mesh.bufferView as number },
      params: entry.params,
    };
  });
}

function glbJson(buffer: ArrayBuffer): Record<string, unknown> {
  const view = new DataView(buffer);
  if (
    buffer.byteLength < 20 ||
    view.getUint32(0, true) !== 0x46546c67 ||
    view.getUint32(4, true) !== 2 ||
    view.getUint32(8, true) !== buffer.byteLength ||
    view.getUint32(16, true) !== 0x4e4f534a ||
    view.getUint32(12, true) > buffer.byteLength - 20
  )
    throw new TwinFormatError('glb', 'invalid accessory GLB');
  const raw: unknown = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 20, view.getUint32(12, true))),
  );
  if (!record(raw)) throw new TwinFormatError('glb', 'invalid accessory JSON');
  return raw;
}

/** Used by the panel without another GLTF parse or changes to the persisted twin pack. */
export function bundledAccessoryDefs(buffer: ArrayBuffer): TwinAccessoryDef[] {
  const json = glbJson(buffer);
  const asset = json.asset;
  const extras = record(asset) ? asset.extras : undefined;
  return parseAccessoryDefs(
    record(extras) ? extras.dtTwin : undefined,
    Array.isArray(json.bufferViews) ? json.bufferViews.length : 0,
  );
}

export function disposeAccessory(root: Object3D): void {
  root.removeFromParent();
  root.traverse((child) => {
    if (!(child instanceof Mesh)) return;
    child.geometry.dispose();
    for (const material of Array.isArray(child.material) ? child.material : [child.material])
      material.dispose();
  });
}

/** No nested GLB is parsed until self-contained resources and head-local transforms have been checked. */
export async function loadAccessory(buffer: ArrayBuffer): Promise<Group> {
  const json = glbJson(buffer);
  const asset = json.asset;
  const extras = record(asset) ? asset.extras : undefined;
  const metadata = record(extras) ? extras.dtAccessory : undefined;
  if (
    !record(metadata) ||
    metadata.id !== 'glasses' ||
    metadata.bone !== 'head' ||
    metadata.coordinateSpace !== 'head-local' ||
    !record(metadata.params)
  )
    throw new TwinFormatError('glb', 'missing head-local accessory metadata');
  for (const list of [json.buffers, json.images]) {
    if (Array.isArray(list) && list.some((item: unknown) => record(item) && 'uri' in item))
      throw new TwinFormatError('glb', 'accessory resources must be embedded');
  }
  if (json.extensionsRequired !== undefined)
    throw new TwinFormatError('glb', 'accessory decoders are unsupported');
  if (Array.isArray(json.images) && json.images.length)
    throw new TwinFormatError('glb', 'accessory textures are unsupported');
  const loaded = await new GLTFLoader().parseAsync(buffer.slice(0), '');
  const group = new Group();
  group.name = 'twin:glasses';
  const meshes: SkinnedMesh[] = [];
  try {
    loaded.scene.traverse((child) => {
      if (child instanceof SkinnedMesh) meshes.push(child);
      if (
        child.position.lengthSq() !== 0 ||
        child.quaternion.x !== 0 ||
        child.quaternion.y !== 0 ||
        child.quaternion.z !== 0 ||
        child.quaternion.w !== 1 ||
        child.scale.x !== 1 ||
        child.scale.y !== 1 ||
        child.scale.z !== 1
      )
        throw new TwinFormatError('glb', 'accessory nodes must have identity transforms');
    });
    // GLTFLoader may split one mesh into separate objects for frame/lens primitives.
    if (!meshes.length) throw new TwinFormatError('noSkin', 'accessory has no rigid skin');
    for (const source of meshes) {
      if (source.skeleton.bones.length !== 1 || source.skeleton.bones[0]?.name !== 'head')
        throw new TwinFormatError('bones', 'accessory must bind only to head');
      const inverse = source.skeleton.boneInverses[0];
      if (!inverse || !inverse.equals(source.bindMatrix.clone().identity()))
        throw new TwinFormatError('rest', 'accessory inverse bind must be identity');
      const geometry = source.geometry;
      const positions = geometry.getAttribute('position');
      const joints = geometry.getAttribute('skinIndex');
      const weights = geometry.getAttribute('skinWeight');
      const indices = geometry.getIndex();
      if (!positions || !joints || !weights || !indices || indices.count % 3)
        throw new TwinFormatError('glb', 'incomplete accessory geometry');
      if (weights.count !== positions.count || joints.count !== positions.count)
        throw new TwinFormatError('glb', 'accessory attribute count mismatch');
      for (let i = 0; i < positions.count; i++) {
        if (
          ![positions.getX(i), positions.getY(i), positions.getZ(i)].every(Number.isFinite) ||
          joints.getX(i) !== 0 ||
          joints.getY(i) !== 0 ||
          joints.getZ(i) !== 0 ||
          joints.getW(i) !== 0 ||
          weights.getX(i) !== 1 ||
          weights.getY(i) !== 0 ||
          weights.getZ(i) !== 0 ||
          weights.getW(i) !== 0
        )
          throw new TwinFormatError('glb', 'invalid rigid head accessory weights');
      }
      for (let i = 0; i < indices.count; i++) {
        if (
          !Number.isInteger(indices.getX(i)) ||
          indices.getX(i) < 0 ||
          indices.getX(i) >= positions.count
        )
          throw new TwinFormatError('glb', 'invalid accessory indices');
      }
      const mesh = new Mesh(geometry, source.material);
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const material of materials) {
        if (material.transparent) material.depthWrite = false;
        if (material instanceof MeshStandardMaterial) {
          material.envMapIntensity = material.transparent ? 0.8 : 0.65;
          if (material.metalness > 0.5) material.roughness = 0.4;
        }
      }
      mesh.castShadow = materials.every((material) => !material.transparent);
      mesh.frustumCulled = false;
      group.add(mesh);
    }
    return group;
  } catch (error) {
    disposeAccessory(loaded.scene);
    throw error;
  }
}

export async function loadTwinAccessories(parser: AccessoryParser): Promise<Group[]> {
  const defs = parseAccessoryDefs(
    parser.json.asset?.extras?.dtTwin,
    parser.json.bufferViews?.length ?? 0,
  );
  const groups: Group[] = [];
  try {
    for (const def of defs)
      groups.push(
        await loadAccessory(await parser.getDependency('bufferView', def.mesh.bufferView)),
      );
    return groups;
  } catch (error) {
    groups.forEach(disposeAccessory);
    throw error;
  }
}

/** A head child automatically follows both skeleton rest rebuilding and every pose update. */
export class TwinAccessories {
  private readonly unsubscribe: () => void;
  private ready = false;
  constructor(
    private readonly groups: readonly Group[],
    skeleton: Skeleton,
  ) {
    const head = skeleton.bones.find((bone) => bone.name === 'head');
    if (groups.length && !head) throw new TwinFormatError('bones', 'shared skeleton has no head');
    for (const group of groups) {
      group.visible = false;
      head?.add(group);
    }
    this.unsubscribe = useTwinAccessoryStore.subscribe(() => this.refresh());
  }
  show(): void {
    this.ready = true;
    this.refresh();
  }
  private refresh(): void {
    for (const group of this.groups)
      group.visible = this.ready && useTwinAccessoryStore.getState().glasses;
  }
  get visible(): boolean {
    return this.groups.some((group) => group.visible);
  }
  dispose(): void {
    this.unsubscribe();
    for (const group of this.groups) group.removeFromParent();
  }
}
