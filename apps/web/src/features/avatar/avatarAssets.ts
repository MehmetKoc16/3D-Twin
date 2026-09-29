import {
  Bone,
  BufferAttribute,
  DynamicDrawUsage,
  Float32BufferAttribute,
  Group,
  MeshPhysicalMaterial,
  SkinnedMesh,
  type Object3D,
  type Skeleton,
} from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { AvatarClient, type SolveEnvelope } from '../../workers/avatarClient';
import { useAvatarLoadStore } from '../../store/avatarLoadStore';
import { buildWeldMap, type WeldMap } from './meshMath';

/** Share of the progress bar taken by base.glb (about 1 of 17 MB); the rest is the morph download in the worker. */
const GLB_SHARE = 0.06;

export interface AvatarAssets {
  scene: Group;
  mesh: SkinnedMesh;
  skeleton: Skeleton;
  material: MeshPhysicalMaterial;
  weld: WeldMap;
  indices: Uint32Array;
  /** Bone index in the skeleton -> parent bone index (-1 for roots). */
  parents: Int32Array;
  /** Bone index in the skeleton -> index in rig.json order (the layout of `joints` in solve results). */
  rigIndexOfBone: Int32Array;
  normalScratch: Float32Array;
  client: AvatarClient;
  /** Subscribes to solve results; returns an unsubscribe function. */
  onSolve: (listener: (envelope: SolveEnvelope) => void) => () => void;
}

let loading: Promise<AvatarAssets> | null = null;

function findSkinnedMesh(root: Object3D): SkinnedMesh {
  let found: SkinnedMesh | null = null;
  root.traverse((o) => {
    if (!found && o instanceof SkinnedMesh) found = o;
  });
  if (!found) throw new Error('base.glb contains no skinned mesh');
  return found;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Loads base.glb and the solver worker once (survives React StrictMode remounts). */
export function loadAvatarAssets(): Promise<AvatarAssets> {
  if (loading) return loading;
  const store = useAvatarLoadStore.getState();
  loading = (async () => {
    const baseUrl = `${import.meta.env.BASE_URL}assets/body/`;
    const listeners = new Set<(envelope: SolveEnvelope) => void>();
    const gltf = await new GLTFLoader().loadAsync(`${baseUrl}base.glb`, (event) => {
      if (event.total > 0) store.setProgress(GLB_SHARE * Math.min(1, event.loaded / event.total));
    });
    const scene = gltf.scene;
    const mesh = findSkinnedMesh(scene);
    const geometry = mesh.geometry;
    const position = geometry.getAttribute('position') as BufferAttribute;
    const index = geometry.getIndex();
    if (!index) throw new Error('base.glb mesh has no index buffer');
    const rest = Float32Array.from(position.array as ArrayLike<number>);
    const indices = Uint32Array.from(index.array as ArrayLike<number>);

    const weld = buildWeldMap(rest);
    // Positions and normals are rewritten on every solve
    position.setUsage(DynamicDrawUsage);
    geometry.setAttribute(
      'normal',
      new Float32BufferAttribute(new Float32Array(rest.length), 3).setUsage(DynamicDrawUsage),
    );

    const material = new MeshPhysicalMaterial({ color: '#e0ac8b', roughness: 0.62, metalness: 0 });
    mesh.material = material;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.frustumCulled = false; // bounds of a posed skinned mesh are not tracked

    const skeleton = mesh.skeleton;
    const boneIndex = new Map(skeleton.bones.map((b, i) => [b.name, i]));
    const parents = new Int32Array(skeleton.bones.length).fill(-1);
    skeleton.bones.forEach((bone, i) => {
      if (bone.parent instanceof Bone) parents[i] = boneIndex.get(bone.parent.name) ?? -1;
    });

    const client = new AvatarClient(
      (envelope) => listeners.forEach((l) => l(envelope)),
      (error) => useAvatarLoadStore.getState().setError(errorMessage(error)),
    );
    const init = await client.init(
      { baseUrl, positions: rest.slice(), indices: indices.slice() },
      (f) => store.setProgress(GLB_SHARE + (1 - GLB_SHARE) * f),
    );
    const rigIndex = new Map(init.boneNames.map((n, i) => [n, i]));
    const rigIndexOfBone = Int32Array.from(skeleton.bones, (b) => {
      const i = rigIndex.get(b.name);
      if (i === undefined) throw new Error(`bone ${b.name} of base.glb is missing in rig.json`);
      return i;
    });

    return {
      scene,
      mesh,
      skeleton,
      material,
      weld,
      indices,
      parents,
      rigIndexOfBone,
      normalScratch: new Float32Array(weld.groupCount * 3),
      client,
      onSolve: (listener: (envelope: SolveEnvelope) => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    };
  })().catch((error: unknown) => {
    useAvatarLoadStore.getState().setError(errorMessage(error));
    throw error;
  });
  return loading;
}
