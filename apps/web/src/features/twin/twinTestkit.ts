import {
  Bone,
  BufferAttribute,
  BufferGeometry,
  Group,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Skeleton,
  SkinnedMesh,
} from 'three';
import type { SolveEnvelope } from '../../workers/avatarClient';
import type { AvatarAssets } from '../avatar/avatarAssets';
import type { TwinModel } from './twinModel';

/**
 * A tiny fake avatar for the twin tests: the body is a strip of 4 quads (10 vertices, 8 triangles), the twin a strip of
 * 9 vertices whose triangles map onto it, three bones as in the real rig (Root, pelvis, spine_01).
 */

export const boneNames = ['Root', 'pelvis', 'spine_01'];
export const heads = [0, 0, 0, 0, 1, 0, 0, 1.2, 0];
// prettier-ignore
export const bodyIndex = Uint32Array.from([
  0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 8, 9, 3, 9, 7,
]);
export const bodyPositions = new Float32Array(10 * 3); // only the vertex count matters to the rig
export const mapping = Uint32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 9]);
export const twinIndex = Uint32Array.from([0, 1, 5, 1, 2, 6, 2, 3, 7, 4, 5, 6, 3, 7, 8]);

export function fakeAssets(client: unknown = {}): AvatarAssets {
  const bones = boneNames.map((name) => Object.assign(new Bone(), { name }));
  bones[0]!.position.set(0, 0, 0);
  bones[1]!.position.set(0, 1, 0);
  bones[2]!.position.set(0, 0.2, 0);
  bones[0]!.add(bones[1]!);
  bones[1]!.add(bones[2]!);
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(bodyPositions, 3));
  const original = new BufferAttribute(bodyIndex, 1);
  geometry.setIndex(original);
  const scene = new Group();
  const mesh = new SkinnedMesh(geometry, new MeshBasicMaterial());
  scene.add(bones[0]!);
  scene.add(mesh);
  scene.updateMatrixWorld(true);
  const skeleton = new Skeleton(bones);
  mesh.bind(skeleton, mesh.matrixWorld);
  return {
    scene,
    mesh,
    skeleton,
    rigIndexOfBone: Int32Array.from([0, 1, 2]),
    client,
  } as unknown as AvatarAssets;
}

export function fakeModel(shift = 0): TwinModel {
  const n = 9;
  const position = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) position.set([(i % 5) * 0.1, 1 + Math.floor(i / 5) * 0.1, 0], i * 3);
  const twinHeads = Float32Array.from(heads.map((v, i) => (i % 3 === 1 ? v - shift : v)));
  return {
    vertexCount: n,
    position,
    normal: new Float32Array(n * 3),
    uv: null,
    index: twinIndex,
    skinIndex: new Uint16Array(n * 4),
    skinWeight: Float32Array.from({ length: n * 4 }, (_, i) => (i % 4 === 0 ? 1 : 0)),
    heads: twinHeads,
    remap: Int32Array.from([0, 1, 2]),
    material: new MeshStandardMaterial(),
    dispose: () => undefined,
  };
}

export function envelope(fixed: boolean, headShift = 0): SolveEnvelope {
  const joints = new Float32Array(boneNames.length * 6);
  boneNames.forEach((_, i) =>
    joints.set([heads[i * 3]!, heads[i * 3 + 1]! + headShift, heads[i * 3 + 2]!], i * 6),
  );
  return {
    result: {
      positions: new Float32Array(bodyPositions),
      joints,
      groundOffsetY: 0,
      achievedCm: {},
      residualsCm: {},
      unreachable: [],
      estimatedMassKg: 70,
      solveMs: 1,
      ...(fixed ? { fixedShape: true as const } : {}),
    },
    params: {} as SolveEnvelope['params'],
    roundTripMs: 1,
  };
}
