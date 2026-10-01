import { describe, expect, it } from 'vitest';
import {
  Bone,
  BufferAttribute,
  BufferGeometry,
  Group,
  Matrix4,
  MeshBasicMaterial,
  Quaternion,
  Skeleton,
  SkinnedMesh,
  Vector3,
} from 'three';
import { restLocalPositions } from '../avatar/meshMath';
import {
  headsFromBoneInverses,
  matchBones,
  normalizeSkinWeights,
  remapSkinIndices,
  restAlignment,
  translatePositions,
} from './twinBinding';
import { TwinFormatError } from './twinDef';

const appNames = ['Root', 'pelvis', 'spine_01'];
const appParents = [-1, 0, 1];
const appHeads = [0, 0, 0, 0, 1, 0, 0, 1.2, 0];
// the twin lists its bones in another order and lives in a frame shifted by `frameShift` (its feet are on y = 0 by
// its own ground rule)
const twinNames = ['spine_01', 'Root', 'pelvis'];
const frameShift: [number, number, number] = [0.02, -0.011, 0.005];
const twinHeads = [0, 1.2, 0, 0, 0, 0, 0, 1, 0].map((v, i) => v - frameShift[i % 3]!);

function translation(x: number, y: number, z: number): Matrix4 {
  return new Matrix4().makeTranslation(x, y, z);
}

describe('rest heads from inverse bind matrices', () => {
  it('reads -translation of pure translations', () => {
    const heads = headsFromBoneInverses([translation(-1, -2, -3), translation(0, 0, 0.5)]);
    expect([...heads]).toEqual([1, 2, 3, -0, -0, -0.5]);
  });

  it('rejects rotated or scaled bind matrices (the pipeline contract is world-aligned bones)', () => {
    const rotated = new Matrix4().makeRotationZ(0.3);
    expect(() => headsFromBoneInverses([rotated])).toThrow(TwinFormatError);
    const scaled = new Matrix4().makeScale(1.1, 1, 1);
    expect(() => headsFromBoneInverses([scaled])).toThrow(/pure translation/);
  });
});

describe('bone matching', () => {
  it('maps the twin bones onto the avatar bone order by name', () => {
    expect([...matchBones(twinNames, appNames, twinNames)]).toEqual([2, 0, 1]);
  });

  it('rejects foreign skeletons and a boneOrder that differs from the glb', () => {
    expect(() => matchBones(['Root', 'pelvis'], appNames)).toThrow(/2 bones/);
    expect(() => matchBones(['Root', 'pelvis', 'spine_09'], appNames)).toThrow(/spine_09/);
    expect(() => matchBones(['Root', 'Root', 'pelvis'], appNames)).toThrow(TwinFormatError);
    expect(() => matchBones(twinNames, appNames, appNames)).toThrow(/boneOrder/);
  });

  it('remaps skin indices and rejects indices outside the skeleton', () => {
    const remap = matchBones(twinNames, appNames);
    expect([...remapSkinIndices([0, 1, 2, 0], remap)]).toEqual([2, 0, 1, 2]);
    expect(() => remapSkinIndices([3], remap)).toThrow(TwinFormatError);
  });

  it('renormalises skin weights and rejects unusable ones', () => {
    const w = Float32Array.from([2, 2, 0, 0, 0.25, 0.25, 0.25, 0.25]);
    normalizeSkinWeights(w);
    expect([...w]).toEqual([0.5, 0.5, 0, 0, 0.25, 0.25, 0.25, 0.25]);
    expect(() => normalizeSkinWeights(Float32Array.from([0, 0, 0, 0]))).toThrow(/no skin weight/);
    expect(() => normalizeSkinWeights(Float32Array.from([1, -1, 1, 0]))).toThrow(/invalid/);
  });
});

describe('rest alignment', () => {
  const remap = matchBones(twinNames, appNames);

  it('finds the frame translation between the two rests', () => {
    const { offset, maxResidual } = restAlignment(twinHeads, remap, appHeads);
    expect(offset[0]).toBeCloseTo(frameShift[0], 12);
    expect(offset[1]).toBeCloseTo(frameShift[1], 12);
    expect(offset[2]).toBeCloseTo(frameShift[2], 12);
    expect(maxResidual).toBeLessThan(1e-12);
  });

  it('reports what a translation cannot explain (files that do not belong together)', () => {
    const skewed = [...twinHeads];
    skewed[1] = skewed[1]! + 0.03; // one bone is 3 cm off
    expect(restAlignment(skewed, remap, appHeads).maxResidual).toBeGreaterThan(0.015);
  });

  it('translatePositions writes base + offset', () => {
    const out = new Float32Array(6);
    translatePositions([1, 2, 3, 4, 5, 6], [0.5, -1, 0], out);
    expect([...out]).toEqual([1.5, 1, 3, 4.5, 4, 6]);
  });
});

describe('twin mesh bound to the avatar skeleton (three.js skinning)', () => {
  // twin vertices in the twin's frame: on the pelvis, on the spine, half and half
  const twinPositions = [0.2, 0.9, 0, 0, 1.3, 0.1, 0.1, 1.1, 0];
  // skin in the twin's own joint order: pelvis = 2, spine_01 = 0
  const twinSkinIndex = [2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0];
  const twinSkinWeight = [1, 0, 0, 0, 1, 0, 0, 0, 0.5, 0.5, 0, 0];

  function build() {
    const remap = matchBones(twinNames, appNames);
    const { offset } = restAlignment(twinHeads, remap, appHeads);
    const positions = new Float32Array(twinPositions.length);
    translatePositions(twinPositions, offset, positions);
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(positions, 3));
    geometry.setAttribute(
      'skinIndex',
      new BufferAttribute(remapSkinIndices(twinSkinIndex, remap), 4),
    );
    geometry.setAttribute('skinWeight', new BufferAttribute(Float32Array.from(twinSkinWeight), 4));
    // the avatar's skeleton, rebuilt exactly like rebuildRestSkeleton does
    const bones = appNames.map((name) => Object.assign(new Bone(), { name }));
    const local = restLocalPositions(appParents, appHeads);
    bones.forEach((bone, i) => {
      bone.position.set(local[i * 3]!, local[i * 3 + 1]!, local[i * 3 + 2]!);
      if (appParents[i]! >= 0) bones[appParents[i]!]!.add(bone);
    });
    const scene = new Group();
    scene.add(bones[0]!);
    const mesh = new SkinnedMesh(geometry, new MeshBasicMaterial());
    scene.add(mesh);
    scene.updateMatrixWorld(true);
    const skeleton = new Skeleton(bones);
    mesh.bind(skeleton, mesh.matrixWorld);
    return { mesh, bones, scene, offset };
  }

  function position(mesh: SkinnedMesh, i: number): Vector3 {
    return mesh.getVertexPosition(i, new Vector3());
  }

  it('at rest every twin vertex sits at its own position moved onto the avatar frame', () => {
    const { mesh, offset } = build();
    for (let i = 0; i < 3; i++) {
      const p = position(mesh, i);
      expect(p.x).toBeCloseTo(twinPositions[i * 3]! + offset[0], 5);
      expect(p.y).toBeCloseTo(twinPositions[i * 3 + 1]! + offset[1], 5);
      expect(p.z).toBeCloseTo(twinPositions[i * 3 + 2]! + offset[2], 5);
    }
  });

  it('a bone rotation of the avatar skeleton turns the twin about the avatar joint', () => {
    const { mesh, bones, scene, offset } = build();
    // the pelvis swings 90 degrees about z; the avatar pelvis head is (0, 1, 0)
    bones[1]!.quaternion.copy(new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), Math.PI / 2));
    scene.updateMatrixWorld(true);
    const rest0 = new Vector3(
      twinPositions[0]! + offset[0],
      twinPositions[1]! + offset[1],
      twinPositions[2]! + offset[2],
    );
    const relative = rest0.clone().sub(new Vector3(0, 1, 0));
    const expected = new Vector3(-relative.y, relative.x, relative.z).add(new Vector3(0, 1, 0)); // Rz(90)
    const p0 = position(mesh, 0);
    expect(p0.distanceTo(expected)).toBeLessThan(1e-5);
    // spine_01 is a child of the pelvis: its vertex follows the pelvis rotation as well
    const rest1 = new Vector3(
      twinPositions[3]! + offset[0],
      twinPositions[4]! + offset[1],
      twinPositions[5]! + offset[2],
    );
    const rel1 = rest1.clone().sub(new Vector3(0, 1, 0));
    expect(
      position(mesh, 1).distanceTo(new Vector3(-rel1.y, rel1.x, rel1.z).add(new Vector3(0, 1, 0))),
    ).toBeLessThan(1e-5);
  });

  it('a blended vertex follows the mean of its two bones', () => {
    const { mesh, bones, scene, offset } = build();
    // only the spine rotates (about its own head (0, 1.2, 0)): vertex 2 is half pelvis (fixed), half spine (rotated)
    bones[2]!.quaternion.copy(new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), Math.PI / 2));
    scene.updateMatrixWorld(true);
    const rest = new Vector3(
      twinPositions[6]! + offset[0],
      twinPositions[7]! + offset[1],
      twinPositions[8]! + offset[2],
    );
    const rel = rest.clone().sub(new Vector3(0, 1.2, 0));
    const spineOnly = new Vector3(-rel.y, rel.x, rel.z).add(new Vector3(0, 1.2, 0));
    const expected = rest.clone().add(spineOnly).multiplyScalar(0.5);
    expect(position(mesh, 2).distanceTo(expected)).toBeLessThan(1e-5);
  });
});
