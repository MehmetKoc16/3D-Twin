import { Quaternion } from 'three';
import type { SolveEnvelope } from '../../workers/avatarClient';
import type { AvatarAssets } from './avatarAssets';
import { computeWeldedNormals, headsFromJoints, restLocalPositions } from './meshMath';

/**
 * Rebuilds the rest skeleton from new joint heads while keeping the current pose rotations:
 * save quaternions -> identity -> local translations from the heads -> new inverse bind matrices -> restore.
 * Bones are world-aligned (identity rest rotation), so local = head - parentHead and the root's local = its head.
 */
export function rebuildRestSkeleton(assets: AvatarAssets, joints: Float32Array): void {
  const { skeleton, mesh, scene, parents, rigIndexOfBone } = assets;
  const bones = skeleton.bones;
  const saved = bones.map((b) => b.quaternion.clone());
  const local = restLocalPositions(parents, headsFromJoints(joints, rigIndexOfBone));
  const identity = new Quaternion();
  bones.forEach((bone, i) => {
    bone.quaternion.copy(identity);
    bone.position.set(local[i * 3]!, local[i * 3 + 1]!, local[i * 3 + 2]!);
  });
  scene.updateMatrixWorld(true);
  skeleton.calculateInverses();
  mesh.bindMatrix.copy(mesh.matrixWorld);
  mesh.bindMatrixInverse.copy(mesh.matrixWorld).invert();
  bones.forEach((bone, i) => bone.quaternion.copy(saved[i]!));
  scene.updateMatrixWorld(true);
}

/** Writes a solve result into the avatar geometry (positions, welded normals, bounds) and the rest skeleton. */
export function applySolveResult(assets: AvatarAssets, envelope: SolveEnvelope): void {
  const { mesh, weld, indices, normalScratch } = assets;
  const geometry = mesh.geometry;
  const position = geometry.getAttribute('position');
  const normal = geometry.getAttribute('normal');
  const { positions, joints } = envelope.result;
  (position.array as Float32Array).set(positions);
  position.needsUpdate = true;
  computeWeldedNormals(positions, indices, weld, normal.array as Float32Array, normalScratch);
  normal.needsUpdate = true;
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  rebuildRestSkeleton(assets, joints);
}
