import { BufferAttribute, BufferGeometry, MeshStandardMaterial, SkinnedMesh } from 'three';
import type { AvatarAssets } from '../avatar/avatarAssets';
import { createSkinMaterial } from '../viewer/skinMaterial';

export const SCAN_WRIST_M = 0.008;
export const BODY_WRIST_M = 0.025;

/** Dominant hand/finger weights, plus a short forearm band in the rest wrist frame. */
export function handVertexMask(
  positions: ArrayLike<number>,
  skinIndex: ArrayLike<number>,
  skinWeight: ArrayLike<number>,
  names: readonly string[],
  heads: ArrayLike<number>,
  extensionM: number,
): Uint8Array {
  const mask = new Uint8Array(positions.length / 3);
  const handBone = (name: string): boolean =>
    /^(hand|thumb_\d+|index_\d+|middle_\d+|ring_\d+|pinky_\d+)_[lr]$/.test(name);
  for (let v = 0; v < mask.length; v++) {
    let dominant = 0;
    for (let j = 1; j < 4; j++)
      if (skinWeight[v * 4 + j]! > skinWeight[v * 4 + dominant]!) dominant = j;
    const name = names[skinIndex[v * 4 + dominant]!] ?? '';
    if (handBone(name)) {
      mask[v] = 1;
      continue;
    }
    if (!/^lowerarm_[lr]$/.test(name)) continue;
    const wrist = names.indexOf(`hand_${name.slice(-1)}`) * 3;
    const elbow = names.indexOf(name) * 3;
    if (wrist < 0 || elbow < 0) continue;
    const dx = heads[wrist]! - heads[elbow]!;
    const dy = heads[wrist + 1]! - heads[elbow + 1]!;
    const dz = heads[wrist + 2]! - heads[elbow + 2]!;
    const length = Math.hypot(dx, dy, dz);
    if (
      length > 1e-6 &&
      ((positions[v * 3]! - heads[wrist]!) * dx +
        (positions[v * 3 + 1]! - heads[wrist + 1]!) * dy +
        (positions[v * 3 + 2]! - heads[wrist + 2]!) * dz) /
        length >=
        -extensionM
    )
      mask[v] = 1;
  }
  return mask;
}

/** Own geometry/material; the hidden body's buffers and wardrobe index are never changed. */
export class TwinHands {
  readonly mesh: SkinnedMesh;
  private readonly geometry = new BufferGeometry();
  private readonly material: MeshStandardMaterial;
  private vertices = new Uint32Array();
  private initialized = false;

  constructor(
    private readonly assets: AvatarAssets,
    skinToneHex: string,
  ) {
    this.material = createSkinMaterial({ color: skinToneHex });
    // Empty but valid geometry: pointer raycasts ignore `visible`, and three reads `attributes.position.count`.
    this.geometry.setAttribute('position', new BufferAttribute(new Float32Array(0), 3));
    this.mesh = new SkinnedMesh(this.geometry, this.material);
    this.mesh.name = 'twin:hands';
    this.mesh.visible = false;
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.mesh.bind(assets.skeleton, assets.mesh.bindMatrix);
    assets.scene.add(this.mesh);
  }

  update(heads: Float32Array): void {
    const source = this.assets.mesh.geometry;
    const position = source.getAttribute('position');
    const normal = source.getAttribute('normal');
    const skinIndex = source.getAttribute('skinIndex');
    const skinWeight = source.getAttribute('skinWeight');
    if (!skinIndex || !skinWeight) return; // synthetic rigs may have no body skin
    if (!this.initialized) {
      const mask = handVertexMask(
        position.array,
        skinIndex.array,
        skinWeight.array,
        this.assets.skeleton.bones.map((b) => b.name),
        heads,
        BODY_WRIST_M,
      );
      const indices = this.assets.indices ?? Uint32Array.from(source.getIndex()!.array);
      const selected: number[] = [];
      const vertexMap = new Map<number, number>();
      const compact: number[] = [];
      for (let t = 0; t < indices.length; t += 3) {
        if (![indices[t]!, indices[t + 1]!, indices[t + 2]!].some((v) => mask[v])) continue;
        for (let j = 0; j < 3; j++) {
          const v = indices[t + j]!;
          if (!vertexMap.has(v)) {
            vertexMap.set(v, selected.length);
            selected.push(v);
          }
          compact.push(vertexMap.get(v)!);
        }
      }
      this.vertices = Uint32Array.from(selected);
      this.geometry.setIndex(compact);
      this.geometry.setAttribute(
        'position',
        new BufferAttribute(new Float32Array(selected.length * 3), 3),
      );
      this.geometry.setAttribute(
        'normal',
        new BufferAttribute(new Float32Array(selected.length * 3), 3),
      );
      const joints = new Uint16Array(selected.length * 4);
      const weights = new Float32Array(selected.length * 4);
      selected.forEach((v, i) => {
        for (let j = 0; j < 4; j++) {
          joints[i * 4 + j] = skinIndex.array[v * 4 + j]!;
          weights[i * 4 + j] = skinWeight.array[v * 4 + j]!;
        }
      });
      this.geometry.setAttribute('skinIndex', new BufferAttribute(joints, 4));
      this.geometry.setAttribute('skinWeight', new BufferAttribute(weights, 4));
      this.initialized = true;
    }
    const out = this.geometry.getAttribute('position');
    const normals = this.geometry.getAttribute('normal');
    this.vertices.forEach((v, i) => {
      for (let j = 0; j < 3; j++) {
        // One millimetre radial overlap conceals the scan cut without enlarging the fingers appreciably.
        out.array[i * 3 + j] = position.array[v * 3 + j]! + (normal?.array[v * 3 + j] ?? 0) * 0.001;
        normals.array[i * 3 + j] = normal?.array[v * 3 + j] ?? 0;
      }
    });
    out.needsUpdate = true;
    normals.needsUpdate = true;
    this.mesh.bindMatrix.copy(this.assets.mesh.bindMatrix);
    this.mesh.bindMatrixInverse.copy(this.assets.mesh.bindMatrixInverse);
    this.mesh.visible = this.vertices.length > 0;
  }

  dispose(): void {
    this.mesh.removeFromParent();
    this.geometry.dispose();
    this.material.dispose();
  }
}
