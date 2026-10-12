import { describe, expect, it } from 'vitest';
import { Bone, Matrix4, MeshStandardMaterial, Skeleton, Vector3 } from 'three';
import { createGarmentSections, garmentSkinWeights, type Vec3 } from '@dt/avatar-core';
import { buildWeldMap } from '../avatar/meshMath';
import { GarmentInstance } from './garmentInstance';
import type { TemplateRuntime } from './templateLoader';
import {
  shoeBinding,
  shoeBodySkin,
  shoeBoneNames,
  shoeDef,
  shoeHeads,
  shoeIndex,
  shoeItem,
  shoeNeutral,
  shoeSidecar,
} from './rigid.testkit';

function fixture(rigid = true) {
  const bones = shoeBoneNames.map((name) => {
    const bone = new Bone();
    bone.name = name;
    return bone;
  });
  // The actual hierarchy matters: calf -> foot -> ball on each side.
  for (const offset of [0, 3]) {
    bones[offset + 2]!.add(bones[offset]!);
    bones[offset]!.add(bones[offset + 1]!);
  }
  const skeleton = new Skeleton(bones);
  const binding = shoeBinding();
  const bodySkin = shoeBodySkin();
  const material = new MeshStandardMaterial();
  const runtime: TemplateRuntime = {
    def: shoeDef,
    vertexCount: shoeNeutral.length / 3,
    index: shoeIndex,
    uv: new Float32Array((shoeNeutral.length / 3) * 2),
    binding,
    neutralPositions: shoeNeutral,
    rigidAttachments: rigid ? shoeSidecar : null,
    ...garmentSkinWeights(binding, bodySkin.skinIndex, bodySkin.skinWeight),
    weld: buildWeldMap(shoeNeutral),
    deleteVerts: new Uint32Array(),
    material,
  };
  const instance = new GarmentInstance(runtime, shoeItem, skeleton, new Matrix4());
  function fit(scale: Vec3) {
    const heads = shoeHeads.map((p): Vec3 => [
      p[0] * scale[0],
      p[1] * scale[1] + 0.02,
      p[2] * scale[2],
    ]);
    bones.forEach((bone, i) => {
      bone.quaternion.identity();
      const parentIndex = bones.indexOf(bone.parent as Bone);
      const head = new Vector3(...heads[i]!);
      if (parentIndex >= 0) head.sub(new Vector3(...heads[parentIndex]!));
      bone.position.copy(head);
    });
    bones[2]!.updateMatrixWorld(true);
    bones[5]!.updateMatrixWorld(true);
    skeleton.calculateInverses();
    instance.syncBind(new Matrix4());
    const body = new Float32Array([...shoeNeutral, 0, 0, 0, 1, 1, 1]);
    for (let v = 0; v < body.length / 3; v++) {
      body[v * 3] = body[v * 3]! * scale[0];
      body[v * 3 + 1] = body[v * 3 + 1]! * scale[1] + 0.02;
      body[v * 3 + 2] = body[v * 3 + 2]! * scale[2];
    }
    const joints = new Float32Array(heads.length * 6);
    heads.forEach((head, i) => {
      joints.set(head, i * 6);
      joints.set(head, i * 6 + 3);
    });
    instance.update({
      body,
      joints,
      sections: createGarmentSections(body),
      achievedCm: { footLength: 25 * scale[2] },
      planes: undefined,
      heatmap: true,
      lowers: [],
    });
  }
  function pose(angles: readonly number[]) {
    bones.forEach((bone, i) => {
      bone.rotation.set(angles[i]!, 0, i % 3 === 2 ? 0.05 : 0);
    });
    bones[2]!.updateMatrixWorld(true);
    bones[5]!.updateMatrixWorld(true);
    instance.mesh.updateMatrixWorld(true);
    skeleton.update();
  }
  const posed = (v: number) => instance.mesh.getVertexPosition(v, new Vector3());
  return {
    instance,
    bones,
    fit,
    pose,
    posed,
    dispose: () => {
      instance.dispose();
      material.dispose();
      skeleton.dispose();
    },
  };
}

describe('mixed shoe skinning on the shared avatar skeleton', () => {
  const poses = [
    { name: 'A', angles: [0, 0, 0, 0, 0, 0] },
    { name: 'T', angles: [0, 0, 0, 0, 0, 0] },
    { name: 'relaxed', angles: [0.05, 0.1, -0.05, -0.05, 0.05, 0.1] },
    { name: 'walk', angles: [-0.5, 0.7, 0.4, 0.3, -0.4, -0.35] },
  ];
  it.each(poses)(
    'preserves all foot/ball sole segment edge lengths in $name after body regrades',
    ({ angles }) => {
      const f = fixture();
      try {
        for (const scale of [
          [1, 1, 1],
          [0.8, 0.9, 0.85],
          [1.2, 1.1, 1.15],
        ] as Vec3[]) {
          f.instance.setItem({ ...shoeItem, selectedSize: scale[0] > 1 ? 'L' : 'M' });
          f.fit(scale);
          const position = f.instance.mesh.geometry.getAttribute('position');
          f.pose(angles);
          for (const attachment of shoeSidecar.attachments) {
            for (let i = 0; i < 3; i++) {
              const a = attachment.vertices[i]!;
              const b = attachment.vertices[(i + 1) % 3]!;
              const p = attachment.localPositionsM[i]!;
              const q = attachment.localPositionsM[(i + 1) % 3]!;
              const expected = Math.hypot(...p.map((n, axis) => (n - q[axis]!) * scale[axis]!));
              expect(f.posed(a).distanceTo(f.posed(b))).toBeCloseTo(expected, 6);
              const skin = f.instance.mesh.geometry.getAttribute('skinWeight');
              expect(skin.getX(a)).toBe(1);
              expect(skin.getY(a) + skin.getZ(a) + skin.getW(a)).toBe(0);
            }
          }
          expect([...position.array].every(Number.isFinite)).toBe(true);
          expect(
            [...f.instance.mesh.geometry.getAttribute('color').array].every(Number.isFinite),
          ).toBe(true);
          const first = Array.from(position.array);
          f.fit(scale);
          expect(Array.from(position.array)).toEqual(first);
        }
      } finally {
        f.dispose();
      }
    },
  );

  it('preserves MHCLO upper geometry and weights, follows the moving foot, and mixes the boundary ring', () => {
    const mixed = fixture();
    const barycentric = fixture(false);
    try {
      for (const f of [mixed, barycentric]) f.fit([1, 1, 1]);
      for (const v of [13, 15]) {
        for (const name of ['position', 'skinIndex', 'skinWeight']) {
          const a = mixed.instance.mesh.geometry.getAttribute(name);
          const b = barycentric.instance.mesh.geometry.getAttribute(name);
          expect(Array.from(a.array.slice(v * a.itemSize, (v + 1) * a.itemSize))).toEqual(
            Array.from(b.array.slice(v * b.itemSize, (v + 1) * b.itemSize)),
          );
        }
      }
      const bandSkin = mixed.instance.mesh.geometry.getAttribute('skinWeight');
      expect(bandSkin.getX(12)).toBeCloseTo(0.75, 5);
      expect(bandSkin.getY(12)).toBeCloseTo(0.25, 5);
      const before = mixed.posed(13).clone();
      const angles = [0.6, 0.8, 0.3, -0.4, -0.6, -0.2];
      for (const f of [mixed, barycentric]) f.pose(angles);
      expect(mixed.posed(13).distanceTo(before)).toBeGreaterThan(0.01);
      expect(mixed.posed(13).distanceTo(barycentric.posed(13))).toBeLessThan(1e-7);
      // Control: transferring mixed body weights to the soles bends them in this synthetic walk.
      expect(
        Math.abs(barycentric.posed(0).distanceTo(barycentric.posed(2)) - 0.01),
      ).toBeGreaterThan(0.0001);
    } finally {
      mixed.dispose();
      barycentric.dispose();
    }
  });
});
