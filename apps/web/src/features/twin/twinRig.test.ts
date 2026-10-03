import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  BufferAttribute,
  BufferGeometry,
  MeshBasicMaterial,
  MeshStandardMaterial,
  SkinnedMesh,
} from 'three';
import { filterBodyIndex, hiddenVertexMask } from '../wardrobe/bodyHide';
import type { TwinRuntimeInfo } from '../../store/twinStore';
import { bodyIndex, envelope, fakeAssets, fakeModel, mapping } from './twinTestkit';
import { MAX_HEAD_RESIDUAL_M, TwinRig } from './twinRig';
import { TwinOpeningRepair } from './twinOpeningRepair';
import { TwinHands } from './twinHands';

afterEach(() => vi.restoreAllMocks());

describe('TwinRig hands', () => {
  /** Every twin vertex is skinned to a hand bone, so the scan-hand swap would hide the whole mesh. */
  function handSetup(options?: { keepOwnHands?: boolean }) {
    const assets = fakeAssets();
    assets.skeleton.bones[2]!.name = 'hand_l';
    const model = fakeModel();
    model.skinIndex.fill(2);
    const update = vi.spyOn(TwinHands.prototype, 'update');
    const rig = new TwinRig(assets, model, null, undefined, '#c99a7e', options);
    rig.onSolve(envelope(true));
    return { rig, update };
  }

  it('swaps in the mannequin hands by default: scan hand triangles are hidden', () => {
    const { rig, update } = handSetup();
    expect(update).toHaveBeenCalledTimes(1);
    expect(rig.mesh.geometry.drawRange.count).toBe(0);
    rig.dispose();
  });

  it('keeps the twin own hands when keepOwnHands is set (dtHasMakeHumanHands)', () => {
    const { rig, update } = handSetup({ keepOwnHands: true });
    expect(update).not.toHaveBeenCalled();
    expect(rig.mesh.geometry.drawRange.count).toBe(15);
    expect(rig.mesh.visible).toBe(true);
    rig.dispose();
  });

  it('keepOwnHands false behaves like the default', () => {
    const { rig } = handSetup({ keepOwnHands: false });
    expect(rig.mesh.geometry.drawRange.count).toBe(0);
    rig.dispose();
  });
});

describe('TwinRig', () => {
  it('pushes clothing coverage and its margin inward once, recomputes normals, and restores on take-off', () => {
    const repairs = vi.spyOn(TwinOpeningRepair.prototype, 'update');
    const assets = fakeAssets();
    const model = fakeModel();
    for (let v = 0; v < model.vertexCount; v++) {
      model.position[v * 3] = model.position[v * 3]! * 0.1;
      model.position[v * 3 + 1] = 1 + Math.floor(v / 5) * 0.01;
      model.normal[v * 3 + 2] = 1;
    }
    const rig = new TwinRig(assets, model, null);
    rig.onSolve(envelope(true));
    const position = rig.mesh.geometry.getAttribute('position');
    const original = Float32Array.from(position.array);
    const garment = new BufferGeometry();
    garment.setAttribute(
      'position',
      new BufferAttribute(
        Float32Array.from([-0.02, 0.98, 0.004, 0.025, 0.98, 0.004, 0, 1.04, 0.004]),
        3,
      ),
    );
    garment.setIndex([0, 1, 2]);
    const mesh = new SkinnedMesh(garment, new MeshBasicMaterial());
    mesh.name = 'garment:test';
    assets.scene.add(mesh);
    assets.mesh.geometry.setIndex(new BufferAttribute(bodyIndex, 1));
    const deltas = Array.from(
      { length: model.vertexCount },
      (_, v) => original[v * 3 + 2]! - position.getZ(v),
    );
    expect(Math.max(...deltas)).toBeCloseTo(0.008);
    expect(deltas.some((d) => d > 0 && d < 0.0079)).toBe(true);
    expect(Array.from(rig.mesh.geometry.getAttribute('normal').array).every(Number.isFinite)).toBe(
      true,
    );
    const first = Float32Array.from(position.array);
    const normals = rig.mesh.geometry.getAttribute('normal') as BufferAttribute;
    const normalVersion = normals.version;
    const repairCount = repairs.mock.calls.length;
    (garment.getAttribute('position') as BufferAttribute).needsUpdate = true; // wardrobe uploads unchanged rest data
    rig.onSolve(envelope(true)); // cached: no accumulating shrink on repeated solves
    expect(position.array).toEqual(first);
    expect(normals.version).toBe(normalVersion); // avoids another footprint / push-in / normals pass
    expect(repairs).toHaveBeenCalledTimes(repairCount); // no band/texel work for repeated solves or poses
    assets.scene.remove(mesh);
    assets.mesh.geometry.setIndex(new BufferAttribute(bodyIndex, 1));
    expect(position.array).toEqual(original);
    expect(repairs).toHaveBeenCalledTimes(repairCount + 1); // take-off restores the texture too
    rig.dispose();
    garment.dispose();
  });
  it('uses neutral matte skin for legacy twins without readable forearm texels', () => {
    const assets = fakeAssets();
    const rig = new TwinRig(assets, fakeModel(), null);
    const hands = assets.scene.children.find((child) => child.name === 'twin:hands') as SkinnedMesh;
    const material = hands.material as MeshStandardMaterial;
    expect(material.color.getHexString()).toBe('c99a7e');
    expect(material.roughness).toBe(0.6);
    expect(material.metalness).toBe(0);
    rig.dispose();
  });
  it('stays hidden and leaves the body visible until a fixed-shape solve aligns it', () => {
    const assets = fakeAssets();
    const rig = new TwinRig(assets, fakeModel(), mapping);
    expect(rig.mesh.parent).toBe(assets.scene);
    expect(rig.mesh.visible).toBe(false);
    expect(assets.mesh.visible).toBe(true);
    expect(rig.onSolve(envelope(false))).toBeNull(); // a stale standard solve is ignored
    expect(rig.mesh.visible).toBe(false);
    const alignment = rig.onSolve(envelope(true));
    expect(alignment?.maxResidual).toBeLessThan(1e-6);
    expect(rig.mesh.visible).toBe(true);
    expect(assets.mesh.visible).toBe(false);
    rig.dispose();
    expect(assets.mesh.visible).toBe(true);
    expect(rig.mesh.parent).toBeNull();
  });

  it('translates the twin onto the avatar rest heads', () => {
    const assets = fakeAssets();
    const rig = new TwinRig(assets, fakeModel(0.013), mapping); // the twin frame is 1.3 cm lower
    const alignment = rig.onSolve(envelope(true))!;
    expect(alignment.offset[1]).toBeCloseTo(0.013, 6);
    const position = rig.mesh.geometry.getAttribute('position');
    expect(position.getY(0)).toBeCloseTo(1 + 0.013, 5);
    rig.dispose();
  });

  it('does not show files whose skeleton does not match', () => {
    const assets = fakeAssets();
    const rig = new TwinRig(assets, fakeModel(), mapping);
    const alignment = rig.onSolve(envelope(true, 0.2))!; // still a pure translation: aligned
    expect(alignment.maxResidual).toBeLessThan(MAX_HEAD_RESIDUAL_M);
    const foreign = envelope(true);
    foreign.result.joints[1 * 6 + 1] = 5; // one bone far away
    const rig2 = new TwinRig(fakeAssets(), fakeModel(), mapping);
    expect(rig2.onSolve(foreign)!.maxResidual).toBeGreaterThan(MAX_HEAD_RESIDUAL_M);
    expect(rig2.mesh.visible).toBe(false);
    rig.dispose();
    rig2.dispose();
  });

  it('follows the index the wardrobe assigns to the body and restores everything on dispose', () => {
    const assets = fakeAssets();
    const info: TwinRuntimeInfo[] = [];
    const rig = new TwinRig(assets, fakeModel(), mapping, (i) => info.push(i));
    rig.onSolve(envelope(true));
    const drawn = () => rig.mesh.geometry.drawRange.count / 3;
    expect(drawn()).toBe(5);
    // the wardrobe hides body vertices 0..7 (delete list): body triangles of quads 0-2 go
    const mask = hiddenVertexMask([Uint32Array.from([0, 1, 2, 3, 4, 5, 6, 7])], 10);
    assets.mesh.geometry.setIndex(new BufferAttribute(filterBodyIndex(bodyIndex, mask), 1));
    // (0,1,5), (1,2,6), (4,5,6) map onto body vertices left without any triangle
    expect(drawn()).toBe(2);
    expect(info.at(-1)?.hiddenTriangles).toBe(3);
    expect(info.at(-1)?.triangles).toBe(5);
    // taking the garment off restores every twin triangle
    assets.mesh.geometry.setIndex(new BufferAttribute(bodyIndex, 1));
    expect(drawn()).toBe(5);
    assets.mesh.geometry.setIndex(new BufferAttribute(filterBodyIndex(bodyIndex, mask), 1));
    expect(drawn()).toBe(2);
    rig.dispose();
    // the shadowing setIndex is gone again, the body geometry is plain three.js
    expect(Object.prototype.hasOwnProperty.call(assets.mesh.geometry, 'setIndex')).toBe(false);
    assets.mesh.geometry.setIndex(new BufferAttribute(bodyIndex, 1));
  });

  it('hides twin triangles under a worn garment surface even without a mapping', () => {
    const assets = fakeAssets();
    const rig = new TwinRig(assets, fakeModel(), null);
    rig.onSolve(envelope(true));
    expect(rig.mesh.geometry.drawRange.count / 3).toBe(5);
    // a garment mesh as the wardrobe mounts it: a big triangle in the plane of the twin, a few mm in front of it
    const garment = new BufferGeometry();
    garment.setAttribute(
      'position',
      new BufferAttribute(Float32Array.from([-1, 0.5, 0.004, 3, 0.5, 0.004, 1, 2.5, 0.004]), 3),
    );
    garment.setIndex(new BufferAttribute(Uint32Array.from([0, 1, 2]), 1));
    const mesh = new SkinnedMesh(garment, new MeshBasicMaterial());
    mesh.name = 'garment:test';
    assets.scene.add(mesh);
    assets.mesh.geometry.setIndex(new BufferAttribute(bodyIndex, 1)); // the wardrobe reports a change
    expect(rig.mesh.geometry.drawRange.count).toBe(0); // every twin vertex lies under the garment
    assets.scene.remove(mesh);
    assets.mesh.geometry.setIndex(new BufferAttribute(bodyIndex, 1));
    expect(rig.mesh.geometry.drawRange.count / 3).toBe(5);
    rig.dispose();
  });

  it('rejects a mapping that does not fit the twin or the body', () => {
    expect(() => new TwinRig(fakeAssets(), fakeModel(), Uint32Array.from([0, 1]))).toThrow(
      /entries/,
    );
    expect(
      () => new TwinRig(fakeAssets(), fakeModel(), Uint32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 99])),
    ).toThrow(/outside the body/);
  });
});
