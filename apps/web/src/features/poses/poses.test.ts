// @ts-expect-error Node types are not in the browser app's tsconfig; Vitest supplies this module.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { Quaternion, Vector3 } from 'three';
import type { BodyManifest, JointRef, PoseDef, RigDef } from '@dt/avatar-core';
import { blendQuaternion, easeInOut, parsePoseDef } from './poseEngine';

const readAsset = <T>(asset: string): T =>
  JSON.parse(
    readFileSync(new URL(`../../../public/assets/${asset}`, import.meta.url), 'utf8'),
  ) as T;
const rig = readAsset<RigDef>('body/rig.json');
const manifest = readAsset<BodyManifest>('body/manifest.json');
const ids = ['t-pose', 'a-pose', 'relaxed', 'hands-on-hips', 'walk', 'side'];
const poses = new Map(
  ids.map((id) => [id, parsePoseDef(readAsset<unknown>(`poses/${id}.json`), id)]),
);
const boneByName = new Map(rig.bones.map((bone) => [bone.name, bone]));

function point(ref: JointRef): Vector3 {
  if (ref.strategy !== 'VERTEX' || ref.vert < manifest.renderVertexCount) {
    throw new Error('Expected a joint point');
  }
  const jointPoint = manifest.jointPoints[ref.vert - manifest.renderVertexCount];
  if (!jointPoint) throw new Error(`Missing joint point ${ref.vert}`);
  return new Vector3(...jointPoint.position);
}

/** FK matches the glTF hierarchy: rest frames are world-aligned and heads translate with parents. */
function fk(pose: PoseDef) {
  const rotations = new Map<string, Quaternion>();
  const heads = new Map<string, Vector3>();
  const tails = new Map<string, Vector3>();
  for (const bone of rig.bones) {
    const restHead = point(bone.head);
    const parent = bone.parent ? boneByName.get(bone.parent) : undefined;
    const parentRotation = parent ? rotations.get(parent.name) : undefined;
    const parentHead = parent ? heads.get(parent.name) : undefined;
    const head =
      parent && parentRotation && parentHead
        ? parentHead
            .clone()
            .add(restHead.clone().sub(point(parent.head)).applyQuaternion(parentRotation))
        : restHead;
    const local = pose.bones[bone.name];
    const rotation = (parentRotation?.clone() ?? new Quaternion()).multiply(
      local ? new Quaternion(...local) : new Quaternion(),
    );
    heads.set(bone.name, head);
    rotations.set(bone.name, rotation);
    tails.set(
      bone.name,
      head.clone().add(point(bone.tail).sub(restHead).applyQuaternion(rotation)),
    );
  }
  const joint = (name: string) => {
    const found = heads.get(name);
    if (!found) throw new Error(`Missing ${name}`);
    return found;
  };
  const end = (name: string) => {
    const found = tails.get(name);
    if (!found) throw new Error(`Missing ${name}`);
    return found;
  };
  const rotation = (name: string) => {
    const found = rotations.get(name);
    if (!found) throw new Error(`Missing ${name}`);
    return found;
  };
  return { joint, end, rotation };
}

const pose = (id: string) => {
  const found = poses.get(id);
  if (!found) throw new Error(`Missing pose ${id}`);
  return found;
};

describe('generated poses on the real rig', () => {
  it('uses valid names and unit quaternions', () => {
    for (const value of poses.values()) {
      for (const [name, q] of Object.entries(value.bones)) {
        expect(boneByName.has(name), `${value.id}: ${name}`).toBe(true);
        expect(Math.abs(Math.hypot(...q) - 1), `${value.id}: ${name}`).toBeLessThan(2e-6);
      }
    }
  });

  it('places both T-pose wrists at shoulder height with straight arms', () => {
    const state = fk(pose('t-pose'));
    for (const side of ['l', 'r']) {
      const shoulder = state.joint(`upperarm_${side}`);
      const elbow = state.joint(`lowerarm_${side}`);
      const wrist = state.joint(`hand_${side}`);
      expect(Math.abs(wrist.y - shoulder.y)).toBeLessThan(0.03);
      expect(Math.abs(wrist.x)).toBeGreaterThan(Math.abs(shoulder.x) + 0.35);
      expect(
        elbow.clone().sub(shoulder).normalize().dot(wrist.clone().sub(elbow).normalize()),
      ).toBeGreaterThan(0.99);
      const forearm = wrist.clone().sub(elbow).normalize();
      const hand = state.end(`hand_${side}`).clone().sub(wrist).normalize();
      expect(forearm.angleTo(hand), `${side} wrist bend`).toBeLessThan((5 * Math.PI) / 180);
      for (const finger of ['index', 'middle', 'ring', 'pinky']) {
        for (const segment of [1, 2, 3]) {
          const name = `${finger}_0${segment}_${side}`;
          const fingerDirection = state.end(name).clone().sub(state.joint(name)).normalize();
          expect(forearm.angleTo(fingerDirection), `${name} bend`).toBeLessThan((7 * Math.PI) / 180);
        }
        expect(Math.abs(state.end(`${finger}_03_${side}`).y - shoulder.y), `${finger} fingertip height`)
          .toBeLessThan(0.03);
      }
      const thumbDirection = state.end(`thumb_03_${side}`).clone().sub(state.joint(`thumb_03_${side}`)).normalize();
      expect(thumbDirection.z).toBeGreaterThan(0.8);
      expect(Math.abs(state.end(`thumb_03_${side}`).y - shoulder.y)).toBeLessThan(0.03);
      const palmNormal = state
        .joint(`middle_01_${side}`)
        .clone()
        .sub(wrist)
        .cross(
          state
            .joint(`pinky_01_${side}`)
            .clone()
            .sub(state.joint(`index_01_${side}`)),
        )
        .normalize();
      expect(palmNormal.y * (side === 'l' ? -1 : 1)).toBeGreaterThan(0.85);
    }
  });

  it('keeps A-pose at rest', () => {
    expect(pose('a-pose').bones).toEqual({});
    const state = fk(pose('a-pose'));
    for (const bone of rig.bones) {
      expect(state.joint(bone.name).distanceTo(point(bone.head))).toBeLessThan(1e-8);
      expect(state.end(bone.name).distanceTo(point(bone.tail))).toBeLessThan(1e-8);
    }
  });

  it('hangs relaxed hands below the pelvis and outside the thighs', () => {
    const state = fk(pose('relaxed'));
    for (const side of ['l', 'r']) {
      const hand = state.joint(`hand_${side}`);
      expect(hand.y).toBeLessThan(state.joint('pelvis').y);
      expect(Math.abs(hand.x)).toBeGreaterThan(Math.abs(state.joint(`thigh_${side}`).x));
      const palmNormal = state.joint(`middle_01_${side}`).clone().sub(hand)
        .cross(state.joint(`pinky_01_${side}`).clone().sub(state.joint(`index_01_${side}`))).normalize();
      expect(palmNormal.x).toBeLessThan(-0.8);
    }
  });

  it('puts hands on the same-side hips with elbows outside', () => {
    const state = fk(pose('hands-on-hips'));
    for (const side of ['l', 'r']) {
      const hand = state.joint(`hand_${side}`);
      const thigh = state.joint(`thigh_${side}`);
      const hipRegion = new Vector3(Math.sign(thigh.x) * 0.225, thigh.y + 0.055, thigh.z + 0.065);
      expect(hand.distanceTo(hipRegion)).toBeLessThan(0.12);
      expect(Math.abs(state.joint(`lowerarm_${side}`).x)).toBeGreaterThan(Math.abs(hand.x));
    }
  });

  it('makes a grounded mid-stride with at least 20 cm foot separation', () => {
    const state = fk(pose('walk'));
    const left = state.joint('foot_l');
    const right = state.joint('foot_r');
    expect(left.z - right.z).toBeGreaterThanOrEqual(0.2);
    for (const foot of [left, right]) {
      expect(foot.y).toBeGreaterThan(0.02);
      expect(foot.y).toBeLessThan(0.22);
    }
  });

  it('turns pelvis and head forward axes 90 degrees about Y', () => {
    const state = fk(pose('side'));
    for (const name of ['pelvis', 'head']) {
      const forward = new Vector3(0, 0, 1).applyQuaternion(state.rotation(name));
      expect(forward.x).toBeGreaterThan(0.99);
      expect(Math.abs(forward.z)).toBeLessThan(0.02);
    }
  });
});

describe('pose engine', () => {
  it('blends at endpoints and follows the shortest quaternion path', () => {
    const from = new Quaternion();
    const to = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), Math.PI * 0.75);
    expect(blendQuaternion(from, to, 0).angleTo(from)).toBeLessThan(1e-8);
    expect(blendQuaternion(from, to, 1).angleTo(to)).toBeLessThan(1e-8);
    const oppositeSign = new Quaternion(-to.x, -to.y, -to.z, -to.w);
    expect(
      blendQuaternion(from, oppositeSign, 0.5).angleTo(blendQuaternion(from, to, 0.5)),
    ).toBeLessThan(1e-8);
    expect(easeInOut(0)).toBe(0);
    expect(easeInOut(1)).toBe(1);
  });

  it('rejects malformed quaternions', () => {
    expect(() =>
      parsePoseDef(
        { id: 'broken', label: { tr: 'x', en: 'x' }, bones: { hand_l: [0, 0, 0, 2] } },
        'broken',
      ),
    ).toThrow();
  });
});
