// @ts-expect-error Node types are not in the browser app's tsconfig; Vitest supplies this module.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { Quaternion, Vector3 } from 'three';
import type { BodyManifest, JointRef, PoseDef, RigDef } from '@dt/avatar-core';
import { primitive, parseGlb } from '../wardrobe/realAssets.testkit';
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

type PoseState = ReturnType<typeof fk>;

/**
 * Unit normal pointing OUT of the palm. fingers x (pinky -> index) is the outward normal of the left palm; the right hand
 * is its mirror image, so the cross product flips there.
 */
function palmNormal(state: PoseState, side: string): Vector3 {
  const normal = state
    .joint(`middle_01_${side}`)
    .clone()
    .sub(state.joint(`hand_${side}`))
    .cross(state.joint(`index_01_${side}`).clone().sub(state.joint(`pinky_01_${side}`)))
    .normalize();
  return side === 'l' ? normal : normal.negate();
}

/** Swing-twist: twist angle (radians, -pi..pi) of q about the unit axis. */
function twistAbout(q: Quaternion, axis: Vector3): number {
  let angle = 2 * Math.atan2(axis.x * q.x + axis.y * q.y + axis.z * q.z, q.w);
  if (angle > Math.PI) angle -= 2 * Math.PI;
  if (angle < -Math.PI) angle += 2 * Math.PI;
  return angle;
}

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
      // the visible hand axis: wrist to the middle knuckle (the hand bone's own tail points 25-40 deg away from it)
      const hand = state.joint(`middle_01_${side}`).clone().sub(wrist).normalize();
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
      // palms DOWN on both hands (the first generator mirrored the target twice and left them up and twisted)
      expect(palmNormal(state, side).y, `${side} palm`).toBeLessThan(-0.85);
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
      // the palm faces the thigh: toward the body midline, -X for the left hand and +X for the right hand
      expect(palmNormal(state, side).x * (side === 'l' ? -1 : 1), `${side} palm`).toBeGreaterThan(0.8);
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

const degrees = (radians: number) => (radians * 180) / Math.PI;

describe('hand roll about the forearm', () => {
  it('stays small on hand and lowerarm in every pose (no twist bones: roll pinches the wrist)', () => {
    for (const value of poses.values()) {
      for (const side of ['l', 'r']) {
        const lower = boneByName.get(`lowerarm_${side}`)!;
        const axis = point(lower.tail).sub(point(lower.head)).normalize();
        for (const bone of [`hand_${side}`, `lowerarm_${side}`]) {
          const q = value.bones[bone];
          const roll = degrees(twistAbout(q ? new Quaternion(...q) : new Quaternion(), axis));
          // a ring pinches to cos(roll / 2): 30 deg keeps it above 0.96, the first generator had 165 deg (0.13)
          expect(Math.abs(roll), `${value.id} ${bone}`).toBeLessThan(30);
        }
      }
    }
  });

  it('is mirror symmetric between the left and the right arm', () => {
    // the first generator used the raw palm cross product on both hands: the hands ended up with opposite meanings
    for (const id of ['relaxed', 't-pose', 'hands-on-hips']) {
      for (const base of ['clavicle', 'upperarm', 'lowerarm', 'hand', 'index_01', 'thumb_02']) {
        const left = pose(id).bones[`${base}_l`] ?? [0, 0, 0, 1];
        const right = pose(id).bones[`${base}_r`] ?? [0, 0, 0, 1];
        const mirrored = new Quaternion(left[0], -left[1], -left[2], left[3]);
        expect(new Quaternion(...right).angleTo(mirrored), `${id} ${base}`).toBeLessThan(0.01);
      }
    }
  });
});

describe('wrists under linear blend skinning on the real mannequin', () => {
  const mesh = primitive(parseGlb(readFileSync(new URL('../../../public/assets/body/base.glb', import.meta.url))));
  const { position, joints, weights } = mesh;
  const restHeads = rig.bones.map((bone) => point(bone.head));
  const vertexCount = position.length / 3;

  function skinPose(value: PoseDef): Float32Array {
    if (!joints || !weights) throw new Error('base.glb has no skin');
    const state = fk(value);
    const rotations = rig.bones.map((bone) => state.rotation(bone.name));
    const heads = rig.bones.map((bone) => state.joint(bone.name));
    const out = new Float32Array(position.length);
    const p = new Vector3();
    const sum = new Vector3();
    for (let i = 0; i < vertexCount; i++) {
      sum.set(0, 0, 0);
      for (let k = 0; k < 4; k++) {
        const w = weights[4 * i + k]!;
        if (w === 0) continue;
        const b = joints[4 * i + k]!;
        p.set(position[3 * i]!, position[3 * i + 1]!, position[3 * i + 2]!)
          .sub(restHeads[b]!)
          .applyQuaternion(rotations[b]!)
          .add(heads[b]!);
        sum.addScaledVector(p, w);
      }
      out.set([sum.x, sum.y, sum.z], 3 * i);
    }
    return out;
  }

  /** Area of the 2D convex hull of the vertices projected on the plane perpendicular to `axis` (monotone chain). */
  function hullArea(positions: ArrayLike<number>, ids: number[], axis: Vector3): number {
    const u = new Vector3().crossVectors(axis, Math.abs(axis.y) < 0.9 ? new Vector3(0, 1, 0) : new Vector3(1, 0, 0)).normalize();
    const v = new Vector3().crossVectors(axis, u).normalize();
    const pts = ids
      .map((i) => {
        const q = new Vector3(positions[3 * i]!, positions[3 * i + 1]!, positions[3 * i + 2]!);
        return [q.dot(u), q.dot(v)] as const;
      })
      .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (o: readonly number[], a: readonly number[], b: readonly number[]) =>
      (a[0]! - o[0]!) * (b[1]! - o[1]!) - (a[1]! - o[1]!) * (b[0]! - o[0]!);
    const chain = (list: (readonly [number, number])[]) => {
      const hull: (readonly [number, number])[] = [];
      for (const pt of list) {
        while (hull.length >= 2 && cross(hull[hull.length - 2]!, hull[hull.length - 1]!, pt) <= 0) hull.pop();
        hull.push(pt);
      }
      hull.pop();
      return hull;
    };
    const ring = [...chain(pts), ...chain([...pts].reverse())];
    let area = 0;
    ring.forEach((a, k) => {
      const b = ring[(k + 1) % ring.length]!;
      area += a[0] * b[1] - b[0] * a[1];
    });
    return Math.abs(area) / 2;
  }

  const slabCenters = [-0.03, -0.015, 0, 0.015, 0.03];

  /** Vertices of thin slabs across the rest forearm / wrist (s = distance from the wrist joint along the forearm). */
  function slabs(side: string): number[][] {
    const axis = point(boneByName.get(`hand_${side}`)!.head)
      .sub(point(boneByName.get(`lowerarm_${side}`)!.head))
      .normalize();
    const wrist = point(boneByName.get(`hand_${side}`)!.head);
    return slabCenters.map((center) => {
      const ids: number[] = [];
      for (let i = 0; i < vertexCount; i++) {
        const rel = new Vector3(position[3 * i]!, position[3 * i + 1]!, position[3 * i + 2]!).sub(wrist);
        const along = rel.dot(axis);
        const radial = rel.addScaledVector(axis, -along).length();
        if (Math.abs(along - center) <= 0.006 && radial < 0.07) ids.push(i);
      }
      return ids;
    });
  }

  /** Plane normal of a slab: the forearm axis before the wrist, the hand axis (wrist to middle knuckle) after it. */
  function ringAxis(center: number, elbow: Vector3, wrist: Vector3, knuckle: Vector3): Vector3 {
    const forearm = wrist.clone().sub(elbow).normalize();
    const hand = knuckle.clone().sub(wrist).normalize();
    if (center < -0.005) return forearm;
    return center > 0.005 ? hand : forearm.add(hand).normalize();
  }

  it.each(['l', 'r'])('keeps the %s forearm and wrist ring open in every pose', (side) => {
    const rings = slabs(side);
    const head = (name: string) => point(boneByName.get(name)!.head);
    const restArea = rings.map((ring, k) => {
      expect(ring.length, 'ring vertices').toBeGreaterThan(8);
      const axis = ringAxis(
        slabCenters[k]!,
        head(`lowerarm_${side}`),
        head(`hand_${side}`),
        head(`middle_01_${side}`),
      );
      return hullArea(position, ring, axis);
    });
    for (const id of ids) {
      const value = pose(id);
      const posed = skinPose(value);
      const state = fk(value);
      rings.forEach((ring, k) => {
        const axis = ringAxis(
          slabCenters[k]!,
          state.joint(`lowerarm_${side}`),
          state.joint(`hand_${side}`),
          state.joint(`middle_01_${side}`),
        );
        const ratio = hullArea(posed, ring, axis) / restArea[k]!;
        // candy wrapper: 0.13 with the first generator's 165 deg hand roll, >= 0.86 after the fix
        expect(ratio, `${id} ${side} slab ${k}`).toBeGreaterThan(0.8);
      });
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
