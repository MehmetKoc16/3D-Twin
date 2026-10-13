// @ts-expect-error Node types are not in the browser app's tsconfig; Vitest supplies this module.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { Box3, Quaternion, Ray, Triangle, Vector3 } from 'three';
import {
  applyMorphs,
  buildBasePositions,
  macroWeights,
  type BodyManifest,
  type JointRef,
  type PoseDef,
  type RigDef,
} from '@dt/avatar-core';
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
function fk(pose: PoseDef, pointOf: (ref: JointRef) => Vector3 = point) {
  const rotations = new Map<string, Quaternion>();
  const heads = new Map<string, Vector3>();
  const tails = new Map<string, Vector3>();
  for (const bone of rig.bones) {
    const restHead = pointOf(bone.head);
    const parent = bone.parent ? boneByName.get(bone.parent) : undefined;
    const parentRotation = parent ? rotations.get(parent.name) : undefined;
    const parentHead = parent ? heads.get(parent.name) : undefined;
    const head =
      parent && parentRotation && parentHead
        ? parentHead
            .clone()
            .add(restHead.clone().sub(pointOf(parent.head)).applyQuaternion(parentRotation))
        : restHead;
    const local = pose.bones[bone.name];
    const rotation = (parentRotation?.clone() ?? new Quaternion()).multiply(
      local ? new Quaternion(...local) : new Quaternion(),
    );
    heads.set(bone.name, head);
    rotations.set(bone.name, rotation);
    tails.set(
      bone.name,
      head.clone().add(pointOf(bone.tail).sub(restHead).applyQuaternion(rotation)),
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
    .cross(
      state
        .joint(`index_01_${side}`)
        .clone()
        .sub(state.joint(`pinky_01_${side}`)),
    )
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
          expect(forearm.angleTo(fingerDirection), `${name} bend`).toBeLessThan(
            (7 * Math.PI) / 180,
          );
        }
        expect(
          Math.abs(state.end(`${finger}_03_${side}`).y - shoulder.y),
          `${finger} fingertip height`,
        ).toBeLessThan(0.03);
      }
      const thumbDirection = state
        .end(`thumb_03_${side}`)
        .clone()
        .sub(state.joint(`thumb_03_${side}`))
        .normalize();
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
      expect(palmNormal(state, side).x * (side === 'l' ? -1 : 1), `${side} palm`).toBeGreaterThan(
        0.8,
      );
    }
  });

  it('puts hands on the same-side hips with elbows outside', () => {
    const state = fk(pose('hands-on-hips'));
    for (const side of ['l', 'r']) {
      const hand = state.joint(`hand_${side}`);
      const thigh = state.joint(`thigh_${side}`);
      // the wrist rests on the iliac crest / waist (a hip-joint-level target sank the palm into the thigh and pelvis)
      const hipRegion = new Vector3(Math.sign(thigh.x) * 0.19, thigh.y + 0.17, thigh.z + 0.065);
      expect(hand.distanceTo(hipRegion)).toBeLessThan(0.12);
      expect(Math.abs(state.joint(`lowerarm_${side}`).x)).toBeGreaterThan(Math.abs(hand.x));
    }
  });

  it('keeps hands-on-hips wrist flexion within 20 degrees with fingers forward and down', () => {
    const state = fk(pose('hands-on-hips'));
    for (const side of ['l', 'r']) {
      const wrist = state.joint(`hand_${side}`);
      const forearm = wrist
        .clone()
        .sub(state.joint(`lowerarm_${side}`))
        .normalize();
      const fingers = state.joint(`middle_01_${side}`).clone().sub(wrist).normalize();
      expect(forearm.angleTo(fingers), `${side} wrist flexion`).toBeLessThanOrEqual(
        (20 * Math.PI) / 180,
      );
      expect(fingers.y, `${side} fingers down`).toBeLessThan(-0.5);
      expect(fingers.z, `${side} fingers forward`).toBeGreaterThan(0.5);
      expect(palmNormal(state, side).x * (side === 'l' ? -1 : 1)).toBeGreaterThan(0.9);
    }
  });

  it('mirrors every hands-on-hips arm and finger rotation and joint position', () => {
    const value = pose('hands-on-hips');
    const state = fk(value);
    for (const bone of rig.bones.filter(({ name }) => name.endsWith('_l'))) {
      const rightName = bone.name.replace(/_l$/, '_r');
      const left = value.bones[bone.name] ?? [0, 0, 0, 1];
      const right = value.bones[rightName] ?? [0, 0, 0, 1];
      expect(
        new Quaternion(...right)
          .normalize()
          .angleTo(new Quaternion(left[0], -left[1], -left[2], left[3]).normalize()),
        bone.name,
      ).toBeLessThan(1e-5);
      const mirrored = state.joint(bone.name).clone();
      mirrored.x *= -1;
      expect(state.joint(rightName).distanceTo(mirrored), bone.name).toBeLessThan(1e-5);
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
  const mesh = primitive(
    parseGlb(readFileSync(new URL('../../../public/assets/body/base.glb', import.meta.url))),
  );
  const { position, joints, weights } = mesh;
  const vertexCount = position.length / 3;

  function skinPose(
    value: PoseDef,
    pointOf: (ref: JointRef) => Vector3 = point,
    rest: ArrayLike<number> = position,
  ): Float32Array {
    if (!joints || !weights) throw new Error('base.glb has no skin');
    const restHeads = rig.bones.map((bone) => pointOf(bone.head));
    const state = fk(value, pointOf);
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
        p.set(rest[3 * i]!, rest[3 * i + 1]!, rest[3 * i + 2]!)
          .sub(restHeads[b]!)
          .applyQuaternion(rotations[b]!)
          .add(heads[b]!);
        sum.addScaledVector(p, w);
      }
      out.set([sum.x, sum.y, sum.z], 3 * i);
    }
    return out;
  }

  /** Segment p0-p1 crosses the triangle (edge-vs-triangle test with a ray; coplanar contact is ignored). */
  function segmentHitsTriangle(p0: Vector3, p1: Vector3, tri: Triangle): boolean {
    const direction = p1.clone().sub(p0);
    const length = direction.length();
    if (length < 1e-9) return false;
    const ray = new Ray(p0, direction.divideScalar(length));
    const hit = ray.intersectTriangle(tri.a, tri.b, tri.c, false, new Vector3());
    return hit !== null && hit.distanceTo(p0) <= length;
  }

  /** Number of (a, b) triangle pairs that cut through each other: some edge of one pierces the other. */
  function countIntersectingPairs(first: Triangle[], second: Triangle[]): number {
    const box = (tri: Triangle) => new Box3().setFromPoints([tri.a, tri.b, tri.c]);
    const secondBoxes = second.map(box);
    let count = 0;
    for (const a of first) {
      const aBox = box(a);
      for (let j = 0; j < second.length; j++) {
        if (!aBox.intersectsBox(secondBoxes[j]!)) continue;
        const b = second[j]!;
        const aEdges: [Vector3, Vector3][] = [
          [a.a, a.b],
          [a.b, a.c],
          [a.c, a.a],
        ];
        const bEdges: [Vector3, Vector3][] = [
          [b.a, b.b],
          [b.b, b.c],
          [b.c, b.a],
        ];
        if (
          aEdges.some(([p, q]) => segmentHitsTriangle(p, q, b)) ||
          bEdges.some(([p, q]) => segmentHitsTriangle(p, q, a))
        ) {
          count++;
        }
      }
    }
    return count;
  }

  it.each(['l', 'r'])('rests the %s hand on the hip surface without penetration', (side) => {
    if (!joints || !weights) throw new Error('base.glb has no skin');
    const value = pose('hands-on-hips');
    // The neutral body (macro defaults) is wider at the hips than the raw base mesh: test the pose on that, with the
    // joint points moved by the same morphs.
    const morphFile = readFileSync(
      new URL('../../../public/assets/body/morphs.bin', import.meta.url),
    );
    const morphBytes = morphFile.buffer.slice(
      morphFile.byteOffset,
      morphFile.byteOffset + morphFile.byteLength,
    );
    const combined = buildBasePositions(position, manifest);
    const neutral = new Float32Array(combined.length);
    applyMorphs(combined, morphBytes, manifest, macroWeights(manifest, {}), neutral);
    const neutralPoint = (ref: JointRef) => {
      if (ref.strategy !== 'VERTEX') throw new Error('Expected a joint point');
      return new Vector3(
        neutral[3 * ref.vert]!,
        neutral[3 * ref.vert + 1]!,
        neutral[3 * ref.vert + 2]!,
      );
    };
    const state = fk(value, neutralPoint);
    const posed = skinPose(value, neutralPoint, neutral);
    const vertex = (positions: ArrayLike<number>, i: number) =>
      new Vector3(positions[3 * i]!, positions[3 * i + 1]!, positions[3 * i + 2]!);
    const dominantBones = Array.from({ length: vertexCount }, (_, i) => {
      let strongest = 0;
      for (let k = 1; k < 4; k++) {
        if (weights[4 * i + k]! > weights[4 * i + strongest]!) strongest = k;
      }
      return rig.bones[joints[4 * i + strongest]!]!.name;
    });
    const handIds = dominantBones.flatMap((name, i) =>
      /^(hand|thumb|index|middle|ring|pinky)_/.test(name) && name.endsWith(`_${side}`) ? [i] : [],
    );
    // Weld UV seam copies before computing the centroid of the whole hand, including the fingers.
    const uniqueHand = [
      ...new Map(handIds.map((i) => [vertex(position, i).toArray().join(','), i])).values(),
    ];
    expect(uniqueHand.length).toBeGreaterThan(100);
    const centroid = uniqueHand
      .reduce((sum, i) => sum.add(vertex(posed, i)), new Vector3())
      .divideScalar(uniqueHand.length);
    const hip = state.joint(`thigh_${side}`);
    expect(Math.abs(centroid.y - hip.y), 'hip height').toBeLessThan(0.15);
    expect(centroid.x * (side === 'l' ? 1 : -1), 'same-side hip').toBeGreaterThan(Math.abs(hip.x));

    // Only torso/pelvis/thigh triangles count as body contact: never the hand itself or another arm.
    const bodyTriangles: Triangle[] = [];
    for (let t = 0; t < mesh.indices.length; t += 3) {
      const a = mesh.indices[t]!;
      const b = mesh.indices[t + 1]!;
      const c = mesh.indices[t + 2]!;
      if ([a, b, c].every((i) => /^(pelvis|spine_|thigh_)/.test(dominantBones[i]!))) {
        bodyTriangles.push(new Triangle(vertex(posed, a), vertex(posed, b), vertex(posed, c)));
      }
    }
    expect(bodyTriangles.length).toBeGreaterThan(100);
    const signedDistance = (p: Vector3) => {
      let nearest = Infinity;
      let signed = Infinity;
      const closest = new Vector3();
      const normal = new Vector3();
      for (const triangle of bodyTriangles) {
        triangle.closestPointToPoint(p, closest);
        const distance = p.distanceTo(closest);
        if (distance < nearest) {
          nearest = distance;
          signed =
            p.clone().sub(closest).dot(triangle.getNormal(normal)) < 0 ? -distance : distance;
        }
      }
      return signed;
    };
    // A centroid lies inside the hand volume, so near-zero surface distance would bury the palm/fingers.
    const centroidDistance = signedDistance(centroid);
    expect(centroidDistance, 'hand centroid outside body').toBeGreaterThan(0);
    expect(centroidDistance, 'hand centroid to hip surface').toBeLessThan(0.07);
    const clearances = uniqueHand.map((i) => signedDistance(vertex(posed, i)));
    expect(Math.min(...clearances), 'no hand vertices inside body').toBeGreaterThanOrEqual(0);
    expect(Math.min(...clearances), 'hand surface close to the hip within 3 cm').toBeLessThan(0.03);

    // Real non-penetration: no hand triangle may cut through a pelvis / spine / thigh triangle. (The vertex test above
    // only looks at the nearest triangle's normal; the previous pose had ~110 intersecting triangle pairs per hand.)
    const handSet = new Set(handIds);
    const handTriangles: Triangle[] = [];
    for (let t = 0; t < mesh.indices.length; t += 3) {
      const a = mesh.indices[t]!;
      const b = mesh.indices[t + 1]!;
      const c = mesh.indices[t + 2]!;
      if (handSet.has(a) && handSet.has(b) && handSet.has(c)) {
        handTriangles.push(new Triangle(vertex(posed, a), vertex(posed, b), vertex(posed, c)));
      }
    }
    expect(handTriangles.length).toBeGreaterThan(100);
    expect(countIntersectingPairs(handTriangles, bodyTriangles), 'hand/body triangle pairs').toBe(
      0,
    );
  });

  /** Area of the 2D convex hull of the vertices projected on the plane perpendicular to `axis` (monotone chain). */
  function hullArea(positions: ArrayLike<number>, ids: number[], axis: Vector3): number {
    const u = new Vector3()
      .crossVectors(axis, Math.abs(axis.y) < 0.9 ? new Vector3(0, 1, 0) : new Vector3(1, 0, 0))
      .normalize();
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
        while (hull.length >= 2 && cross(hull[hull.length - 2]!, hull[hull.length - 1]!, pt) <= 0)
          hull.pop();
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
        const rel = new Vector3(position[3 * i]!, position[3 * i + 1]!, position[3 * i + 2]!).sub(
          wrist,
        );
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
        // Hands-on-hips must preserve at least 97% of every wrist slab, rather than just avoiding severe collapse.
        expect(ratio, `${id} ${side} slab ${k}`).toBeGreaterThanOrEqual(
          id === 'hands-on-hips' ? 0.97 : 0.8,
        );
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
