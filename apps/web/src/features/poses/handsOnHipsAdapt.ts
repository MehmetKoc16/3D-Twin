import { Matrix4, Quaternion, Vector3, Bone } from 'three';
import type { JointRef, PoseDef, Quat } from '@dt/avatar-core';

/**
 * Standard joint vertex indices for arm and hip reference bones in MakeHuman game_engine rig.
 */
const BONE_HEAD_VERTS: Record<string, number> = {
  clavicle_l: 14521,
  upperarm_l: 14547,
  lowerarm_l: 14522,
  hand_l: 14544,
  middle_01_l: 14531,
  index_01_l: 14527,
  pinky_01_l: 14539,
  thigh_l: 14549,

  clavicle_r: 14553,
  upperarm_r: 14579,
  lowerarm_r: 14554,
  hand_r: 14576,
  middle_01_r: 14563,
  index_01_r: 14559,
  pinky_01_r: 14571,
  thigh_r: 14581,
};

const BONE_TAIL_VERTS: Record<string, number> = {
  clavicle_l: 14547,
  upperarm_l: 14522,
  lowerarm_l: 14544,
  hand_l: 14545,

  clavicle_r: 14579,
  upperarm_r: 14554,
  lowerarm_r: 14576,
  hand_r: 14577,
};

/** Default neutral hip surface half-width in meters. */
export const DEFAULT_NEUTRAL_HIP_HALF_WIDTH = 0.157;

/**
 * Measures the maximum lateral half-width of the torso/pelvis surface in the contact zone
 * around the wrist target height (hip.y + 0.17 m) and forward depth (hip.z + 0.065 m).
 * Both left (+X) and right (-X) sides are checked to ensure symmetric results.
 */
export function computeHipHalfWidth(
  positions: ArrayLike<number>,
  hip: { y: number; z: number },
): number {
  const targetY = hip.y + 0.17;
  const targetZ = hip.z + 0.065;
  // Bounding slice across the iliac crest and waist
  const yMin = targetY - 0.05;
  const yMax = targetY + 0.03;
  const zMin = targetZ - 0.04;
  const zMax = targetZ + 0.08;

  let maxAbsX = 0;
  const renderCount = Math.min(Math.floor(positions.length / 3), 14517);
  for (let i = 0; i < renderCount; i++) {
    const absX = Math.abs(positions[3 * i]!);
    // Lateral torso/hip window: excludes spine midline and hanging arms in A-pose
    if (absX < 0.08 || absX > 0.3) continue;
    const y = positions[3 * i + 1]!;
    if (y < yMin || y > yMax) continue;
    const z = positions[3 * i + 2]!;
    if (z < zMin || z > zMax) continue;
    if (absX > maxAbsX) maxAbsX = absX;
  }
  // Fallback to broader window if mesh has unexpected vertical offset
  if (maxAbsX === 0) {
    for (let i = 0; i < renderCount; i++) {
      const absX = Math.abs(positions[3 * i]!);
      if (absX < 0.05 || absX > 0.35) continue;
      const y = positions[3 * i + 1]!;
      if (Math.abs(y - targetY) > 0.12) continue;
      if (absX > maxAbsX) maxAbsX = absX;
    }
  }
  return maxAbsX > 0 ? maxAbsX : DEFAULT_NEUTRAL_HIP_HALF_WIDTH;
}

/**
 * Computes world rest position of a Three.js bone by summing local translations up the hierarchy
 * (since all MakeHuman bones have identity rest rotation).
 */
export function getBoneRestHead(bone: { position: Vector3; parent: unknown }): Vector3 {
  const p = new Vector3();
  let curr: Bone | null = bone as Bone;
  while (curr && curr.isBone) {
    p.add(curr.position);
    curr = curr.parent instanceof Bone ? curr.parent : null;
  }
  return p;
}

interface JointResolver {
  head(name: string): Vector3;
  tail(name: string): Vector3;
}

function createJointResolver(options: HandsOnHipsAdaptOptions): JointResolver | null {
  if (options.pointOf) {
    const pointOf = options.pointOf;
    return {
      head: (name) => {
        const v = BONE_HEAD_VERTS[name];
        if (v === undefined) throw new Error(`Unknown bone head ${name}`);
        return pointOf({ strategy: 'VERTEX', vert: v });
      },
      tail: (name) => {
        const v = BONE_TAIL_VERTS[name];
        if (v === undefined) throw new Error(`Unknown bone tail ${name}`);
        return pointOf({ strategy: 'VERTEX', vert: v });
      },
    };
  }
  if (options.bones) {
    const bones = options.bones;
    const childTail: Record<string, string> = {
      clavicle_l: 'upperarm_l',
      upperarm_l: 'lowerarm_l',
      lowerarm_l: 'hand_l',
      hand_l: 'middle_01_l',
      clavicle_r: 'upperarm_r',
      upperarm_r: 'lowerarm_r',
      lowerarm_r: 'hand_r',
      hand_r: 'middle_01_r',
    };
    return {
      head: (name) => {
        const b = bones.get(name);
        if (!b) throw new Error(`Missing bone ${name}`);
        return getBoneRestHead(b);
      },
      tail: (name) => {
        const childName = childTail[name];
        if (!childName) throw new Error(`Missing child for tail ${name}`);
        const childBone = bones.get(childName);
        if (!childBone) throw new Error(`Missing bone ${childName}`);
        return getBoneRestHead(childBone);
      },
    };
  }
  return null;
}

function onSide(side: string, vector: Vector3): Vector3 {
  return side === 'l' ? vector.clone() : new Vector3(-vector.x, vector.y, vector.z);
}

function restPalmNormal(resolver: JointResolver, side: string): Vector3 {
  const fingers = resolver
    .head(`middle_01_${side}`)
    .clone()
    .sub(resolver.head(`hand_${side}`));
  const towardThumb = resolver
    .head(`index_01_${side}`)
    .clone()
    .sub(resolver.head(`pinky_01_${side}`));
  const normal = fingers.cross(towardThumb).normalize();
  return side === 'l' ? normal : normal.negate();
}

function frameRotation(forward: Vector3, normal: Vector3): Quaternion {
  const f = forward.clone().normalize();
  const n = normal.clone().projectOnPlane(f).normalize();
  return new Quaternion().setFromRotationMatrix(
    new Matrix4().makeBasis(f, n, new Vector3().crossVectors(f, n)),
  );
}

function twistAngle(q: Quaternion, axis: Vector3): number {
  let angle = 2 * Math.atan2(axis.x * q.x + axis.y * q.y + axis.z * q.z, q.w);
  if (angle > Math.PI) angle -= 2 * Math.PI;
  if (angle < -Math.PI) angle += 2 * Math.PI;
  return angle;
}

export interface HandsOnHipsAdaptOptions {
  /** Morphed vertex positions (render vertices or combined). */
  positions?: ArrayLike<number> | null;
  /** Function returning rest position of a joint ref. */
  pointOf?: (ref: JointRef) => Vector3;
  /** Skeleton bones from useAvatarRuntimeStore. */
  bones?: ReadonlyMap<string, { position: Vector3; parent: unknown }>;
  /** Direct wrist target X distance in meters (bypasses half-width calculation if given). */
  wristX?: number;
  /** Pre-calculated hip half-width, if available. */
  hipHalfWidth?: number;
  /** Pre-calculated neutral hip half-width reference. */
  neutralHipHalfWidth?: number;
}

/**
 * Adapts hands-on-hips arm and hand rotations to match the current body shape:
 * 1. Computes the horizontal half-width of the hip/waist region from mesh surface vertices.
 * 2. Shifts the wrist target outward/inward so hand clearance stays within 0.5–2 cm without penetration.
 * 3. Solves clavicle drop, outward two-bone IK (upperarm & lowerarm), palm normal alignment, and roll balance.
 * 4. Preserves exact mirror symmetry between left and right arms.
 */
export function adaptHandsOnHipsPose(basePose: PoseDef, options: HandsOnHipsAdaptOptions): PoseDef {
  const resolver = createJointResolver(options);
  if (!resolver) return basePose;

  const shoulderRestX = Math.abs(resolver.head('upperarm_l').x);
  // Female shapes have narrower shoulders (shoulderRestX <= 0.155 m) and wider hips.
  // Smoothly blend femaleFactor: 0 (male) to 1 (female).
  const femaleFactor = Math.max(0, Math.min(1, (0.178 - shoulderRestX) / 0.028));

  const neutralWidth = options.neutralHipHalfWidth ?? DEFAULT_NEUTRAL_HIP_HALF_WIDTH;
  let hipHalfWidth = neutralWidth;
  if (options.hipHalfWidth !== undefined && Number.isFinite(options.hipHalfWidth)) {
    hipHalfWidth = options.hipHalfWidth;
  } else if (options.positions) {
    const hip = resolver.head('thigh_l');
    const measuredWidth = computeHipHalfWidth(options.positions, hip);
    if (measuredWidth > 0) {
      hipHalfWidth = measuredWidth;
    }
  }

  let wristX =
    options.wristX !== undefined && Number.isFinite(options.wristX)
      ? options.wristX
      : 0.19 + (hipHalfWidth - neutralWidth);

  // Adjust inward clearance offset based on hip and shoulder body measurements:
  // Slender female shapes retain a 23 mm baseline shift (keeping >= 0.5 cm clearance),
  // while wider hips receive an additional ~6.8 mm inward push (reaching ~1.5 cm clearance)
  // without penetrating narrow hips or displacing male hands.
  const wideHipFactor = Math.max(0, Math.min(1, (hipHalfWidth - (neutralWidth - 0.009)) / 0.012));
  const femaleClearanceShift = (0.023 + 0.0068 * wideHipFactor) * femaleFactor;
  wristX -= femaleClearanceShift;
  // Clamp to safe human biomechanical reach range [14 cm, 26 cm]
  wristX = Math.max(0.14, Math.min(0.26, wristX));

  const adaptedBones: Record<string, Quat> = { ...basePose.bones };

  for (const side of ['l', 'r'] as const) {
    const sign = side === 'l' ? 1 : -1;
    const clavicleDropDeg = 12;
    const desiredClavicleDir = new Vector3(
      sign * Math.cos((clavicleDropDeg * Math.PI) / 180),
      -Math.sin((clavicleDropDeg * Math.PI) / 180),
      -0.05,
    ).normalize();
    const restClavicleDir = resolver
      .tail(`clavicle_${side}`)
      .clone()
      .sub(resolver.head(`clavicle_${side}`))
      .normalize();
    const clavicleWorld = new Quaternion().setFromUnitVectors(restClavicleDir, desiredClavicleDir);
    const clavicleLocal = clavicleWorld.clone();

    const clavHead = resolver.head(`clavicle_${side}`);
    const shoulderRest = resolver.head(`upperarm_${side}`);
    const shoulder = clavHead
      .clone()
      .add(shoulderRest.clone().sub(clavHead).applyQuaternion(clavicleWorld));

    const hip = resolver.head(`thigh_${side}`);
    const wrist = new Vector3(sign * wristX, hip.y + 0.17, hip.z + 0.065);

    const reach = wrist.clone().sub(shoulder);
    const upperLength = resolver
      .tail(`upperarm_${side}`)
      .distanceTo(resolver.head(`upperarm_${side}`));
    const lowerLength = resolver
      .tail(`lowerarm_${side}`)
      .distanceTo(resolver.head(`lowerarm_${side}`));

    const maxReach = (upperLength + lowerLength) * 0.999;
    if (reach.length() > maxReach) {
      reach.multiplyScalar(maxReach / reach.length());
      wrist.copy(shoulder).add(reach);
    }
    const distance = reach.length();
    const axis = reach.clone().normalize();
    const along = (upperLength ** 2 - lowerLength ** 2 + distance ** 2) / (2 * distance);
    const outwardDir = new Vector3(
      sign * (1.2 + 0.6 * femaleFactor),
      0,
      -(3.0 + 0.5 * femaleFactor),
    );
    const outward = outwardDir.projectOnPlane(axis).normalize();
    const offset = Math.sqrt(Math.max(0, upperLength ** 2 - along ** 2));
    const elbow = shoulder.clone().addScaledVector(axis, along).addScaledVector(outward, offset);

    const restUpperDir = resolver
      .tail(`upperarm_${side}`)
      .clone()
      .sub(resolver.head(`upperarm_${side}`))
      .normalize();
    const desiredUpperDir = elbow.clone().sub(shoulder).normalize();
    const upperarmWorld = new Quaternion().setFromUnitVectors(restUpperDir, desiredUpperDir);
    const upperarmLocal = clavicleWorld.clone().invert().multiply(upperarmWorld).normalize();

    const restLowerDir = resolver
      .tail(`lowerarm_${side}`)
      .clone()
      .sub(resolver.head(`lowerarm_${side}`))
      .normalize();
    const forearmDir = wrist.clone().sub(elbow).normalize();
    const lowerarmWorld = new Quaternion().setFromUnitVectors(restLowerDir, forearmDir);
    let lowerarmLocal = upperarmWorld.clone().invert().multiply(lowerarmWorld).normalize();

    const restHandAxis = resolver
      .head(`middle_01_${side}`)
      .clone()
      .sub(resolver.head(`hand_${side}`));
    const restNormal = restPalmNormal(resolver, side);
    const restFrame = frameRotation(restHandAxis, restNormal);
    const palmTiltY = -0.16 + 0.08 * femaleFactor;
    const targetFrame = frameRotation(forearmDir, onSide(side, new Vector3(-1, palmTiltY, 0)));
    const desiredHandWorld = targetFrame.clone().multiply(restFrame.clone().invert()).normalize();

    let handLocal = lowerarmWorld.clone().invert().multiply(desiredHandWorld).normalize();
    const rollAxis = restLowerDir.clone();
    const wristRoll = twistAngle(handLocal, rollAxis);
    const elbowRoll = twistAngle(lowerarmLocal, rollAxis);
    let shareAngle = (wristRoll - elbowRoll) / 2;
    const maxRollRad = (28 * Math.PI) / 180;
    if (elbowRoll + shareAngle > maxRollRad) {
      shareAngle = maxRollRad - elbowRoll;
    } else if (elbowRoll + shareAngle < -maxRollRad) {
      shareAngle = -maxRollRad - elbowRoll;
    }
    if (wristRoll - shareAngle > maxRollRad) {
      shareAngle = wristRoll - maxRollRad;
    } else if (wristRoll - shareAngle < -maxRollRad) {
      shareAngle = wristRoll + maxRollRad;
    }
    const share = new Quaternion().setFromAxisAngle(rollAxis, shareAngle);
    lowerarmLocal = lowerarmLocal.clone().multiply(share).normalize();

    const rolledLowerarmWorld = upperarmWorld.clone().multiply(lowerarmLocal);
    handLocal = rolledLowerarmWorld.clone().invert().multiply(desiredHandWorld).normalize();

    const toQuat = (q: Quaternion): Quat => {
      const norm = q.clone().normalize();
      // Keep w >= 0 so left and right quaternions mirror component-wise
      const sign = norm.w < 0 ? -1 : 1;
      return [
        Math.round(norm.x * sign * 1e6) / 1e6 || 0,
        Math.round(norm.y * sign * 1e6) / 1e6 || 0,
        Math.round(norm.z * sign * 1e6) / 1e6 || 0,
        Math.round(norm.w * sign * 1e6) / 1e6 || 0,
      ];
    };

    adaptedBones[`clavicle_${side}`] = toQuat(clavicleLocal);
    adaptedBones[`upperarm_${side}`] = toQuat(upperarmLocal);
    adaptedBones[`lowerarm_${side}`] = toQuat(lowerarmLocal);
    adaptedBones[`hand_${side}`] = toQuat(handLocal);
  }

  return {
    ...basePose,
    bones: adaptedBones,
  };
}
