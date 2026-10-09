import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { Matrix4, Quaternion, Vector3 } from 'three';
import { poseSpecs } from './poseSpecs.mjs';

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const bodyDir = path.join(webRoot, 'public/assets/body');
const poseDir = path.join(webRoot, 'public/assets/poses');
const [rig, manifest] = await Promise.all([
  readFile(path.join(bodyDir, 'rig.json'), 'utf8').then(JSON.parse),
  readFile(path.join(bodyDir, 'manifest.json'), 'utf8').then(JSON.parse),
]);
const bones = new Map(rig.bones.map((bone) => [bone.name, bone]));
const restPoint = (ref) => {
  if (ref.strategy !== 'VERTEX' || ref.vert < manifest.renderVertexCount) {
    throw new Error('Pose generation expects a joint point VERTEX reference');
  }
  const point = manifest.jointPoints[ref.vert - manifest.renderVertexCount];
  if (!point) throw new Error(`Missing joint point ${ref.vert}`);
  return new Vector3(...point.position);
};
const head = (name) => restPoint(bones.get(name).head);
const tail = (name) => restPoint(bones.get(name).tail);
const direction = (name) => tail(name).sub(head(name)).normalize();
const identity = () => new Quaternion();
const degrees = (angle) => (angle * Math.PI) / 180;
const rounded = (value) => Math.round(value * 1e6) / 1e6;

function createPose(id, label) {
  const local = new Map();
  const world = (name) => {
    const bone = bones.get(name);
    if (!bone) throw new Error(`Unknown bone ${name}`);
    const parent = bone.parent ? world(bone.parent) : identity();
    return parent.multiply(local.get(name) ?? identity());
  };
  const posedHead = (name) => {
    const bone = bones.get(name);
    if (!bone.parent) return head(name);
    return posedHead(bone.parent).add(
      head(name).sub(head(bone.parent)).applyQuaternion(world(bone.parent)),
    );
  };
  const setWorldRotation = (name, desiredWorld) => {
    const parent = bones.get(name).parent;
    const parentWorld = parent ? world(parent) : identity();
    local.set(name, parentWorld.invert().multiply(desiredWorld).normalize());
  };
  const aim = (name, desiredDirection) => {
    setWorldRotation(
      name,
      new Quaternion().setFromUnitVectors(direction(name), desiredDirection.clone().normalize()),
    );
  };
  const setLocal = (name, rotation) => local.set(name, rotation.clone().normalize());
  const getLocal = (name) => (local.get(name) ?? identity()).clone();
  const output = () => {
    const result = {};
    for (const bone of rig.bones) {
      const q = local.get(bone.name);
      if (!q || q.angleTo(identity()) < 1e-7) continue;
      // q and -q are the same rotation: keep w >= 0 so left and right quaternions mirror component-wise
      const values = (q.w < 0 ? [-q.x, -q.y, -q.z, -q.w] : [q.x, q.y, q.z, q.w]).map(rounded);
      if (Math.abs(Math.hypot(...values) - 1) > 2e-6) {
        throw new Error(`Non-unit rotation ${bone.name}`);
      }
      result[bone.name] = values;
    }
    return { id, label, bones: result };
  };
  return { posedHead, aim, setWorldRotation, setLocal, getLocal, output };
}

/** Left-hand-frame vector (character's left = +X) expressed on the given side: the right side mirrors X. */
const onSide = (side, vector) =>
  side === 'l' ? vector.clone() : new Vector3(-vector.x, vector.y, vector.z);

/**
 * Unit normal pointing OUT of the rest palm. `fingers x (pinky -> index)` is the outward normal of the LEFT palm; the right
 * hand is its mirror image, so the cross product flips sign there. (The first generator used the raw cross product on
 * both sides and mirrored the target twice, which put both palms ~165 deg away from the thighs.)
 */
function restPalmNormal(side) {
  const fingers = head(`middle_01_${side}`).sub(head(`hand_${side}`));
  const towardThumb = head(`index_01_${side}`).sub(head(`pinky_01_${side}`));
  const normal = fingers.cross(towardThumb).normalize();
  return side === 'l' ? normal : normal.negate();
}

/** Rotation of the frame (forward, normal, forward x normal); `normal` is projected perpendicular to `forward`. */
function frameRotation(forward, normal) {
  const f = forward.clone().normalize();
  const n = normal.clone().projectOnPlane(f).normalize();
  return new Quaternion().setFromRotationMatrix(
    new Matrix4().makeBasis(f, n, new Vector3().crossVectors(f, n)),
  );
}

/** Twist angle (radians, -pi..pi) of q about the unit `axis` (swing-twist decomposition, q = swing * twist). */
function twistAngle(q, axis) {
  let angle = 2 * Math.atan2(axis.x * q.x + axis.y * q.y + axis.z * q.z, q.w);
  if (angle > Math.PI) angle -= 2 * Math.PI;
  if (angle < -Math.PI) angle += 2 * Math.PI;
  return angle;
}

/**
 * Linear blend skinning has no twist bones on this 53-bone rig: a roll of the hand about the forearm axis is absorbed by
 * the few cm where the lowerarm weights hand over to the hand weights, and the ring there shrinks to cos(roll / 2).
 * Rolling the lowerarm by half of the difference between the wrist roll and the elbow roll spreads the same hand
 * orientation over both transitions (elbow and wrist), so neither pinches.
 */
function balanceForearmRoll(pose, side, desiredHandWorld) {
  const lower = `lowerarm_${side}`;
  const axis = direction(lower);
  const wristRoll = twistAngle(pose.getLocal(`hand_${side}`), axis);
  const elbowRoll = twistAngle(pose.getLocal(lower), axis);
  const share = new Quaternion().setFromAxisAngle(axis, (wristRoll - elbowRoll) / 2);
  pose.setLocal(lower, pose.getLocal(lower).multiply(share));
  // the hand keeps its world orientation, only its local rotation (relative to the rolled forearm) shrinks
  pose.setWorldRotation(`hand_${side}`, desiredHandWorld);
}

/**
 * Orients the hand so that its `reference` axis points along `desiredDirection` and the palm faces `desiredPalmNormal`
 * (both in left-hand world coordinates, mirrored for the right side). `reference` is the hand bone axis (T-pose, where
 * the finger bones are aimed explicitly) or the wrist-to-knuckle axis of the visible hand (`'fingers'`).
 */
function orientHand(pose, side, desiredDirection, desiredPalmNormal, reference = 'bone') {
  const restAxis =
    reference === 'fingers'
      ? head(`middle_01_${side}`).sub(head(`hand_${side}`))
      : direction(`hand_${side}`);
  const restFrame = frameRotation(restAxis, restPalmNormal(side));
  const targetFrame = frameRotation(desiredDirection, onSide(side, desiredPalmNormal));
  const handWorld = targetFrame.multiply(restFrame.invert());
  pose.setWorldRotation(`hand_${side}`, handWorld);
  balanceForearmRoll(pose, side, handWorld);
}

function straightFingers(pose, side) {
  const sign = side === 'l' ? 1 : -1;
  const spread = { index: 0.10, middle: 0.035, ring: -0.035, pinky: -0.10 };
  for (const [finger, forward] of Object.entries(spread)) {
    for (const segment of [1, 2, 3]) {
      pose.aim(`${finger}_0${segment}_${side}`, new Vector3(sign, 0, forward));
    }
  }
  for (const segment of [1, 2, 3]) {
    pose.aim(`thumb_0${segment}_${side}`, new Vector3(sign * 0.4, 0, 0.9));
  }
}

function curlFingers(pose) {
  for (const side of ['l', 'r']) {
    for (const finger of ['index', 'middle', 'ring', 'pinky', 'thumb']) {
      for (const [segment, angle] of [
        [1, 5],
        [2, 9],
        [3, 7],
      ]) {
        pose.setLocal(
          `${finger}_0${segment}_${side}`,
          new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), degrees(angle)),
        );
      }
    }
  }
}

function relaxedArms(pose, intent) {
  for (const side of ['l', 'r']) {
    const sign = side === 'l' ? 1 : -1;
    const outward = degrees(intent.upperArmOutwardDeg);
    const forward = intent.upperArmForward[side];
    pose.aim(
      `clavicle_${side}`,
      new Vector3(
        sign * Math.cos(degrees(intent.clavicleDropDeg)),
        -Math.sin(degrees(intent.clavicleDropDeg)),
        -0.05,
      ),
    );
    pose.aim(
      `upperarm_${side}`,
      new Vector3(sign * Math.sin(outward), -Math.cos(outward), forward),
    );
    pose.aim(
      `lowerarm_${side}`,
      new Vector3(
        sign * Math.sin(outward - degrees(1)),
        -Math.cos(outward),
        forward + Math.sin(degrees(intent.elbowFlexDeg)),
      ),
    );
    // fingers hang down and slightly forward, the palm faces the thigh (-X for the left hand, mirrored for the right)
    orientHand(pose, side, new Vector3(sign * 0.08, -0.94, 0.3), new Vector3(-1, 0, 0), 'fingers');
  }
  curlFingers(pose);
}

function hipArms(pose, intent) {
  for (const side of ['l', 'r']) {
    const sign = side === 'l' ? 1 : -1;
    pose.aim(
      `clavicle_${side}`,
      new Vector3(
        sign * Math.cos(degrees(intent.clavicleDropDeg)),
        -Math.sin(degrees(intent.clavicleDropDeg)),
        -0.05,
      ),
    );
    const shoulder = pose.posedHead(`upperarm_${side}`);
    const hip = head(`thigh_${side}`);
    const wrist = new Vector3(
      sign * intent.wristX,
      hip.y + intent.wristAboveHipM,
      hip.z + intent.wristForwardM,
    );
    const reach = wrist.clone().sub(shoulder);
    const distance = reach.length();
    const upperLength = tail(`upperarm_${side}`).distanceTo(head(`upperarm_${side}`));
    const lowerLength = tail(`lowerarm_${side}`).distanceTo(head(`lowerarm_${side}`));
    if (distance > upperLength + lowerLength || distance < Math.abs(upperLength - lowerLength)) {
      throw new Error(`Hip target is unreachable on ${side}`);
    }
    const axis = reach.clone().normalize();
    const along = (upperLength ** 2 - lowerLength ** 2 + distance ** 2) / (2 * distance);
    const outward = new Vector3(sign, 0, 0).addScaledVector(axis, -axis.x * sign).normalize();
    const offset = Math.sqrt(Math.max(0, upperLength ** 2 - along ** 2));
    const elbow = shoulder.clone().addScaledVector(axis, along).addScaledVector(outward, offset);
    pose.aim(`upperarm_${side}`, elbow.clone().sub(shoulder));
    pose.aim(`lowerarm_${side}`, wrist.clone().sub(elbow));
    pose.aim(`hand_${side}`, new Vector3(-sign * 0.25, -0.75, 0.3));
  }
  curlFingers(pose);
}

function walkLegs(pose, intent) {
  for (const [side, thighAngle, calfAngle] of [
    ['l', intent.leftThighForwardDeg, intent.leftThighForwardDeg - intent.leftKneeFlexDeg],
    ['r', -intent.rightThighBackDeg, -intent.rightThighBackDeg + intent.rightKneeFlexDeg],
  ]) {
    pose.aim(
      `thigh_${side}`,
      new Vector3(0, -Math.cos(degrees(thighAngle)), Math.sin(degrees(thighAngle))),
    );
    pose.aim(
      `calf_${side}`,
      new Vector3(0, -Math.cos(degrees(calfAngle)), Math.sin(degrees(calfAngle))),
    );
    pose.aim(`foot_${side}`, new Vector3(0, -0.47, 0.88));
  }
}

function build(spec) {
  const pose = createPose(spec.id, spec.label);
  switch (spec.id) {
    case 't-pose':
      for (const side of ['l', 'r']) {
        const sign = side === 'l' ? 1 : -1;
        for (const part of ['upperarm', 'lowerarm']) {
          pose.aim(`${part}_${side}`, new Vector3(sign, 0.045, 0));
        }
        orientHand(pose, side, new Vector3(sign, 0, 0), new Vector3(0, -1, 0.35), 'fingers');
        straightFingers(pose, side);
      }
      break;
    case 'a-pose':
      break;
    case 'relaxed':
      relaxedArms(pose, spec.intent);
      break;
    case 'hands-on-hips':
      hipArms(pose, spec.intent);
      break;
    case 'walk':
      relaxedArms(pose, spec.intent);
      walkLegs(pose, spec.intent);
      break;
    case 'side':
      relaxedArms(pose, spec.intent);
      pose.setLocal(
        'Root',
        new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), degrees(spec.intent.rootYawDeg)),
      );
      break;
    default:
      throw new Error(`No generator for ${spec.id}`);
  }
  return pose.output();
}

await mkdir(poseDir, { recursive: true });
for (const spec of poseSpecs) {
  await writeFile(
    path.join(poseDir, `${spec.id}.json`),
    `${JSON.stringify(build(spec), null, 2)}\n`,
  );
}
await writeFile(
  path.join(poseDir, 'index.json'),
  `${JSON.stringify(
    poseSpecs.map(({ id, label }) => ({ id, label })),
    null,
    2,
  )}\n`,
);
