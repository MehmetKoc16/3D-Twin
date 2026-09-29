import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { Quaternion, Vector3 } from 'three';
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
  const output = () => {
    const result = {};
    for (const bone of rig.bones) {
      const q = local.get(bone.name);
      if (!q || q.angleTo(identity()) < 1e-7) continue;
      const values = [q.x, q.y, q.z, q.w].map(rounded);
      if (Math.abs(Math.hypot(...values) - 1) > 2e-6) {
        throw new Error(`Non-unit rotation ${bone.name}`);
      }
      result[bone.name] = values;
    }
    return { id, label, bones: result };
  };
  return { posedHead, aim, setWorldRotation, setLocal, output };
}

function horizontalHand(pose, side) {
  const sign = side === 'l' ? 1 : -1;
  const axis = new Vector3(sign, 0, 0);
  const swing = new Quaternion().setFromUnitVectors(direction(`hand_${side}`), axis);
  // Palm plane from wrist, middle knuckle, and index-to-pinky knuckle span.
  const palmNormal = head(`middle_01_${side}`)
    .sub(head(`hand_${side}`))
    .cross(head(`pinky_01_${side}`).sub(head(`index_01_${side}`)))
    .normalize();
  const swungNormal = palmNormal
    .applyQuaternion(swing)
    .addScaledVector(axis, -palmNormal.dot(axis))
    .normalize();
  const twist = new Quaternion().setFromUnitVectors(swungNormal, new Vector3(0, -1, 0));
  pose.setWorldRotation(`hand_${side}`, twist.multiply(swing));
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
    pose.aim(`hand_${side}`, new Vector3(sign * 0.08, -0.94, 0.3));
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
          pose.aim(`${part}_${side}`, new Vector3(sign, 0, 0));
        }
        horizontalHand(pose, side);
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
