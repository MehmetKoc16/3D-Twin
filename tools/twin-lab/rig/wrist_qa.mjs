// Wrist / forearm cross-section QA of a rigged GLB under the app's pose library (headless three.js skinning, the same
// maths as PoseDriver + check_pose.mjs).
//
// node wrist_qa.mjs <rigged.glb|base.glb> [--poses=relaxed,a-pose,...] [--pose-dir=<dir>] [--json=out.json] [--png=<dir>]
//
// For each pose and arm it reports
//   * twist     : swing-twist angle (degrees, about the rest forearm axis) of the hand and lowerarm local rotations,
//                 plus the palm-normal direction in world space and its angle to the inward (thigh-facing) direction;
//   * sections  : cross-section AREA of the forearm / wrist / hand skin, posed divided by rest, on planes perpendicular
//                 to the posed arm polyline at s = 0.35 .. 1.30 (s = 0 elbow, 1 wrist, > 1 along the hand);
//   * summary   : areaAtWrist (s = 1), minForearm (min ratio for s in 0.5..1.2), volumeRatio (forearm s in 0.35..1.0).
// A candy-wrapper wrist shows as a small areaAtWrist / minForearm. The twin and the mannequin are comparable: both
// are skinned to the same 53 bones and take the same poses.
// --png=<dir> also writes a smooth-shaded orthographic crop of the hands (front and side) per pose, a quick look at the
// wrists and palm directions without a browser. Hands and bodies only, no faces.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import zlib from 'node:zlib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Quaternion, Vector3 } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../../..');
const arg = (name) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const glbPath = process.argv[2];
if (!glbPath || glbPath.startsWith('--'))
  throw new Error(
    'usage: wrist_qa.mjs <rigged.glb> [--poses=a,b] [--pose-dir=dir] [--json=out.json]',
  );
const poseDir = arg('pose-dir') ?? path.join(repo, 'apps/web/public/assets/poses');
const poseIds = (arg('poses') ?? 'a-pose,relaxed,t-pose,walk,hands-on-hips,side').split(',');
const jsonOut = arg('json');
const pngDir = arg('png');

/** Drop textures/images so GLTFLoader can parse in Node (no image decoding needed for skinning maths). */
function stripImages(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(Buffer.from(buf.buffer, buf.byteOffset + 20, jsonLen).toString('utf8'));
  delete json.images;
  delete json.textures;
  delete json.samplers;
  for (const m of json.materials ?? []) {
    delete m.normalTexture;
    delete m.occlusionTexture;
    delete m.emissiveTexture;
    if (m.pbrMetallicRoughness) {
      delete m.pbrMetallicRoughness.baseColorTexture;
      delete m.pbrMetallicRoughness.metallicRoughnessTexture;
    }
  }
  let js = Buffer.from(JSON.stringify(json), 'utf8');
  js = Buffer.concat([js, Buffer.alloc((4 - (js.length % 4)) % 4, 0x20)]);
  const rest = buf.subarray(20 + jsonLen);
  const head = Buffer.alloc(20);
  head.write('glTF', 0, 'ascii');
  head.writeUInt32LE(2, 4);
  head.writeUInt32LE(12 + 8 + js.length + rest.length, 8);
  head.writeUInt32LE(js.length, 12);
  head.writeUInt32LE(0x4e4f534a, 16);
  const out = Buffer.concat([head, js, rest]);
  return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength);
}

const raw = await readFile(glbPath);
const gltf = await new Promise((resolve, reject) =>
  new GLTFLoader().parse(stripImages(raw), '', resolve, reject),
);
const meshes = [];
gltf.scene.traverse((o) => o.isSkinnedMesh && meshes.push(o));
// the body is the largest non-hair skinned mesh (the twin glb also carries the dtHair node)
const mesh = meshes
  .filter((m) => !/hair/i.test(m.name))
  .sort((a, b) => b.geometry.attributes.position.count - a.geometry.attributes.position.count)[0];
if (!mesh) throw new Error('no body skinned mesh');
const bones = new Map(mesh.skeleton.bones.map((b) => [b.name, b]));
const index = mesh.geometry.index.array;
const vertexCount = mesh.geometry.attributes.position.count;

function applyPose(pose) {
  for (const b of bones.values()) b.quaternion.set(0, 0, 0, 1);
  if (pose)
    for (const [n, q] of Object.entries(pose.bones))
      bones.get(n)?.quaternion.set(q[0], q[1], q[2], q[3]);
  gltf.scene.updateMatrixWorld(true);
}
const bonePos = (n) => bones.get(n).getWorldPosition(new Vector3());
const boneQuat = (n) => bones.get(n).getWorldQuaternion(new Quaternion());
function skinned() {
  const out = new Float32Array(vertexCount * 3);
  const v = new Vector3();
  for (let i = 0; i < vertexCount; i++) {
    mesh.getVertexPosition(i, v);
    out[3 * i] = v.x;
    out[3 * i + 1] = v.y;
    out[3 * i + 2] = v.z;
  }
  return out;
}

applyPose(null);
const restPos = skinned();
const rest = {};
for (const n of bones.keys()) rest[n] = bonePos(n);

const deg = (r) => (r * 180) / Math.PI;
/** Twist angle (deg, in -180..180) of q about the unit axis. */
function twistAbout(q, axis) {
  const s = axis.x * q.x + axis.y * q.y + axis.z * q.z;
  let t = 2 * Math.atan2(s, q.w);
  if (t > Math.PI) t -= 2 * Math.PI;
  if (t < -Math.PI) t += 2 * Math.PI;
  return deg(t);
}

/** Per-arm geometry that does not depend on the pose. */
function armSetup(side) {
  const elbow = rest[`lowerarm_${side}`];
  const wrist = rest[`hand_${side}`];
  const knuckle = rest[`middle_01_${side}`];
  const forearmLen = elbow.distanceTo(wrist);
  const fAxis = wrist.clone().sub(elbow).normalize();
  const hAxis = knuckle.clone().sub(wrist).normalize();
  // palm normal (rest), out of the palm: fingers x (pinky -> index) is the left palm's outward normal; the right hand
  // is the mirror image, so the cross product flips
  const f = hAxis;
  const t = rest[`index_01_${side}`].clone().sub(rest[`pinky_01_${side}`]);
  const palmN = new Vector3().crossVectors(f, t).normalize();
  if (side === 'r') palmN.negate();
  // region: vertices of the forearm (s 0.25..1.6) within 8 cm of the arm polyline in the rest pose
  const inRegion = new Uint8Array(vertexCount);
  const p = new Vector3();
  for (let i = 0; i < vertexCount; i++) {
    p.set(restPos[3 * i], restPos[3 * i + 1], restPos[3 * i + 2]);
    const rel = p.clone().sub(elbow);
    const s1 = rel.dot(fAxis) / forearmLen;
    if (s1 < 0.25) continue;
    let s;
    let radial;
    if (s1 <= 1) {
      s = s1;
      radial = rel.clone().addScaledVector(fAxis, -rel.dot(fAxis)).length();
    } else {
      const relH = p.clone().sub(wrist);
      s = 1 + relH.dot(hAxis) / forearmLen;
      radial = relH.clone().addScaledVector(hAxis, -relH.dot(hAxis)).length();
    }
    if (s < 0.25 || s > 1.65 || radial > 0.08) continue;
    inRegion[i] = 1;
  }
  const tris = [];
  for (let t3 = 0; t3 < index.length; t3 += 3) {
    if (inRegion[index[t3]] && inRegion[index[t3 + 1]] && inRegion[index[t3 + 2]]) tris.push(t3);
  }
  return { side, elbow, wrist, knuckle, forearmLen, fAxis, hAxis, palmN, tris };
}

/** Cross-section area and perimeter of the triangles `tris` on the plane (point c, normal n). */
function section(pos, tris, c, n) {
  const up = Math.abs(n.y) < 0.9 ? new Vector3(0, 1, 0) : new Vector3(1, 0, 0);
  const u = new Vector3().crossVectors(n, up).normalize();
  const v = new Vector3().crossVectors(n, u).normalize();
  const d = new Float64Array(3);
  const P = [new Vector3(), new Vector3(), new Vector3()];
  let area2 = 0;
  let perimeter = 0;
  const ptU = new Vector3();
  const ptD = new Vector3();
  for (const t3 of tris) {
    let neg = 0;
    for (let k = 0; k < 3; k++) {
      const i = index[t3 + k];
      P[k].set(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
      d[k] = P[k].clone().sub(c).dot(n);
      if (d[k] < 0) neg++;
    }
    if (neg === 0 || neg === 3) continue;
    let haveU = false;
    let haveD = false;
    for (let k = 0; k < 3; k++) {
      const a = k;
      const b = (k + 1) % 3;
      if (d[a] < 0 === d[b] < 0) continue;
      const tt = d[a] / (d[a] - d[b]);
      const q = P[a].clone().lerp(P[b], tt);
      if (d[a] < 0) {
        ptU.copy(q);
        haveU = true;
      } else {
        ptD.copy(q);
        haveD = true;
      }
    }
    if (!haveU || !haveD) continue;
    const ax = ptU.clone().sub(c);
    const bx = ptD.clone().sub(c);
    area2 += ax.dot(u) * bx.dot(v) - ax.dot(v) * bx.dot(u);
    perimeter += ptU.distanceTo(ptD);
  }
  return { area: Math.abs(area2) / 2, perimeter };
}

const sGrid = [];
for (let s = 0.35; s <= 1.3001; s += 0.05) sGrid.push(+s.toFixed(2));

function sections(arm, pos, posed) {
  const L = arm.forearmLen;
  const out = [];
  for (const s of sGrid) {
    let c;
    let n;
    if (s < 0.97) {
      c = posed.elbow.clone().addScaledVector(posed.wrist.clone().sub(posed.elbow), s);
      n = posed.fAxis;
    } else if (s <= 1.03) {
      c = posed.wrist.clone();
      n = posed.fAxis.clone().add(posed.hAxis).normalize();
    } else {
      c = posed.wrist.clone().addScaledVector(posed.hAxis, (s - 1) * L);
      n = posed.hAxis;
    }
    out.push({ s, ...section(pos, arm.tris, c, n) });
  }
  return out;
}

function posedGeometry(arm) {
  const elbow = bonePos(`lowerarm_${arm.side}`);
  const wrist = bonePos(`hand_${arm.side}`);
  const knuckle = bonePos(`middle_01_${arm.side}`);
  return {
    elbow,
    wrist,
    knuckle,
    fAxis: wrist.clone().sub(elbow).normalize(),
    hAxis: knuckle.clone().sub(wrist).normalize(),
  };
}

// ---------------------------------------------------------------------------------------------- PNG preview
function crc32(buf) {
  return zlib.crc32(buf) >>> 0;
}
function pngBytes(width, height, rgb) {
  const chunk = (type, data) => {
    const out = Buffer.alloc(12 + data.length);
    out.writeUInt32BE(data.length, 0);
    out.write(type, 4, 'ascii');
    data.copy(out, 8);
    out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
    return out;
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const raw = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 3 + 1)] = 0;
    rgb.copy(raw, y * (width * 3 + 1) + 1, y * width * 3, (y + 1) * width * 3);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw, { level: 6 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

/**
 * Smooth-shaded orthographic z-buffer render of the triangles with at least one vertex inside the world box. Everything
 * above the neck joint (rest pose) is left out: the crops are for hands and arms only, never for faces.
 */
function renderCrop(pos, box, view, size) {
  const neckY = rest.neck_01.y;
  const { min, max } = box;
  // view: 'front' camera at +z (screen right = +x), 'side' camera at +x (screen right = -z), 'back' camera at -z,
  // 'top' camera at +y (screen right = +x, screen up = -z: the character's front is at the bottom)
  const project = {
    front: (x, y, z) => [x, y, z],
    side: (x, y, z) => [-z, y, x],
    back: (x, y, z) => [-x, y, -z],
    top: (x, y, z) => [x, -z, y],
  }[view];
  const toCam = (i) => project(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
  const cMin = project(min.x, min.y, min.z);
  const cMax = project(max.x, max.y, max.z);
  const u0 = Math.min(cMin[0], cMax[0]);
  const u1 = Math.max(cMin[0], cMax[0]);
  const v0 = Math.min(cMin[1], cMax[1]);
  const v1 = Math.max(cMin[1], cMax[1]);
  const scale = size / Math.max(u1 - u0, v1 - v0);
  const W = Math.round((u1 - u0) * scale);
  const H = Math.round((v1 - v0) * scale);
  // vertex normals of the whole posed mesh (area weighted)
  const nrm = new Float32Array(vertexCount * 3);
  for (let t = 0; t < index.length; t += 3) {
    const a = index[t];
    const b = index[t + 1];
    const c = index[t + 2];
    const ux = pos[3 * b] - pos[3 * a];
    const uy = pos[3 * b + 1] - pos[3 * a + 1];
    const uz = pos[3 * b + 2] - pos[3 * a + 2];
    const wx = pos[3 * c] - pos[3 * a];
    const wy = pos[3 * c + 1] - pos[3 * a + 1];
    const wz = pos[3 * c + 2] - pos[3 * a + 2];
    const nx = uy * wz - uz * wy;
    const ny = uz * wx - ux * wz;
    const nz = ux * wy - uy * wx;
    for (const i of [a, b, c]) {
      nrm[3 * i] += nx;
      nrm[3 * i + 1] += ny;
      nrm[3 * i + 2] += nz;
    }
  }
  const inBox = (i) =>
    pos[3 * i] >= min.x &&
    pos[3 * i] <= max.x &&
    pos[3 * i + 1] >= min.y &&
    pos[3 * i + 1] <= max.y &&
    pos[3 * i + 2] >= min.z &&
    pos[3 * i + 2] <= max.z;
  const depth = new Float32Array(W * H).fill(-1e9);
  const rgb = Buffer.alloc(W * H * 3, 28);
  const shade = (i) => {
    const n = project(nrm[3 * i], 0, nrm[3 * i + 2]);
    const m = project(0, nrm[3 * i + 1], 0);
    const cam = [n[0] + m[0], n[1] + m[1], n[2] + m[2]];
    const l = Math.hypot(...cam) || 1;
    return Math.max(0, cam[2] / l);
  };
  for (let t = 0; t < index.length; t += 3) {
    const ids = [index[t], index[t + 1], index[t + 2]];
    if (!ids.some(inBox) || ids.some((i) => restPos[3 * i + 1] > neckY)) continue;
    const P = ids.map((i) => {
      const [u, v, d] = toCam(i);
      return [(u - u0) * scale, (v1 - v) * scale, d, shade(i)];
    });
    const minX = Math.max(0, Math.floor(Math.min(P[0][0], P[1][0], P[2][0])));
    const maxX = Math.min(W - 1, Math.ceil(Math.max(P[0][0], P[1][0], P[2][0])));
    const minY = Math.max(0, Math.floor(Math.min(P[0][1], P[1][1], P[2][1])));
    const maxY = Math.min(H - 1, Math.ceil(Math.max(P[0][1], P[1][1], P[2][1])));
    const den =
      (P[1][1] - P[2][1]) * (P[0][0] - P[2][0]) + (P[2][0] - P[1][0]) * (P[0][1] - P[2][1]);
    if (Math.abs(den) < 1e-9) continue;
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const px = x + 0.5;
        const py = y + 0.5;
        const l0 =
          ((P[1][1] - P[2][1]) * (px - P[2][0]) + (P[2][0] - P[1][0]) * (py - P[2][1])) / den;
        const l1 =
          ((P[2][1] - P[0][1]) * (px - P[2][0]) + (P[0][0] - P[2][0]) * (py - P[2][1])) / den;
        const l2 = 1 - l0 - l1;
        if (l0 < -1e-4 || l1 < -1e-4 || l2 < -1e-4) continue;
        const d = l0 * P[0][2] + l1 * P[1][2] + l2 * P[2][2];
        if (d <= depth[y * W + x]) continue;
        depth[y * W + x] = d;
        const sh = 0.25 + 0.75 * (l0 * P[0][3] + l1 * P[1][3] + l2 * P[2][3]);
        const o = (y * W + x) * 3;
        rgb[o] = Math.round(222 * sh);
        rgb[o + 1] = Math.round(178 * sh);
        rgb[o + 2] = Math.round(150 * sh);
      }
    }
  }
  return pngBytes(W, H, rgb);
}

const arms = ['l', 'r'].map(armSetup);
const restSections = {};
for (const arm of arms) {
  const g = {
    elbow: arm.elbow,
    wrist: arm.wrist,
    knuckle: arm.knuckle,
    fAxis: arm.fAxis,
    hAxis: arm.hAxis,
  };
  restSections[arm.side] = sections(arm, restPos, g);
}

const results = { file: path.basename(glbPath), mesh: mesh.name, vertices: vertexCount, poses: {} };
const r3 = (x) => +x.toFixed(3);
for (const id of poseIds) {
  const pose =
    id === 'rest' ? null : JSON.parse(await readFile(path.join(poseDir, `${id}.json`), 'utf8'));
  applyPose(pose);
  const pos = skinned();
  const entry = {};
  for (const arm of arms) {
    const side = arm.side;
    const posed = posedGeometry(arm);
    const secs = sections(arm, pos, posed);
    const ratio = secs.map((x, i) => ({
      s: x.s,
      area: r3(x.area / restSections[side][i].area),
      perimeter: r3(x.perimeter / restSections[side][i].perimeter),
    }));
    const at = (s) => ratio.find((x) => Math.abs(x.s - s) < 1e-6);
    const minForearm = Math.min(
      ...ratio.filter((x) => x.s >= 0.5 && x.s <= 1.2).map((x) => x.area),
    );
    const vol = (list) =>
      list.filter((x) => x.s >= 0.35 && x.s <= 1.0001).reduce((acc, x) => acc + x.area, 0);
    const volumeRatio = vol(secs) / vol(restSections[side]);
    // twist bookkeeping: local rotations are parent-relative and the rest frames are world-aligned
    const qHand = bones.get(`hand_${side}`).quaternion;
    const qLower = bones.get(`lowerarm_${side}`).quaternion;
    const qUpper = bones.get(`upperarm_${side}`).quaternion;
    const worldHand = boneQuat(`hand_${side}`);
    const palmWorld = arm.palmN.clone().applyQuaternion(worldHand);
    const inward = new Vector3(side === 'l' ? -1 : 1, 0, 0);
    entry[side] = {
      wristBendDeg: r3(deg(posed.fAxis.angleTo(posed.hAxis))),
      restWristBendDeg: r3(deg(arm.fAxis.angleTo(arm.hAxis))),
      forearmAxisWorld: posed.fAxis.toArray().map(r3),
      fingerAxisWorld: posed.hAxis.toArray().map(r3),
      twistDeg: {
        hand: r3(twistAbout(qHand, arm.fAxis)),
        lowerarm: r3(twistAbout(qLower, arm.fAxis)),
        upperarm: r3(
          twistAbout(
            qUpper,
            rest[`lowerarm_${side}`].clone().sub(rest[`upperarm_${side}`]).normalize(),
          ),
        ),
      },
      handLocalAngleDeg: r3(deg(2 * Math.acos(Math.min(1, Math.abs(qHand.w))))),
      palmNormalWorld: palmWorld.toArray().map(r3),
      palmToInwardDeg: r3(deg(palmWorld.angleTo(inward))),
      areaAtWrist: at(1).area,
      perimeterAtWrist: at(1).perimeter,
      minForearm: r3(minForearm),
      volumeRatio: r3(volumeRatio),
      areaProfile: ratio.map((x) => [x.s, x.area]),
    };
  }
  results.poses[id] = entry;
  if (pngDir) {
    await mkdir(pngDir, { recursive: true });
    const tag = path.basename(glbPath, '.glb').replace(/[^a-z0-9]+/gi, '_');
    const wristL = bonePos('hand_l');
    const wristR = bonePos('hand_r');
    const lo = new Vector3(
      Math.min(wristL.x, wristR.x) - 0.2,
      Math.min(wristL.y, wristR.y) - 0.28,
      Math.min(wristL.z, wristR.z) - 0.3,
    );
    const hi = new Vector3(
      Math.max(wristL.x, wristR.x) + 0.2,
      Math.max(wristL.y, wristR.y) + 0.32,
      Math.max(wristL.z, wristR.z) + 0.3,
    );
    await writeFile(
      path.join(pngDir, `${tag}-${id}-front.png`),
      renderCrop(pos, { min: lo, max: hi }, 'front', 760),
    );
    const sideBox = (w) => ({
      min: new Vector3(w.x - 0.25, w.y - 0.28, w.z - 0.3),
      max: new Vector3(w.x + 0.25, w.y + 0.32, w.z + 0.3),
    });
    await writeFile(
      path.join(pngDir, `${tag}-${id}-left-side.png`),
      renderCrop(pos, sideBox(wristL), 'side', 560),
    );
    const knuckleL = bonePos('middle_01_l');
    const handBox = {
      min: new Vector3(knuckleL.x - 0.14, knuckleL.y - 0.14, knuckleL.z - 0.14),
      max: new Vector3(knuckleL.x + 0.14, knuckleL.y + 0.14, knuckleL.z + 0.14),
    };
    for (const view of ['front', 'side', 'top'])
      await writeFile(
        path.join(pngDir, `${tag}-${id}-left-hand-${view}.png`),
        renderCrop(pos, handBox, view, 420),
      );
  }
  const f = (e) =>
    `wrist area x${e.areaAtWrist.toFixed(2)} min x${e.minForearm.toFixed(2)} vol x${e.volumeRatio.toFixed(2)} twist hand ${e.twistDeg.hand}/lower ${e.twistDeg.lowerarm} palm->inward ${e.palmToInwardDeg} deg wrist bend ${e.wristBendDeg} (rest ${e.restWristBendDeg})`;
  console.log(`${id.padEnd(14)} L ${f(entry.l)}`);
  console.log(`${''.padEnd(14)} R ${f(entry.r)}`);
}
if (jsonOut) await writeFile(jsonOut, JSON.stringify(results, null, 1));
