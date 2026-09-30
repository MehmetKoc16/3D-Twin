// Headless three.js sanity check of a rigged GLB with our pose library (same math as PoseDriver: local bone
// quaternions from apps/web/public/assets/poses/<id>.json set on the skeleton bones, three's skinning).
//
// node check_pose.mjs <rigged.glb> [--json=out.json]
// Exit code 1 when a hard check fails (bone names, NaNs, T-pose geometry).
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Vector3 } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../../..');
const glbPath = process.argv[2];
const jsonOut = process.argv.find((a) => a.startsWith('--json='))?.slice(7);
if (!glbPath) throw new Error('usage: check_pose.mjs <rigged.glb> [--json=out.json]');

/** Drop textures/images so GLTFLoader can parse in Node (no image decoding needed for skinning maths). */
function stripImages(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(Buffer.from(buf.buffer, buf.byteOffset + 20, jsonLen).toString('utf8'));
  if (!json.images && !json.textures)
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
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
const rig = JSON.parse(
  await readFile(path.join(repo, 'apps/web/public/assets/body/rig.json'), 'utf8'),
);
const results = { file: path.basename(glbPath), checks: [], poses: {} };
let failed = false;
const check = (name, ok, detail = '') => {
  results.checks.push({ name, ok, detail });
  if (!ok) failed = true;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};

check('has skinned mesh', meshes.length > 0, `${meshes.length} mesh(es)`);
const mesh = meshes[0];
const bones = new Map(mesh.skeleton.bones.map((b) => [b.name, b]));
const expected = rig.bones.map((b) => b.name);
check(
  'bone names identical to rig.json (53)',
  expected.length === bones.size && expected.every((n) => bones.has(n)),
  `${bones.size} bones`,
);
check(
  'bone parents identical to rig.json',
  rig.bones.every(
    (b) => (bones.get(b.name).parent?.isBone ? bones.get(b.name).parent.name : null) === b.parent,
  ),
);
const restId = [...bones.values()].every((b) => Math.abs(b.quaternion.w - 1) < 1e-9);
check('identity rest rotations (world-aligned)', restId);

// weights sanity
for (const m of meshes) {
  const w = m.geometry.attributes.skinWeight;
  let bad = 0;
  for (let i = 0; i < w.count; i++) {
    const s = w.getX(i) + w.getY(i) + w.getZ(i) + w.getW(i);
    if (!(Math.abs(s - 1) < 1e-3)) bad++;
  }
  check(`skin weights sum to 1 (${m.name})`, bad === 0, `${bad} bad of ${w.count}`);
}

async function loadPose(id) {
  return JSON.parse(
    await readFile(path.join(repo, `apps/web/public/assets/poses/${id}.json`), 'utf8'),
  );
}
function applyPose(pose) {
  for (const b of bones.values()) b.quaternion.set(0, 0, 0, 1);
  if (pose)
    for (const [n, q] of Object.entries(pose.bones))
      bones.get(n)?.quaternion.set(q[0], q[1], q[2], q[3]);
  gltf.scene.updateMatrixWorld(true);
}
const wp = (n) => bones.get(n).getWorldPosition(new Vector3());
function skinnedPositions(m) {
  const n = m.geometry.attributes.position.count;
  const out = new Float32Array(n * 3);
  const v = new Vector3();
  for (let i = 0; i < n; i++) {
    m.getVertexPosition(i, v);
    out[3 * i] = v.x;
    out[3 * i + 1] = v.y;
    out[3 * i + 2] = v.z;
  }
  return out;
}
let restHeadPos = [];
function edgeStats(m, rest, posed) {
  const idx = m.geometry.index.array;
  const seen = new Set();
  const ratios = [];
  const bad = [];
  const n = rest.length / 3;
  const len = (a, i, j) =>
    Math.hypot(a[3 * i] - a[3 * j], a[3 * i + 1] - a[3 * j + 1], a[3 * i + 2] - a[3 * j + 2]);
  for (let t = 0; t < idx.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = idx[t + e];
      const b = idx[t + ((e + 1) % 3)];
      const key = a < b ? a * n + b : b * n + a;
      if (seen.has(key)) continue;
      seen.add(key);
      const l0 = len(rest, a, b);
      if (l0 < 1e-5) continue;
      const rr = len(posed, a, b) / l0;
      ratios.push(rr);
      bad.push([a, b, rr]);
    }
  }
  const restHeads = mesh.skeleton.bones.map((b) => b.name);
  const nearest = (i) => {
    let best = 0,
      bd = Infinity;
    for (let k = 0; k < restHeadPos.length; k++) {
      const d = Math.hypot(
        rest[3 * i] - restHeadPos[k][0],
        rest[3 * i + 1] - restHeadPos[k][1],
        rest[3 * i + 2] - restHeadPos[k][2],
      );
      if (d < bd) {
        bd = d;
        best = k;
      }
    }
    return restHeads[best];
  };
  const hot = {};
  for (const [a, b, r] of bad) {
    if (r > 2 || r < 0.5) {
      const nb = nearest(a);
      hot[nb] = (hot[nb] ?? 0) + 1;
    }
  }
  edgeStats.lastHot = Object.entries(hot)
    .sort((x, y) => y[1] - x[1])
    .slice(0, 6);
  ratios.sort((x, y) => x - y);
  const q = (p) => ratios[Math.min(ratios.length - 1, Math.floor(p * ratios.length))];
  return {
    edges: ratios.length,
    p01: +q(0.01).toFixed(3),
    p99: +q(0.99).toFixed(3),
    max: +ratios[ratios.length - 1].toFixed(3),
    min: +ratios[0].toFixed(3),
    over2: ratios.filter((r) => r > 2).length,
    hot: edgeStats.lastHot,
    under05: ratios.filter((r) => r < 0.5).length,
  };
}

applyPose(null);
restHeadPos = [...bones.values()].map((b) => b.getWorldPosition(new Vector3()).toArray());
const restPos = skinnedPositions(mesh);
let minY = Infinity;
let maxY = -Infinity;
for (let i = 1; i < restPos.length; i += 3) {
  minY = Math.min(minY, restPos[i]);
  maxY = Math.max(maxY, restPos[i]);
}
const height = maxY - minY;
check(
  'rest: feet on y=0, plausible height',
  Math.abs(minY) < 0.02 && height > 1.3 && height < 2.2,
  `minY ${minY.toFixed(3)} height ${height.toFixed(3)} m`,
);

for (const id of ['a-pose', 't-pose', 'walk', 'hands-on-hips', 'relaxed', 'side']) {
  const pose = await loadPose(id);
  applyPose(pose);
  const posed = skinnedPositions(mesh);
  const nan = posed.some((x) => !Number.isFinite(x));
  const joints = Object.fromEntries(
    [...bones.keys()].map((n) => [
      n,
      wp(n)
        .toArray()
        .map((x) => +x.toFixed(4)),
    ]),
  );
  const r = { nan, edge: edgeStats(mesh, restPos, posed) };
  check(`${id}: no NaN in skinned vertices`, !nan);
  if (id === 't-pose') {
    const sl = wp('upperarm_l');
    const hl = wp('hand_l');
    const sr = wp('upperarm_r');
    const hr = wp('hand_r');
    r.handHeightOffsetL = +(hl.y - sl.y).toFixed(4);
    r.handHeightOffsetR = +(hr.y - sr.y).toFixed(4);
    r.armSpan = +Math.abs(hl.x - hr.x).toFixed(3);
    check(
      't-pose: hands at shoulder height (|dy| < 0.06 m)',
      Math.abs(hl.y - sl.y) < 0.06 && Math.abs(hr.y - sr.y) < 0.06,
      `dy L ${r.handHeightOffsetL} R ${r.handHeightOffsetR}`,
    );
    check(
      't-pose: arms horizontal outwards (|dx| > 0.4 m each side)',
      hl.x - sl.x > 0.4 && sr.x - hr.x > 0.4,
    );
    check(
      't-pose: arm span ~ body height (0.65..1.2)',
      r.armSpan / height > 0.65 && r.armSpan / height < 1.2,
      `${r.armSpan} / ${height.toFixed(2)}`,
    );
  }
  if (id === 'walk') {
    const fl = wp('foot_l');
    const fr = wp('foot_r');
    r.footStepZ = +Math.abs(fl.z - fr.z).toFixed(3);
    check('walk: feet apart in z (> 0.08 m)', r.footStepZ > 0.08, `${r.footStepZ} m`);
  }
  r.selectedJoints = Object.fromEntries(
    ['pelvis', 'head', 'hand_l', 'hand_r', 'foot_l', 'foot_r'].map((n) => [n, joints[n]]),
  );
  console.log(
    `      ${id}: edge stretch p01 ${r.edge.p01} p99 ${r.edge.p99} min ${r.edge.min} max ${r.edge.max} (>2x: ${r.edge.over2}, <0.5x: ${r.edge.under05}) hot: ${JSON.stringify(r.edge.hot)}`,
  );
  results.poses[id] = r;
  if (id === 't-pose') results.tposeJoints = joints;
}
if (jsonOut) await writeFile(jsonOut, JSON.stringify(results, null, 1));
console.log(failed ? 'RESULT: FAIL' : 'RESULT: OK');
process.exit(failed ? 1 : 0);
