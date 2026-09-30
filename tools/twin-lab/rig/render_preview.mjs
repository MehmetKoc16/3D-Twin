// Headless preview renderer: loads a rigged GLB in Chromium (playwright, software GL), applies our pose JSONs the way
// PoseDriver does (bone.quaternion.set from apps/web/public/assets/poses/<id>.json) and writes PNG contact sheets.
//
// node render_preview.mjs <rigged.glb> <outDir> [prefix] [--poses=rest,t-pose,walk,hands-on-hips]
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../../..');
const require = createRequire(path.join(repo, 'package.json'));
const { chromium } = require('playwright');

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const opts = Object.fromEntries(
  process.argv
    .slice(2)
    .filter((a) => a.startsWith('--'))
    .map((a) => a.slice(2).split('=')),
);
const [glbPath, outDir, prefix = 'rig'] = args;
if (!glbPath || !outDir)
  throw new Error('usage: render_preview.mjs <rigged.glb> <outDir> [prefix]');
const poseIds = (opts.poses ?? 'rest,t-pose,walk,hands-on-hips').split(',');
const { mkdir, writeFile } = await import('node:fs/promises');
await mkdir(outDir, { recursive: true });

const html = `<!doctype html><meta charset="utf-8"><body style="margin:0;background:#dcdfe4">
<canvas id="c" width="1800" height="700"></canvas>
<script type="importmap">{"imports":{"three":"/three/build/three.module.js","three/addons/":"/three/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdcdfe4);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(1.5, 3, 3); scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.8); fill.position.set(-2, 1, -2); scene.add(fill);
const gltf = await new GLTFLoader().loadAsync('/model.glb');
scene.add(gltf.scene);
let skinned; gltf.scene.traverse((o) => { if (o.isSkinnedMesh) skinned = o; });
skinned.frustumCulled = false;
const bones = new Map(skinned.skeleton.bones.map((b) => [b.name, b]));
const restQ = new Map([...bones].map(([n, b]) => [n, b.quaternion.clone()]));
async function applyPose(id) {
  for (const [n, b] of bones) b.quaternion.copy(restQ.get(n));
  if (id !== 'rest') {
    const p = await (await fetch('/poses/' + id + '.json')).json();
    for (const [n, q] of Object.entries(p.bones)) bones.get(n)?.quaternion.set(q[0], q[1], q[2], q[3]);
  }
  scene.updateMatrixWorld(true);
}
const cam = new THREE.PerspectiveCamera(30, 1, 0.05, 50);
function panel(i, n, target, dist, azimuthDeg, elevDeg = 8, fov = 30) {
  const w = canvas.width / n, h = canvas.height;
  renderer.setViewport(i * w, 0, w, h); renderer.setScissor(i * w, 0, w, h); renderer.setScissorTest(true);
  cam.fov = fov; cam.aspect = w / h; cam.updateProjectionMatrix();
  const az = azimuthDeg * Math.PI / 180, el = elevDeg * Math.PI / 180;
  cam.position.set(target.x + dist * Math.sin(az) * Math.cos(el), target.y + dist * Math.sin(el), target.z + dist * Math.cos(az) * Math.cos(el));
  cam.lookAt(target);
  renderer.render(scene, cam);
}
const wp = (name, dx = 0, dy = 0, dz = 0) => bones.get(name).getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(dx, dy, dz));
window.renderSheet = async (poseId, kind) => {
  await applyPose(poseId);
  skinned.computeBoundingBox();
  const bb = skinned.boundingBox.clone().applyMatrix4(skinned.matrixWorld);
  const c = bb.getCenter(new THREE.Vector3()); const size = bb.getSize(new THREE.Vector3());
  renderer.setScissorTest(false); renderer.clear();
  const d = Math.max(size.y * 1.15, (size.x * 1.1) / (canvas.width / 3 / canvas.height)) / (2 * Math.tan(15 * Math.PI / 180));
  if (kind === 'body') {
    panel(0, 3, c, d, 0); panel(1, 3, c, d, 40); panel(2, 3, c, d, 90);
  } else if (kind === 'focus') {
    const t = wp(window.focusBone);
    for (let i = 0; i < 4; i++) panel(i, 4, t, 0.7, i * 90, 20, 30);
  } else if (kind === 'detail') {
    // shoulder+armpit front, shoulder from above-back, hand, crotch front, crotch side
    const n = 5;
    panel(0, n, wp('upperarm_l', -0.05, -0.05), 0.75, 20, 5, 30);
    panel(1, n, wp('upperarm_l', 0, 0), 0.75, 150, 25, 30);
    panel(2, n, wp('hand_l'), 0.5, 0, 25, 30);
    panel(3, n, wp('pelvis', 0, -0.1), 1.0, 0, 5, 30);
    panel(4, n, wp('pelvis', 0, -0.1), 1.0, 90, 5, 30);
  }
  await new Promise((r) => requestAnimationFrame(() => r()));
  return true;
};
window.ready = true;
</script>`;

const server = http.createServer(async (req, res) => {
  try {
    const url = decodeURIComponent(req.url.split('?')[0]);
    let file;
    if (url === '/' || url === '/index.html') {
      res.writeHead(200, { 'content-type': 'text/html' });
      return res.end(html);
    }
    if (url === '/model.glb') file = glbPath;
    else if (url.startsWith('/three/')) file = path.join(repo, 'node_modules/three', url.slice(7));
    else if (url.startsWith('/poses/'))
      file = path.join(repo, 'apps/web/public/assets/poses', url.slice(7));
    else {
      res.writeHead(404);
      return res.end();
    }
    const buf = await readFile(file);
    const type =
      file.endsWith('.js') || file.endsWith('.mjs')
        ? 'text/javascript'
        : file.endsWith('.json')
          ? 'application/json'
          : 'application/octet-stream';
    res.writeHead(200, { 'content-type': type });
    res.end(buf);
  } catch (e) {
    res.writeHead(500);
    res.end(String(e));
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const browser = await chromium.launch({
  args: [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--ignore-gpu-blocklist',
  ],
});
const page = await browser.newPage({ viewport: { width: 1800, height: 700 } });
page.on('console', (m) => {
  if (m.type() === 'error') console.error('[page]', m.text());
});
page.on('pageerror', (e) => console.error('[pageerror]', e.message));
await page.goto(`http://127.0.0.1:${port}/`);
await page.waitForFunction('window.ready === true', null, { timeout: 120000 });
for (const id of poseIds) {
  for (const kind of opts.focus ? ['focus'] : ['body', 'detail']) {
    if (kind === 'detail' && id !== 't-pose' && id !== 'walk' && id !== 'hands-on-hips') continue;
    await page.evaluate(
      ([p, k, fb]) => {
        window.focusBone = fb;
        return window.renderSheet(p, k);
      },
      [id, kind, opts.focus ?? ''],
    );
    const file = path.join(outDir, `${prefix}_${id}_${kind}.png`);
    await page.locator('#c').screenshot({ path: file });
    console.log('wrote', file);
  }
}
await browser.close();
server.close();
