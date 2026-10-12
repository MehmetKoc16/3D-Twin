// Read-only bridge: run the exported `<id>.rigid.json` through the app's own avatar-core sidecar parser and
// preparation (parseGarmentRigidAttachments / prepareGarmentRigidAttachments). Throws when the app would reject it.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createJiti } from 'jiti';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const jiti = createJiti(import.meta.url, { fsCache: false });
const core = await jiti.import(path.join(root, 'packages/avatar-core/src/index.ts'));
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const sidecar = JSON.parse(fs.readFileSync(input.rigidPath, 'utf8'));
const neutral = new Float32Array(input.neutral);
const vertexCount = neutral.length / 3;
const parsed = core.parseGarmentRigidAttachments(sidecar, vertexCount);
const prepared = core.prepareGarmentRigidAttachments(sidecar, neutral, input.index, input.bones);
let attached = 0;
let blended = 0;
for (let v = 0; v < vertexCount; v++) {
  if (prepared.bones[v] >= 0) {
    if (prepared.blend[v] === 1) attached++;
    else blended++;
  }
}
// Rebuild the sole at the neutral scale: it must reproduce the exported GLB positions.
const joints = new Map();
for (const [bone, head] of Object.entries(input.heads)) joints.set(input.bones.indexOf(bone), head);
const positions = neutral.slice();
core.fitGarmentRigidAttachments(prepared, [1, 1, 1], (i) => joints.get(i), positions);
let maxErrorM = 0;
for (let i = 0; i < positions.length; i++)
  maxErrorM = Math.max(maxErrorM, Math.abs(positions[i] - neutral[i]));
fs.writeFileSync(
  process.argv[3],
  JSON.stringify({
    ok: true,
    soleHeightMm: parsed.soleHeightMm,
    attachments: parsed.attachments.map((a) => ({ bone: a.bone, vertices: a.vertices.length })),
    rigidVertices: attached,
    bandVertices: blended,
    neutralRebuildMaxErrorM: maxErrorM,
  }),
);
