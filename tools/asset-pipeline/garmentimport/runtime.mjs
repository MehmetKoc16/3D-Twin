// Read-only bridge to the real avatar-core solver, grading, fit report and heatmap.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createJiti } from 'jiti';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const jiti = createJiti(import.meta.url, { fsCache: false });
const core = await jiti.import(path.join(root, 'packages/avatar-core/src/index.ts'));
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const bodyDir = input.bodyDir;
const manifest = JSON.parse(fs.readFileSync(path.join(bodyDir, 'manifest.json'), 'utf8'));
const measures = JSON.parse(fs.readFileSync(path.join(bodyDir, 'measures.json'), 'utf8'));
const morphBytes = fs.readFileSync(path.join(bodyDir, 'morphs.bin'));
const morphs = morphBytes.buffer.slice(
  morphBytes.byteOffset,
  morphBytes.byteOffset + morphBytes.byteLength,
);
const bindingBytes = fs.readFileSync(input.bindingPath);
const binding = core.parseGarmentBinding(
  bindingBytes.buffer.slice(
    bindingBytes.byteOffset,
    bindingBytes.byteOffset + bindingBytes.byteLength,
  ),
  manifest.renderVertexCount,
);
const solver = core.createBodySolver({
  manifest,
  measures,
  morphs,
  base: new Float32Array(input.base.flat()),
  indices: input.faces.flat(),
});
// These are the realData.test.ts profiles; there is no exported small/medium/large preset catalogue.
const profile = (gender, heightCm, weightKg, chestCm, waistCm, hipCm, neckCm, shoulderCm, eu) => ({
  gender,
  heightCm,
  weightKg,
  chestCm,
  waistCm,
  hipCm,
  neckCm,
  shoulderCm,
  shoe: { system: 'EU', size: eu },
});
const profiles = {
  small: profile(0.5, 150, 45, 80, 62, 84, 32, 38, 36),
  medium: profile(0, 165, 58, 88, 70, 96, 33, 40, 38),
  large: profile(1, 180, 80, 100, 85, 98, 39, 46, 43),
};
const result = {};
for (const [name, params] of Object.entries(profiles)) {
  const solved = solver.solve(params);
  const body = solved.positions;
  const bound = new Float32Array(binding.count * 3);
  core.bindGarment(binding, input.template.scaleRefs, body, bound);
  const rings = [];
  for (const id of ['chest', 'waist', 'hip', 'thigh']) {
    const native = input.template.nativeMeasures[id];
    const achieved = solved.achievedCm[id];
    const definition = measures.measures.find((m) => m.id === id);
    if (native === undefined || achieved === undefined || !definition?.verts) continue;
    rings.push({
      planeY: definition.verts.reduce((s, v) => s + body[v * 3 + 1], 0) / definition.verts.length,
      deltaCircumferenceM: (native - (achieved + (input.template.defaultEase[id] ?? 0))) / 100,
    });
  }
  const graded = new Float32Array(bound.length);
  core.gradeGarment(bound, binding, body, rings, graded);
  const clearance = new Float32Array(binding.count);
  core.garmentClearance(graded, binding, body, clearance);
  const item = {
    id: 'validation',
    name: 'Validation',
    templateId: input.template.id,
    color: input.template.baseColor,
    sizes: ['native'],
    selectedSize: 'native',
    chart: Object.fromEntries(
      Object.entries(input.template.nativeMeasures).map(([id, value]) => [id, [value]]),
    ),
  };
  const report = core.analyzeFit(
    item,
    input.template,
    core.bodyMeasuresForGarment(solved.achievedCm),
  );
  const colors = new Float32Array(3);
  core.clearanceToColor(Math.min(...clearance), colors);
  result[name] = {
    body: Array.from(body),
    bound: Array.from(bound),
    graded: Array.from(graded),
    achievedCm: solved.achievedCm,
    unreachable: solved.unreachable,
    fitReport: report,
    heatmapMinMm: Math.min(...clearance) * 1000,
    heatmapMaxMm: Math.max(...clearance) * 1000,
    heatmapFinite:
      Array.from(clearance).every(Number.isFinite) && Array.from(colors).every(Number.isFinite),
  };
}
fs.writeFileSync(process.argv[3], JSON.stringify(result));
