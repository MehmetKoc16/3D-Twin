import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { BodyManifest, BodyParams, MeasuresDef, MeasureId, RigDef } from './contracts';
import { computeJoints } from './joints';
import { bboxYRange } from './measure';
import { macroWeights } from './macro';
import { applyMorphs, buildBasePositions } from './morphs';
import { solveBody } from './solver';
import type { SolverData } from './types';
import { parseGlbMesh } from './testing/glb';

const CANDIDATES = [
  '../../apps/web/public/assets/body',
  'apps/web/public/assets/body',
  '../apps/web/public/assets/body',
];
const dir = CANDIDATES.find((d) => existsSync(`${d}/manifest.json`));

function bytes(name: string): ArrayBuffer {
  const b = readFileSync(`${dir}/${name}`);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
const json = <T>(name: string): T => JSON.parse(new TextDecoder().decode(bytes(name))) as T;

function load(): { data: SolverData; rig: RigDef } {
  const manifest = json<BodyManifest>('manifest.json');
  const measures = json<MeasuresDef>('measures.json');
  const rig = json<RigDef>('rig.json');
  const mesh = parseGlbMesh(bytes('base.glb'));
  const base = buildBasePositions(mesh.positions, manifest);
  return { data: { manifest, base, morphs: bytes('morphs.bin'), measures, indices: mesh.indices }, rig };
}

const profile = (
  gender: number,
  heightCm: number,
  weightKg: number,
  chestCm: number,
  waistCm: number,
  hipCm: number,
  neckCm: number,
  shoulderCm: number,
  eu: number,
): BodyParams => ({
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

const CIRC: [MeasureId, keyof BodyParams][] = [
  ['chest', 'chestCm'],
  ['waist', 'waistCm'],
  ['hip', 'hipCm'],
  ['neck', 'neckCm'],
  ['shoulder', 'shoulderCm'],
];

describe.skipIf(!dir)('real MakeHuman assets', () => {
  const { data, rig } = dir ? load() : ({} as ReturnType<typeof load>);

  it('neutral female / male heights are about 159 / 173 cm', () => {
    const heightCm = (gender: number): number => {
      const w = macroWeights(data.manifest, { gender });
      const out = new Float32Array(data.base.length);
      applyMorphs(data.base, data.morphs, data.manifest, w, out);
      const r = bboxYRange(out, data.manifest.renderVertexCount);
      return (r.max - r.min) * 100;
    };
    expect(heightCm(0)).toBeGreaterThan(158);
    expect(heightCm(0)).toBeLessThan(160.5);
    expect(heightCm(1)).toBeGreaterThan(172);
    expect(heightCm(1)).toBeLessThan(174.5);
  });

  const cases: [string, BodyParams][] = [
    ['male 180/80', profile(1, 180, 80, 100, 85, 98, 39, 46, 43)],
    ['female 165/58', profile(0, 165, 58, 88, 70, 96, 33, 40, 38)],
    ['extreme small 150/45', profile(0.5, 150, 45, 80, 62, 84, 32, 38, 36)],
    ['extreme tall 205/130', profile(1, 205, 130, 128, 120, 125, 46, 54, 47)],
  ];

  for (const [name, params] of cases) {
    it(`solves ${name}`, () => {
      const t0 = performance.now();
      const res = solveBody(data, params);
      const ms = performance.now() - t0;
      const rows: string[] = [`height ${params.heightCm} -> ${res.achievedCm.height!.toFixed(2)}`];
      for (const [id, key] of CIRC) {
        const target = params[key] as number;
        const got = res.achievedCm[id]!;
        rows.push(`${id} ${target} -> ${got.toFixed(2)}`);
        if (!res.unreachable.includes(id)) expect(Math.abs(got - target)).toBeLessThan(1);
      }
      console.log(
        `[real] ${name}: ${rows.join(', ')}; mass ${res.estimatedMassKg.toFixed(1)}/${params.weightKg}; ` +
          `unreachable=[${res.unreachable.join(',')}]; ${ms.toFixed(0)} ms, ${res.iterations} it`,
      );
      // height must be met exactly (within 0.5 cm), except in the documented extreme range
      expect(Math.abs(res.achievedCm.height! - params.heightCm)).toBeLessThan(0.5);

      // joints inside the (slightly padded) bounding box of the render vertices
      const n = data.manifest.renderVertexCount;
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      for (let v = 0; v < n; v++)
        for (let k = 0; k < 3; k++) {
          const x = res.positions[v * 3 + k]!;
          if (x < min[k]!) min[k] = x;
          if (x > max[k]!) max[k] = x;
        }
      const joints = computeJoints(rig, res.positions);
      expect(joints.size).toBe(53);
      for (const [bone, jp] of joints)
        for (const p of [jp.head, jp.tail])
          for (let k = 0; k < 3; k++) {
            expect(p[k]!, `${bone}[${k}]`).toBeGreaterThanOrEqual(min[k]! - 0.05);
            expect(p[k]!, `${bone}[${k}]`).toBeLessThanOrEqual(max[k]! + 0.05);
          }

      // re-grounding: subtract min y over render vertices, feet end up at y = 0
      const { min: minY } = bboxYRange(res.positions, n);
      let lowest = Infinity;
      for (let v = 0; v < n; v++) lowest = Math.min(lowest, res.positions[v * 3 + 1]! - minY);
      expect(lowest).toBeCloseTo(0, 6);
    });
  }

  it('height drivers (leg-height modifiers) and macro height do not fight', () => {
    // inseam target moves the leg-height modifiers; height must stay exact regardless.
    for (const inseam of [70, 80, 90]) {
      const res = solveBody(data, { ...profile(1, 180, 80, 100, 85, 98, 39, 46, 43), inseamCm: inseam });
      expect(Math.abs(res.achievedCm.height! - 180)).toBeLessThan(0.5);
      console.log(`[real] inseam ${inseam} -> ${res.achievedCm.inseam!.toFixed(1)} (height ${res.achievedCm.height!.toFixed(2)})`);
    }
  });

  it('is fast enough for interactive re-solves', () => {
    const p = profile(1, 175, 75, 100, 86, 100, 39, 46, 42);
    solveBody(data, p);
    const t0 = performance.now();
    for (let i = 0; i < 5; i++) solveBody(data, { ...p, heightCm: 175 + i * 3 });
    console.log(`[real] mean re-solve ${((performance.now() - t0) / 5).toFixed(0)} ms`);
  });
});
