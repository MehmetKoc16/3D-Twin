import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { BodyManifest, BodyParams, MeasuresDef } from '../contracts';
import type { FaceMapDef } from '../faceContracts';
import { AvatarCoreError } from '../errors';
import { buildBasePositions } from '../morphs';
import { solveBody } from '../solver';
import { mulberry32 } from '../testing/syntheticBody';
import { parseGlbMesh } from '../testing/glb';
import type { SolverData } from '../types';
import {
  faceLandmarkPositions,
  fitFaceModifiers,
  stableLandmarkIndices,
  MEDIAPIPE_EYEBROWS,
  type FitFaceOptions,
} from './fitFace';

const CANDIDATES = [
  '../../apps/web/public/assets/body',
  'apps/web/public/assets/body',
  '../apps/web/public/assets/body',
];
const dir = CANDIDATES.find(
  (d) => existsSync(`${d}/manifest.json`) && existsSync(`${d}/face-map.json`),
);

function bytes(name: string): ArrayBuffer {
  const b = readFileSync(`${dir}/${name}`);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
const json = <T>(name: string): T => JSON.parse(new TextDecoder().decode(bytes(name))) as T;

function load(): { data: SolverData; faceMap: FaceMapDef } {
  const manifest = json<BodyManifest>('manifest.json');
  const measures = json<MeasuresDef>('measures.json');
  const mesh = parseGlbMesh(bytes('base.glb'));
  const base = buildBasePositions(mesh.positions, manifest);
  return {
    data: { manifest, base, morphs: bytes('morphs.bin'), measures, indices: mesh.indices },
    faceMap: json<FaceMapDef>('face-map.json'),
  };
}

const params = (gender: number, heightCm: number, weightKg: number): BodyParams => ({
  gender,
  heightCm,
  weightKg,
  shoe: { system: 'EU', size: 42 },
});

const W = 1200;
const H = 1600;

interface Sim {
  /** pixels per meter */
  scale: number;
  rotDeg: number;
  tx: number;
  ty: number;
}

interface Pose {
  yawDeg: number;
  pitchDeg: number;
}

/**
 * MediaPipe-like landmarks (478 x 3, normalized) from avatar landmark positions: head pose (yaw about the vertical
 * axis, pitch; positive = nose right / up), similarity in the image plane, y flip, gaussian noise (also on depth).
 */
function synthLandmarks(
  pos: Float64Array,
  sim: Sim,
  noisePx: number,
  rng: () => number,
  pose: Pose = { yawDeg: 0, pitchDeg: 0 },
  depthScale = 1,
): Float32Array {
  const out = new Float32Array(478 * 3);
  const rad = Math.PI / 180;
  const c = Math.cos(sim.rotDeg * rad) * sim.scale;
  const s = Math.sin(sim.rotDeg * rad) * sim.scale;
  const cy = Math.cos(pose.yawDeg * rad);
  const sy = Math.sin(pose.yawDeg * rad);
  const cp = Math.cos(pose.pitchDeg * rad);
  const sp = Math.sin(pose.pitchDeg * rad);
  const gauss = (): number => {
    const u = Math.max(rng(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  };
  for (let i = 0; i < 468; i++) {
    const x0 = pos[i * 3]!;
    const y0 = pos[i * 3 + 1]!;
    const z0 = pos[i * 3 + 2]! - 0.05; // rotate about a point inside the head
    // Ry(yaw) then Rx(-pitch)
    const x1 = cy * x0 + sy * z0;
    const z1 = -sy * x0 + cy * z0;
    const y2 = cp * y0 + sp * z1;
    const z2 = -sp * y0 + cp * z1;
    const wx = c * x1 - s * y2 + sim.tx + noisePx * gauss();
    const wy = s * x1 + c * y2 + sim.ty + noisePx * gauss();
    out[i * 3] = wx / W;
    out[i * 3 + 1] = -wy / H; // photo y points down
    // MediaPipe: smaller z = closer, z has the scale of x (normalized by the image width)
    out[i * 3 + 2] = -(depthScale * sim.scale * z2 + 2 * noisePx * gauss()) / W;
  }
  return out;
}

const defaultSim: Sim = { scale: 3800, rotDeg: 0, tx: W / 2, ty: -H / 2 };

describe('stable landmark subset', () => {
  it('has the documented shape without the real assets', () => {
    // synthetic face map skeleton: only regions matter
    const regions = {
      leftEye: [263, 249],
      rightEye: [33, 7],
      lips: [61, 291, 13, 14, 0],
      leftCheek: [],
      rightCheek: [],
      forehead: [],
    };
    const fm = { regions } as unknown as FaceMapDef;
    const stable = stableLandmarkIndices(fm);
    for (const i of [263, 249, 33, 7, 13, 14, 0, ...MEDIAPIPE_EYEBROWS])
      expect(stable).not.toContain(i);
    expect(stable).toContain(61);
    expect(stable).toContain(291);
    expect(stable).toContain(152); // chin
    expect(stable).toEqual([...stable].sort((a, b) => a - b));
  });
});

describe.skipIf(!dir)('fitFaceModifiers on the real assets', () => {
  const { data, faceMap } = dir ? load() : ({} as ReturnType<typeof load>);
  const male = dir ? solveBody(data, params(1, 178, 76)) : undefined;
  const female = dir ? solveBody(data, params(0, 165, 58)) : undefined;
  const ids = faceMap?.fitModifiers ?? [];

  const randomTruth = (seed: number, amp: number): Record<string, number> => {
    const rng = mulberry32(seed);
    const out: Record<string, number> = {};
    for (const id of ids) out[id] = (rng() * 2 - 1) * amp;
    return out;
  };

  const run = (
    body: NonNullable<typeof male>,
    truth: Record<string, number>,
    sim: Sim,
    noisePx: number,
    seed: number,
    pose?: Pose,
    options?: FitFaceOptions,
    depthScale?: number,
  ) => {
    const pos = faceLandmarkPositions(data, body.weights, faceMap, truth);
    const lm = synthLandmarks(pos, sim, noisePx, mulberry32(seed), pose, depthScale);
    return fitFaceModifiers({
      data,
      baseWeights: body.weights,
      faceMap,
      landmarks: lm,
      imageWidth: W,
      imageHeight: H,
      options,
    });
  };

  it('the face map exposes the modifiers the fit relies on', () => {
    expect(ids.length).toBeGreaterThanOrEqual(8);
    const known = new Set(data.manifest.modifiers.map((m) => m.id));
    for (const id of ids) expect(known.has(id), id).toBe(true);
  });

  it('recovers known modifier values within 0.1 (noise, random similarity transform)', () => {
    for (let seed = 1; seed <= 6; seed++) {
      const truth = randomTruth(seed, 0.7);
      const rng = mulberry32(100 + seed);
      const sim: Sim = {
        scale: 3000 + rng() * 2000,
        rotDeg: (rng() - 0.5) * 30,
        tx: W * (0.4 + 0.2 * rng()),
        ty: -H * (0.4 + 0.2 * rng()),
      };
      const res = run(seed % 2 ? male! : female!, truth, sim, 0.4, 1000 + seed);
      const errs = ids.map((id) => Math.abs(res.modifierValues[id]! - truth[id]!));
      console.log(
        `[face] seed ${seed}: max err ${Math.max(...errs).toFixed(3)} rms ${res.rmsResidual.toFixed(3)} mm, ${res.iterations} it`,
      );
      ids.forEach((id, k) => expect(errs[k], `${id} seed ${seed}`).toBeLessThan(0.1));
      expect(res.rmsResidual).toBeLessThan(0.6);
    }
  });

  it('is nearly exact without noise', () => {
    const truth = randomTruth(42, 0.8);
    const res = run(male!, truth, { scale: 4200, rotDeg: -12, tx: 500, ty: -900 }, 0, 7);
    for (const id of ids)
      expect(Math.abs(res.modifierValues[id]! - truth[id]!), id).toBeLessThan(0.03);
    expect(res.rmsResidual).toBeLessThan(0.05);
  });

  it('returns about zero for a zero-modifier head', () => {
    for (const body of [male!, female!]) {
      const res = run(body, {}, defaultSim, 0.4, 5);
      for (const id of ids) expect(Math.abs(res.modifierValues[id]!), id).toBeLessThan(0.05);
      expect(res.rmsResidual).toBeLessThan(0.4);
    }
    const exact = run(male!, {}, { scale: 3500, rotDeg: 9, tx: 300, ty: -400 }, 0, 5);
    for (const id of ids) expect(Math.abs(exact.modifierValues[id]!), id).toBeLessThan(0.01);
  });

  it('is invariant to the similarity transform of the photo', () => {
    const truth = randomTruth(9, 0.6);
    const a = run(male!, truth, { scale: 3000, rotDeg: 0, tx: 600, ty: -800 }, 0, 1);
    const b = run(male!, truth, { scale: 5200, rotDeg: 25, tx: 200, ty: -1300 }, 0, 1);
    for (const id of ids)
      expect(Math.abs(a.modifierValues[id]! - b.modifierValues[id]!), id).toBeLessThan(2e-3);
  });

  it('values stay inside the modifier ranges (clamped for extreme faces)', () => {
    const truth: Record<string, number> = {};
    ids.forEach((id, k) => (truth[id] = k % 2 ? 1 : -1));
    // ground truth at the very ends of the ranges (one-sided modifiers are clamped by the map itself)
    const res = run(male!, truth, defaultSim, 0.2, 3);
    for (const id of ids) {
      const def = data.manifest.modifiers.find((m) => m.id === id)!;
      expect(res.modifierValues[id]!).toBeGreaterThanOrEqual(def.min - 1e-9);
      expect(res.modifierValues[id]!).toBeLessThanOrEqual(def.max + 1e-9);
    }
    // the fit reduces the residual compared with the neutral head
    const pos = faceLandmarkPositions(data, male!.weights, faceMap, truth);
    const lm = synthLandmarks(pos, defaultSim, 0.2, mulberry32(3));
    const zero = fitFaceModifiers({
      data,
      baseWeights: male!.weights,
      faceMap,
      landmarks: lm,
      imageWidth: W,
      imageHeight: H,
      options: { maxIterations: 0 },
    });
    expect(res.rmsResidual).toBeLessThan(zero.rmsResidual * 0.3);
  });

  it('removes head pitch and yaw using the landmark depth', () => {
    const truth = randomTruth(21, 0.6);
    const cases: [number, number][] = [
      [0, 8],
      [0, -12],
      [8, 0],
      [-10, 6],
      [12, -10],
    ];
    for (const [yaw, pitch] of cases) {
      const res = run(male!, truth, { scale: 3900, rotDeg: 5, tx: 600, ty: -800 }, 0.4, 3, {
        yawDeg: yaw,
        pitchDeg: pitch,
      });
      const worst = Math.max(...ids.map((id) => Math.abs(res.modifierValues[id]! - truth[id]!)));
      console.log(
        `[face] yaw ${yaw} pitch ${pitch}: max err ${worst.toFixed(3)}, pose ${JSON.stringify(res.pose)}`,
      );
      expect(worst, `yaw ${yaw} pitch ${pitch}`).toBeLessThan(0.1);
      expect(res.pose).not.toBeNull();
      expect(Math.abs(res.pose!.yawDeg - yaw)).toBeLessThan(3);
      expect(Math.abs(res.pose!.pitchDeg - pitch)).toBeLessThan(3);
      expect(Math.abs(res.pose!.rollDeg - 5)).toBeLessThan(3);
    }
  });

  it('pose compensation copes with inaccurate depth (30 % scale error, noise)', () => {
    const truth = randomTruth(23, 0.6);
    for (const depthScale of [0.7, 1.3]) {
      const res = run(
        female!,
        truth,
        defaultSim,
        0.6,
        4,
        { yawDeg: 6, pitchDeg: 10 },
        undefined,
        depthScale,
      );
      const worst = Math.max(...ids.map((id) => Math.abs(res.modifierValues[id]! - truth[id]!)));
      console.log(`[face] depth scale ${depthScale}: max err ${worst.toFixed(3)}`);
      expect(worst).toBeLessThan(0.2);
    }
  });

  it('without depth the photo is treated as frontal (pose null); pose "off" ignores depth', () => {
    const truth = randomTruth(25, 0.6);
    const pos = faceLandmarkPositions(data, male!.weights, faceMap, truth);
    const lm = synthLandmarks(pos, defaultSim, 0.3, mulberry32(9));
    for (let i = 0; i < 478; i++) lm[i * 3 + 2] = 0;
    const flat = fitFaceModifiers({
      data,
      baseWeights: male!.weights,
      faceMap,
      landmarks: lm,
      imageWidth: W,
      imageHeight: H,
    });
    expect(flat.pose).toBeNull();
    for (const id of ids)
      expect(Math.abs(flat.modifierValues[id]! - truth[id]!), id).toBeLessThan(0.06);
    const off = run(male!, truth, defaultSim, 0.3, 9, undefined, { pose: 'off' });
    expect(off.pose).toBeNull();
    for (const id of ids)
      expect(Math.abs(off.modifierValues[id]! - truth[id]!), id).toBeLessThan(0.06);
  });

  it('is fast (< 100 ms cold, < 50 ms warm)', () => {
    const truth = randomTruth(11, 0.5);
    const pos = faceLandmarkPositions(data, female!.weights, faceMap, truth);
    const lm = synthLandmarks(pos, defaultSim, 0.4, mulberry32(11));
    const call = () =>
      fitFaceModifiers({
        data,
        baseWeights: female!.weights,
        faceMap: structuredClone(faceMap), // fresh object: forces the precompute (cold path)
        landmarks: lm,
        imageWidth: W,
        imageHeight: H,
      });
    const t0 = performance.now();
    call();
    const cold = performance.now() - t0;
    const warm = () => {
      const t1 = performance.now();
      fitFaceModifiers({
        data,
        baseWeights: female!.weights,
        faceMap,
        landmarks: lm,
        imageWidth: W,
        imageHeight: H,
      });
      return performance.now() - t1;
    };
    warm();
    const w = Math.min(warm(), warm(), warm());
    console.log(`[face] fit time: cold ${cold.toFixed(1)} ms, warm ${w.toFixed(1)} ms`);
    expect(cold).toBeLessThan(100);
    expect(w).toBeLessThan(50);
  });

  it('rejects invalid input', () => {
    const lm = new Float32Array(478 * 3);
    const base = {
      data,
      baseWeights: male!.weights,
      faceMap,
      landmarks: lm,
      imageWidth: W,
      imageHeight: H,
    };
    expect(() => fitFaceModifiers({ ...base, imageWidth: 0 })).toThrow(AvatarCoreError);
    expect(() => fitFaceModifiers({ ...base, landmarks: new Float32Array(30) })).toThrow(
      AvatarCoreError,
    );
    const nan = new Float32Array(478 * 3).fill(0.5);
    nan[5 * 3] = Number.NaN;
    expect(() => fitFaceModifiers({ ...base, landmarks: nan })).toThrow(AvatarCoreError);
    const bad = { ...faceMap, fitModifiers: ['nope/nope'] };
    expect(() => fitFaceModifiers({ ...base, faceMap: bad })).toThrow(AvatarCoreError);
  });
});
