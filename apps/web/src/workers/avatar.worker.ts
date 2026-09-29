/// <reference lib="webworker" />
import * as Comlink from 'comlink';
import {
  applyMorphs,
  buildBasePositions,
  computeJoints,
  createBodySolver,
  fitFaceModifiers,
  mergeWeights,
  modifierWeights,
  parseMorphs,
  validateRig,
  type BodyManifest,
  type BodyParams,
  type BodySolver,
  type FaceMapDef,
  type MeasuresDef,
  type RigDef,
} from '@dt/avatar-core';
import type {
  AvatarInit,
  AvatarInitResult,
  AvatarSolveResult,
  AvatarWorkerApi,
} from './avatarProtocol';
import { flattenJoints, groundRenderPositions } from './ground';
import { FaceShapeState } from './faceShape';

const MORPHS_FALLBACK_BYTES = 15_789_632;

let solver: BodySolver | null = null;
let rig: RigDef | null = null;
let manifest: BodyManifest | null = null;
let boneNames: string[] = [];
let solverData: { manifest: BodyManifest; base: Float32Array; morphs: ArrayBuffer } | null = null;
let baseUrl = '';
let faceMap: FaceMapDef | null = null;
let solvedWeights: ReadonlyMap<string, number> | null = null;
const face = new FaceShapeState((request) => {
  if (!solverData || !solvedWeights) throw new Error('avatar worker: face fit before the first solve');
  const fit = fitFaceModifiers({ data: solverData, baseWeights: solvedWeights, ...request });
  return { modifierValues: fit.modifierValues, rmsResidual: fit.rmsResidual };
});

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return (await res.json()) as T;
}

async function fetchBinary(url: string, onFraction: (f: number) => void): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  const declared = Number(res.headers.get('content-length'));
  const total = Number.isFinite(declared) && declared > 0 ? declared : MORPHS_FALLBACK_BYTES;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onFraction(Math.min(1, received / total));
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out.buffer;
}

const api: AvatarWorkerApi = {
  async init(init: AvatarInit, onProgress?: (fraction: number) => void): Promise<AvatarInitResult> {
    const report = (f: number): void => onProgress?.(f);
    const [m, measures, rigDef] = await Promise.all([
      fetchJson<BodyManifest>(`${init.baseUrl}manifest.json`),
      fetchJson<MeasuresDef>(`${init.baseUrl}measures.json`),
      fetchJson<RigDef>(`${init.baseUrl}rig.json`),
    ]);
    report(0.02);
    const morphs = await fetchBinary(`${init.baseUrl}${m.morphs}`, (f) => report(0.02 + f * 0.9));
    validateRig(rigDef, m.vertexCount);
    const base = buildBasePositions(init.positions, m);
    parseMorphs(morphs, m); // validate once (the solver parses and caches its own copy)
    solver = createBodySolver({ manifest: m, base, morphs, measures, indices: init.indices });
    manifest = m;
    solverData = { manifest: m, base, morphs };
    baseUrl = init.baseUrl;
    rig = rigDef;
    boneNames = rigDef.bones.map((b) => b.name);
    report(1);
    return { boneNames, renderVertexCount: m.renderVertexCount };
  },

  solve(params: BodyParams): AvatarSolveResult {
    if (!solver || !rig || !manifest) throw new Error('avatar worker: solve() before init()');
    const t0 = performance.now();
    const res = solver.solve(params);
    solvedWeights = res.weights;
    let bodyPositions = res.positions;
    const faced = face.apply();
    if (faced.values && solverData) {
      // face modifiers on top of the solved body weights
      const merged = mergeWeights(res.weights, modifierWeights(manifest, faced.values));
      bodyPositions = new Float32Array(res.positions.length);
      applyMorphs(solverData.base, solverData.morphs, manifest, merged, bodyPositions);
    }
    const solveMs = performance.now() - t0;
    const { positions, offsetY } = groundRenderPositions(bodyPositions, manifest.renderVertexCount);
    const joints = flattenJoints(boneNames, computeJoints(rig, bodyPositions), offsetY);
    const result: AvatarSolveResult = {
      positions,
      joints,
      groundOffsetY: offsetY,
      achievedCm: res.achievedCm,
      residualsCm: res.residualsCm,
      unreachable: res.unreachable,
      estimatedMassKg: res.estimatedMassKg,
      solveMs,
      ...(faced.report !== undefined ? { faceFit: faced.report } : {}),
    };
    return Comlink.transfer(result, [positions.buffer, joints.buffer]);
  },

  async setFaceLandmarks(landmarks: Float32Array, imageWidth: number, imageHeight: number): Promise<void> {
    if (!solver) throw new Error('avatar worker: setFaceLandmarks() before init()');
    faceMap ??= await fetchJson<FaceMapDef>(`${baseUrl}face-map.json`);
    face.set({ faceMap, landmarks, imageWidth, imageHeight });
  },

  clearFace(): void {
    face.clear();
  },
};

Comlink.expose(api);
