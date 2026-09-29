/// <reference lib="webworker" />
import * as Comlink from 'comlink';
import {
  buildBasePositions,
  computeJoints,
  createBodySolver,
  parseMorphs,
  validateRig,
  type BodyManifest,
  type BodyParams,
  type BodySolver,
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

const MORPHS_FALLBACK_BYTES = 15_789_632;

let solver: BodySolver | null = null;
let rig: RigDef | null = null;
let manifest: BodyManifest | null = null;
let boneNames: string[] = [];

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
    rig = rigDef;
    boneNames = rigDef.bones.map((b) => b.name);
    report(1);
    return { boneNames, renderVertexCount: m.renderVertexCount };
  },

  solve(params: BodyParams): AvatarSolveResult {
    if (!solver || !rig || !manifest) throw new Error('avatar worker: solve() before init()');
    const t0 = performance.now();
    const res = solver.solve(params);
    const solveMs = performance.now() - t0;
    const { positions, offsetY } = groundRenderPositions(res.positions, manifest.renderVertexCount);
    const joints = flattenJoints(boneNames, computeJoints(rig, res.positions), offsetY);
    const result: AvatarSolveResult = {
      positions,
      joints,
      groundOffsetY: offsetY,
      achievedCm: res.achievedCm,
      residualsCm: res.residualsCm,
      unreachable: res.unreachable,
      estimatedMassKg: res.estimatedMassKg,
      solveMs,
    };
    return Comlink.transfer(result, [positions.buffer, joints.buffer]);
  },
};

Comlink.expose(api);
