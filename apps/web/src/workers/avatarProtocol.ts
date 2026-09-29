import type { BodyParams, MeasureId } from '@dt/avatar-core';

/** Sent once by the main thread: raw (un-morphed) render vertices and triangle indices of base.glb. */
export interface AvatarInit {
  /** Base URL of the body assets, with trailing slash (e.g. `/assets/body/`). */
  baseUrl: string;
  /** Raw render positions, length 3 * renderVertexCount (transferred; the worker owns it afterwards). */
  positions: Float32Array;
  /** Triangle indices into the render vertices. */
  indices: Uint32Array;
}

export interface AvatarInitResult {
  /** Bone names in rig order; `joints` of a solve result follows this order. */
  boneNames: string[];
  renderVertexCount: number;
}

export interface AvatarSolveResult {
  /** Grounded render positions (min y = 0), length 3 * renderVertexCount. Transferred. */
  positions: Float32Array;
  /** Grounded bone joints: 6 floats per bone in rig order (head xyz, tail xyz). Transferred. */
  joints: Float32Array;
  groundOffsetY: number;
  achievedCm: Partial<Record<MeasureId, number>>;
  residualsCm: Partial<Record<MeasureId, number>>;
  unreachable: MeasureId[];
  estimatedMassKg: number;
  /** Pure solver time inside the worker. */
  solveMs: number;
}

export interface AvatarWorkerApi {
  init(init: AvatarInit, onProgress?: (fraction: number) => void): Promise<AvatarInitResult>;
  solve(params: BodyParams): AvatarSolveResult;
}
