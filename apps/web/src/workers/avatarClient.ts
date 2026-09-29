import * as Comlink from 'comlink';
import type { BodyParams } from '@dt/avatar-core';
import type {
  AvatarInit,
  AvatarInitResult,
  AvatarSolveResult,
  AvatarWorkerApi,
} from './avatarProtocol';
import { LatestWinsRunner } from './latestWins';

export interface SolveEnvelope {
  result: AvatarSolveResult;
  params: BodyParams;
  /** Time from posting the request to the worker until the result arrived on the main thread. */
  roundTripMs: number;
}

/** Main-thread handle of the avatar worker; `solve` calls are coalesced (latest wins). */
export class AvatarClient {
  private readonly worker: Worker;
  private readonly api: Comlink.Remote<AvatarWorkerApi>;
  private readonly runner: LatestWinsRunner<BodyParams, SolveEnvelope>;
  private lastParams: BodyParams | null = null;

  constructor(onResult: (envelope: SolveEnvelope) => void, onError: (error: unknown) => void) {
    this.worker = new Worker(new URL('./avatar.worker.ts', import.meta.url), { type: 'module' });
    this.api = Comlink.wrap<AvatarWorkerApi>(this.worker);
    this.runner = new LatestWinsRunner<BodyParams, SolveEnvelope>(
      async (params) => {
        const t0 = performance.now();
        const result: AvatarSolveResult = await this.api.solve(params);
        return { result, params, roundTripMs: performance.now() - t0 };
      },
      (envelope) => onResult(envelope),
      onError,
    );
  }

  init(init: AvatarInit, onProgress: (fraction: number) => void): Promise<AvatarInitResult> {
    return this.api.init(
      Comlink.transfer(init, [init.positions.buffer, init.indices.buffer]),
      Comlink.proxy(onProgress),
    );
  }

  solve(params: BodyParams): void {
    this.lastParams = { ...params, shoe: { ...params.shoe } };
    this.runner.submit(this.lastParams);
  }

  /** Sends photo landmarks; the worker fits the face shape during the re-solve triggered here. */
  async setFace(landmarks: Float32Array, imageWidth: number, imageHeight: number): Promise<void> {
    await this.api.setFaceLandmarks(landmarks.slice(), imageWidth, imageHeight);
    if (this.lastParams) this.runner.submit(this.lastParams);
  }

  async clearFace(): Promise<void> {
    await this.api.clearFace();
    if (this.lastParams) this.runner.submit(this.lastParams);
  }

  dispose(): void {
    this.worker.terminate();
  }
}
