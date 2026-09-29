/**
 * Runs an async job with "latest wins" coalescing: at most one job is in flight and at most one request waits.
 * While a job runs, newer submissions replace the waiting one, so a slider drag never builds a queue.
 */
export class LatestWinsRunner<P, R> {
  private busy = false;
  private hasPending = false;
  private pending: P | undefined;

  constructor(
    private readonly run: (params: P) => Promise<R>,
    private readonly onResult: (result: R, params: P) => void,
    private readonly onError: (error: unknown) => void = () => undefined,
  ) {}

  get isBusy(): boolean {
    return this.busy;
  }

  submit(params: P): void {
    if (this.busy) {
      this.pending = params;
      this.hasPending = true;
      return;
    }
    void this.start(params);
  }

  private async start(first: P): Promise<void> {
    this.busy = true;
    let params = first;
    for (;;) {
      try {
        const result = await this.run(params);
        this.onResult(result, params);
      } catch (error) {
        this.onError(error);
      }
      if (!this.hasPending) break;
      params = this.pending as P;
      this.pending = undefined;
      this.hasPending = false;
    }
    this.busy = false;
  }
}
