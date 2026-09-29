import { describe, expect, it } from 'vitest';
import { LatestWinsRunner } from './latestWins';

describe('LatestWinsRunner', () => {
  it('drops intermediate requests and always finishes with the latest', async () => {
    const ran: number[] = [];
    const results: number[] = [];
    const runner = new LatestWinsRunner<number, number>(
      async (p) => {
        ran.push(p);
        await new Promise((r) => setTimeout(r, 5));
        return p * 2;
      },
      (r) => results.push(r),
    );
    for (let i = 1; i <= 10; i++) runner.submit(i);
    await new Promise((r) => setTimeout(r, 40));
    expect(ran).toEqual([1, 10]);
    expect(results).toEqual([2, 20]);
    expect(runner.isBusy).toBe(false);
    runner.submit(3);
    await new Promise((r) => setTimeout(r, 15));
    expect(results.at(-1)).toBe(6);
  });
  it('keeps going after an error', async () => {
    const errors: unknown[] = [];
    const results: number[] = [];
    const runner = new LatestWinsRunner<number, number>(
      async (p) => {
        if (p === 1) throw new Error('boom');
        return p;
      },
      (r) => results.push(r),
      (e) => errors.push(e),
    );
    runner.submit(1);
    runner.submit(2);
    await new Promise((r) => setTimeout(r, 10));
    expect(errors).toHaveLength(1);
    expect(results).toEqual([2]);
  });
});
