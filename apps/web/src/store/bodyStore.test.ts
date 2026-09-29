import { beforeEach, describe, expect, it } from 'vitest';
import { defaultBodyParams, useBodyStore } from './bodyStore';

describe('body store', () => {
  beforeEach(() => useBodyStore.getState().reset());
  it('starts with the requested defaults', () => {
    expect(defaultBodyParams).toMatchObject({ gender: 1, heightCm: 175, weightKg: 75, shoe: { system: 'EU', size: 42 } });
    expect(useBodyStore.getState().params).toEqual(defaultBodyParams);
  });
  it('resets edits including optional manual values', () => {
    useBodyStore.getState().setField('heightCm', 190);
    useBodyStore.getState().setField('thighCm', 57);
    useBodyStore.getState().setShoe({ system: 'UK', size: 8 });
    useBodyStore.getState().reset();
    expect(useBodyStore.getState().params).toEqual(defaultBodyParams);
  });
});
