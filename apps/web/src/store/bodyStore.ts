import { get, set } from 'idb-keyval';
import { create } from 'zustand';
import type { BodyParams } from '@dt/avatar-core';

const storageKey = 'digital-twin-body-v1';
export const defaultBodyParams: BodyParams = {
  gender: 1, heightCm: 175, weightKg: 75,
  shoulderCm: 46, neckCm: 39, chestCm: 100, waistCm: 86, hipCm: 100,
  shoe: { system: 'EU', size: 42 },
};

type BodyField = Exclude<keyof BodyParams, 'shoe'>;
interface BodyState {
  params: BodyParams;
  hydrated: boolean;
  setField: (field: BodyField, value: number | undefined) => void;
  setShoe: (shoe: BodyParams['shoe']) => void;
  replace: (params: BodyParams) => void;
  reset: () => void;
  hydrate: () => Promise<void>;
  save: () => Promise<boolean>;
}

export function isBodyParams(value: unknown): value is BodyParams {
  if (typeof value !== 'object' || value === null) return false;
  const candidate = value as Record<string, unknown>;
  const shoe = candidate.shoe;
  const inRange = (input: unknown, min: number, max: number) => typeof input === 'number' && Number.isFinite(input) && input >= min && input <= max;
  const optionalRanges: [string, number, number][] = [
    ['shoulderCm', 35, 60], ['neckCm', 30, 50], ['chestCm', 70, 140], ['waistCm', 55, 140],
    ['hipCm', 70, 150], ['thighCm', 35, 85], ['upperArmCm', 20, 55], ['armLengthCm', 45, 85], ['inseamCm', 55, 105],
  ];
  return inRange(candidate.gender, 0, 1)
    && inRange(candidate.heightCm, 140, 210)
    && inRange(candidate.weightKg, 40, 150)
    && optionalRanges.every(([key, min, max]) => candidate[key] === undefined || inRange(candidate[key], min, max))
    && typeof shoe === 'object' && shoe !== null
    && ['EU', 'UK', 'US_M', 'US_W'].includes((shoe as Record<string, unknown>).system as string)
    && inRange((shoe as Record<string, unknown>).size, 2, 50);
}

export const useBodyStore = create<BodyState>((update, read) => ({
  params: { ...defaultBodyParams, shoe: { ...defaultBodyParams.shoe } },
  hydrated: false,
  setField: (field, value) => update((state) => ({ params: { ...state.params, [field]: value } })),
  setShoe: (shoe) => update((state) => ({ params: { ...state.params, shoe } })),
  replace: (params) => update({ params }),
  reset: () => update({ params: { ...defaultBodyParams, shoe: { ...defaultBodyParams.shoe } } }),
  hydrate: async () => {
    try {
      const stored: unknown = await get(storageKey);
      if (isBodyParams(stored)) update({ params: stored, hydrated: true });
      else update({ hydrated: true });
    } catch { update({ hydrated: true }); }
  },
  save: async () => {
    try { await set(storageKey, read().params); return true; }
    catch { return false; }
  },
}));

let saveTimer: ReturnType<typeof setTimeout> | undefined;
useBodyStore.subscribe((state, previous) => {
  if (!state.hydrated || state.params === previous.params) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => { void state.save(); }, 500);
});

