import { beforeEach, describe, expect, it, vi } from 'vitest';

const db = new Map<string, unknown>();
let failing = false;
vi.mock('idb-keyval', () => ({
  get: vi.fn(async (key: string) => {
    if (failing) throw new Error('storage unavailable');
    return db.get(key);
  }),
  set: vi.fn(async (key: string, value: unknown) => {
    if (failing) throw new Error('storage unavailable');
    db.set(key, value);
  }),
  del: vi.fn(async (key: string) => void db.delete(key)),
}));

import {
  APPEARANCE_KEYS,
  DEFAULT_EYE_COLOR,
  DEFAULT_HAIR_COLOR,
  effectiveEyebrowColor,
  resetAppearanceTouched,
  useAppearanceStore,
} from './appearanceStore';

const pristine = useAppearanceStore.getState();

beforeEach(() => {
  db.clear();
  failing = false;
  resetAppearanceTouched();
  useAppearanceStore.setState({ ...pristine, hydrated: false }, true);
});

const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

describe('appearanceStore defaults', () => {
  it('starts bald with the default brows, brown eyes and brows following the hair', () => {
    const s = useAppearanceStore.getState();
    expect(s.hairId).toBeNull();
    expect(s.eyebrowId).toBe('eyebrows-default');
    expect(s.eyeColor).toBe(DEFAULT_EYE_COLOR);
    expect(s.hairColor).toBe(DEFAULT_HAIR_COLOR);
    expect(effectiveEyebrowColor(s)).toBe(DEFAULT_HAIR_COLOR);
  });

  it('eyebrows follow the hair colour until an own colour is chosen', () => {
    const s = useAppearanceStore.getState();
    s.setHairColor('#c9a35c');
    expect(effectiveEyebrowColor(useAppearanceStore.getState())).toBe('#c9a35c');
    s.setEyebrowColor('#111111');
    s.setHairColor('#93411f');
    expect(effectiveEyebrowColor(useAppearanceStore.getState())).toBe('#111111');
    s.setEyebrowColor(null);
    expect(effectiveEyebrowColor(useAppearanceStore.getState())).toBe('#93411f');
  });

  it('ignores invalid colours', () => {
    useAppearanceStore.getState().setEyeColor('not-a-colour');
    useAppearanceStore.getState().setHairColor('#12');
    expect(useAppearanceStore.getState().eyeColor).toBe(DEFAULT_EYE_COLOR);
    expect(useAppearanceStore.getState().hairColor).toBe(DEFAULT_HAIR_COLOR);
  });
});

describe('appearanceStore persistence', () => {
  it('writes every change below the appearance: prefix', async () => {
    const s = useAppearanceStore.getState();
    s.setHair('hair-long');
    s.setHairColor('#93411f');
    s.setEyebrow('eyebrows-thick');
    s.setEyebrowColor('#222222');
    s.setEyeColor('#3d6fa8');
    await flush();
    expect(db.get(APPEARANCE_KEYS.hairId)).toBe('hair-long');
    expect(db.get(APPEARANCE_KEYS.hairColor)).toBe('#93411f');
    expect(db.get(APPEARANCE_KEYS.eyebrowId)).toBe('eyebrows-thick');
    expect(db.get(APPEARANCE_KEYS.eyebrowColor)).toBe('#222222');
    expect(db.get(APPEARANCE_KEYS.eyeColor)).toBe('#3d6fa8');
    for (const key of db.keys()) expect(key.startsWith('appearance:')).toBe(true);
  });

  it('restores the stored appearance on hydrate', async () => {
    db.set(APPEARANCE_KEYS.hairId, 'hair-bob');
    db.set(APPEARANCE_KEYS.hairColor, '#9a9a9a');
    db.set(APPEARANCE_KEYS.eyebrowColor, null);
    db.set(APPEARANCE_KEYS.eyeColor, '#3f7a4a');
    db.set(APPEARANCE_KEYS.toneIndex, 3);
    db.set(APPEARANCE_KEYS.mode, 'mannequin');
    await useAppearanceStore.getState().hydrate();
    const s = useAppearanceStore.getState();
    expect(s.hairId).toBe('hair-bob');
    expect(s.hairColor).toBe('#9a9a9a');
    expect(s.eyeColor).toBe('#3f7a4a');
    expect(s.toneIndex).toBe(3);
    expect(s.mode).toBe('mannequin');
    expect(s.eyebrowColor).toBeNull();
  });

  it('rejects malformed stored values', async () => {
    db.set(APPEARANCE_KEYS.hairColor, 'red');
    db.set(APPEARANCE_KEYS.toneIndex, 99);
    db.set(APPEARANCE_KEYS.mode, 'wireframe');
    db.set(APPEARANCE_KEYS.eyeColor, 42);
    await useAppearanceStore.getState().hydrate();
    const s = useAppearanceStore.getState();
    expect(s.hairColor).toBe(DEFAULT_HAIR_COLOR);
    expect(s.toneIndex).toBe(1);
    expect(s.mode).toBe('skin');
    expect(s.eyeColor).toBe(DEFAULT_EYE_COLOR);
  });

  it('keeps what the user chose while the stored values were loading', async () => {
    db.set(APPEARANCE_KEYS.eyeColor, '#3f7a4a');
    const loading = useAppearanceStore.getState().hydrate();
    useAppearanceStore.getState().setEyeColor('#3d6fa8');
    await loading;
    expect(useAppearanceStore.getState().eyeColor).toBe('#3d6fa8');
  });

  it('survives unavailable storage (private mode)', async () => {
    failing = true;
    useAppearanceStore.getState().setHair('hair-short');
    await useAppearanceStore.getState().hydrate();
    await flush();
    expect(useAppearanceStore.getState().hairId).toBe('hair-short');
  });
});
