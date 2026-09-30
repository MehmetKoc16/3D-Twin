import { get, set } from 'idb-keyval';
import { create } from 'zustand';

export type AvatarMaterialMode = 'skin' | 'mannequin';

export interface SkinTone {
  id: string;
  color: string;
}

/** Five neutral skin-tone presets, light to dark (sRGB hex). */
export const skinTones: readonly SkinTone[] = [
  { id: 'porcelain', color: '#f0cdb6' },
  { id: 'light', color: '#e0ac8b' },
  { id: 'medium', color: '#c68a63' },
  { id: 'tan', color: '#94603f' },
  { id: 'deep', color: '#5e3a26' },
];

export interface ColorPreset {
  id: string;
  color: string;
}

/** Hair / eyebrow colour presets (sRGB hex). */
export const hairColors: readonly ColorPreset[] = [
  { id: 'black', color: '#1a1614' },
  { id: 'darkBrown', color: '#3b2a1e' },
  { id: 'brown', color: '#6b4429' },
  { id: 'blonde', color: '#c9a35c' },
  { id: 'red', color: '#93411f' },
  { id: 'grey', color: '#9a9a9a' },
];

/** Eye (iris) colour presets (sRGB hex). */
export const eyeColors: readonly ColorPreset[] = [
  { id: 'brown', color: '#5a3820' },
  { id: 'hazel', color: '#8a6a2f' },
  { id: 'green', color: '#3f7a4a' },
  { id: 'blue', color: '#3d6fa8' },
  { id: 'grey', color: '#7d8b93' },
];

export const MANNEQUIN_COLOR = '#b4b9be';

export const DEFAULT_EYEBROW_ID = 'eyebrows-default';
export const DEFAULT_HAIR_COLOR = hairColors[1]!.color;
export const DEFAULT_EYE_COLOR = eyeColors[0]!.color;

/** IndexedDB keys: every appearance record lives below the `appearance:` prefix (idb-keyval). */
export const APPEARANCE_KEYS = {
  mode: 'appearance:mode',
  toneIndex: 'appearance:toneIndex',
  useFaceTone: 'appearance:useFaceTone',
  hairId: 'appearance:hairId',
  hairColor: 'appearance:hairColor',
  eyebrowId: 'appearance:eyebrowId',
  eyebrowColor: 'appearance:eyebrowColor',
  eyeColor: 'appearance:eyeColor',
} as const;

type PersistedKey = keyof typeof APPEARANCE_KEYS;

const HEX = /^#[0-9a-fA-F]{6}$/;

export function isHexColor(value: unknown): value is string {
  return typeof value === 'string' && HEX.test(value);
}

interface AppearanceValues {
  mode: AvatarMaterialMode;
  toneIndex: number;
  /** Use the skin tone estimated from the selfie (when a face is baked) instead of the preset. */
  useFaceTone: boolean;
  /** Selected hair part id (parts/index.json), `null` = bald. */
  hairId: string | null;
  hairColor: string;
  eyebrowId: string;
  /** `null`: the eyebrows follow the hair colour. */
  eyebrowColor: string | null;
  eyeColor: string;
}

interface AppearanceState extends AppearanceValues {
  hydrated: boolean;
  setUseFaceTone: (value: boolean) => void;
  setMode: (mode: AvatarMaterialMode) => void;
  setTone: (toneIndex: number) => void;
  setHair: (hairId: string | null) => void;
  /** Sets the hair colour; the eyebrows follow it unless an own eyebrow colour was chosen. */
  setHairColor: (color: string) => void;
  setEyebrow: (eyebrowId: string) => void;
  /** `null` makes the eyebrows follow the hair colour again. */
  setEyebrowColor: (color: string | null) => void;
  setEyeColor: (color: string) => void;
  /** Restores the stored appearance (values changed by the user meanwhile win). */
  hydrate: () => Promise<void>;
}

/** The colour the eyebrows are rendered in. */
export function effectiveEyebrowColor(state: Pick<AppearanceValues, 'hairColor' | 'eyebrowColor'>): string {
  return state.eyebrowColor ?? state.hairColor;
}

const touched = new Set<PersistedKey>();

async function persist(key: PersistedKey, value: unknown): Promise<void> {
  touched.add(key);
  try {
    await set(APPEARANCE_KEYS[key], value);
  } catch {
    /* storage unavailable (private mode / quota): the session still works */
  }
}

export const useAppearanceStore = create<AppearanceState>((update, read) => ({
  mode: 'skin',
  toneIndex: 1,
  useFaceTone: true,
  hairId: null,
  hairColor: DEFAULT_HAIR_COLOR,
  eyebrowId: DEFAULT_EYEBROW_ID,
  eyebrowColor: null,
  eyeColor: DEFAULT_EYE_COLOR,
  hydrated: false,

  setUseFaceTone: (useFaceTone) => {
    update({ useFaceTone });
    void persist('useFaceTone', useFaceTone);
  },
  setMode: (mode) => {
    update({ mode });
    void persist('mode', mode);
  },
  setTone: (toneIndex) => {
    update({ toneIndex, mode: 'skin', useFaceTone: false });
    void persist('toneIndex', toneIndex);
    void persist('mode', 'skin');
    void persist('useFaceTone', false);
  },
  setHair: (hairId) => {
    update({ hairId });
    void persist('hairId', hairId);
  },
  setHairColor: (hairColor) => {
    if (!isHexColor(hairColor)) return;
    update({ hairColor });
    void persist('hairColor', hairColor);
  },
  setEyebrow: (eyebrowId) => {
    update({ eyebrowId });
    void persist('eyebrowId', eyebrowId);
  },
  setEyebrowColor: (eyebrowColor) => {
    if (eyebrowColor !== null && !isHexColor(eyebrowColor)) return;
    update({ eyebrowColor });
    void persist('eyebrowColor', eyebrowColor);
  },
  setEyeColor: (eyeColor) => {
    if (!isHexColor(eyeColor)) return;
    update({ eyeColor });
    void persist('eyeColor', eyeColor);
  },

  hydrate: async () => {
    if (read().hydrated) return;
    update({ hydrated: true });
    const stored: Partial<Record<PersistedKey, unknown>> = {};
    try {
      await Promise.all(
        (Object.keys(APPEARANCE_KEYS) as PersistedKey[]).map(async (key) => {
          stored[key] = await get(APPEARANCE_KEYS[key]);
        }),
      );
    } catch {
      return; // nothing stored or storage unavailable
    }
    const next: Partial<AppearanceValues> = {};
    const take = (key: PersistedKey): boolean => !touched.has(key) && stored[key] !== undefined;
    if (take('mode') && (stored.mode === 'skin' || stored.mode === 'mannequin')) next.mode = stored.mode;
    if (
      take('toneIndex') &&
      typeof stored.toneIndex === 'number' &&
      Number.isInteger(stored.toneIndex) &&
      stored.toneIndex >= 0 &&
      stored.toneIndex < skinTones.length
    )
      next.toneIndex = stored.toneIndex;
    if (take('useFaceTone') && typeof stored.useFaceTone === 'boolean') next.useFaceTone = stored.useFaceTone;
    if (take('hairId') && (stored.hairId === null || typeof stored.hairId === 'string')) next.hairId = stored.hairId;
    if (take('hairColor') && isHexColor(stored.hairColor)) next.hairColor = stored.hairColor;
    if (take('eyebrowId') && typeof stored.eyebrowId === 'string') next.eyebrowId = stored.eyebrowId;
    if (take('eyebrowColor') && (stored.eyebrowColor === null || isHexColor(stored.eyebrowColor)))
      next.eyebrowColor = stored.eyebrowColor;
    if (take('eyeColor') && isHexColor(stored.eyeColor)) next.eyeColor = stored.eyeColor;
    if (Object.keys(next).length > 0) update(next);
  },
}));

/** Test helper: forget which keys the user touched in this session. */
export function resetAppearanceTouched(): void {
  touched.clear();
}
