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

export const MANNEQUIN_COLOR = '#b4b9be';

interface AppearanceState {
  mode: AvatarMaterialMode;
  toneIndex: number;
  setMode: (mode: AvatarMaterialMode) => void;
  setTone: (toneIndex: number) => void;
}

export const useAppearanceStore = create<AppearanceState>((set) => ({
  mode: 'skin',
  toneIndex: 1,
  setMode: (mode) => set({ mode }),
  setTone: (toneIndex) => set({ toneIndex, mode: 'skin' }),
}));
