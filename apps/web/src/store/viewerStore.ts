import { create } from 'zustand';
import type { FocusPreset } from '../features/viewer/focusTargets';

export type RenderQuality = 'high' | 'performance';

function initialQuality(): RenderQuality {
  if (typeof window === 'undefined') return 'high';
  try {
    const saved = window.localStorage.getItem('dt:viewer:quality');
    if (saved === 'high' || saved === 'performance') return saved;
  } catch { /* Rendering still works when browser storage is unavailable. */ }
  return window.matchMedia?.('(pointer: coarse)').matches ? 'performance' : 'high';
}

interface ViewerState {
  focusPreset: FocusPreset;
  focusRequest: number;
  focusPoint: [number, number, number] | null;
  autoRotate: boolean;
  quality: RenderQuality;
  setQuality: (quality: RenderQuality) => void;
  requestFocus: (preset: FocusPreset) => void;
  requestPoint: (point: [number, number, number]) => void;
  toggleAutoRotate: () => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  focusPreset: 'full', focusRequest: 0, focusPoint: null, autoRotate: false,
  quality: initialQuality(),
  setQuality: (quality) => {
    try { if (typeof window !== 'undefined') window.localStorage.setItem('dt:viewer:quality', quality); }
    catch { /* Storage is optional. */ }
    set({ quality });
  },
  requestFocus: (focusPreset) => set((state) => ({ focusPreset, focusPoint: null, focusRequest: state.focusRequest + 1 })),
  requestPoint: (focusPoint) => set((state) => ({ focusPoint, focusRequest: state.focusRequest + 1 })),
  toggleAutoRotate: () => set((state) => ({ autoRotate: !state.autoRotate })),
}));
