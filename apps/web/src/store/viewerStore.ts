import { create } from 'zustand';
import type { FocusPreset } from '../features/viewer/focusTargets';

interface ViewerState {
  focusPreset: FocusPreset;
  focusRequest: number;
  focusPoint: [number, number, number] | null;
  autoRotate: boolean;
  requestFocus: (preset: FocusPreset) => void;
  requestPoint: (point: [number, number, number]) => void;
  toggleAutoRotate: () => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  focusPreset: 'full', focusRequest: 0, focusPoint: null, autoRotate: false,
  requestFocus: (focusPreset) => set((state) => ({ focusPreset, focusPoint: null, focusRequest: state.focusRequest + 1 })),
  requestPoint: (focusPoint) => set((state) => ({ focusPoint, focusRequest: state.focusRequest + 1 })),
  toggleAutoRotate: () => set((state) => ({ autoRotate: !state.autoRotate })),
}));
