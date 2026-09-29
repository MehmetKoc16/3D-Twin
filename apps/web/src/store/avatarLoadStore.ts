import { create } from 'zustand';

export type AvatarLoadStatus = 'loading' | 'ready' | 'error';

interface AvatarLoadState {
  status: AvatarLoadStatus;
  /** 0..1 over the whole download (about 17 MB). */
  progress: number;
  error: string | null;
  setProgress: (progress: number) => void;
  setReady: () => void;
  setError: (message: string) => void;
}

export const useAvatarLoadStore = create<AvatarLoadState>((set) => ({
  status: 'loading',
  progress: 0,
  error: null,
  setProgress: (progress) => set((state) => (state.status === 'loading' ? { progress } : state)),
  setReady: () => set({ status: 'ready', progress: 1, error: null }),
  setError: (error) => set({ status: 'error', error }),
}));
