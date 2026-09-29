import { create } from 'zustand';

export type PoseId = 't-pose' | 'a-pose' | 'relaxed' | 'hands-on-hips' | 'walk' | 'side';
interface PoseState { poseId: PoseId; setPose: (poseId: PoseId) => void }
export const usePoseStore = create<PoseState>((set) => ({ poseId: 't-pose', setPose: (poseId) => set({ poseId }) }));
