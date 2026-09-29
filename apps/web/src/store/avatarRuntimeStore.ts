import { create } from 'zustand';
import type { Bone, Skeleton } from 'three';

/**
 * Runtime handles of the loaded avatar.
 * Writer: features/avatar. Readers: features/poses (PoseDriver), features/viewer (focus targets).
 */
export interface AvatarRuntimeState {
  skeleton: Skeleton | null;
  /** Bones by rig.json name. Empty until the avatar has loaded. */
  bones: ReadonlyMap<string, Bone>;
  /** Incremented every time the skeleton rest pose is rebuilt after a body-shape change. */
  restVersion: number;
  setSkeleton: (skeleton: Skeleton | null) => void;
  bumpRestVersion: () => void;
}

export const useAvatarRuntimeStore = create<AvatarRuntimeState>((set) => ({
  skeleton: null,
  bones: new Map(),
  restVersion: 0,
  setSkeleton: (skeleton) =>
    set({
      skeleton,
      bones: new Map(skeleton ? skeleton.bones.map((bone) => [bone.name, bone]) : []),
    }),
  bumpRestVersion: () => set((state) => ({ restVersion: state.restVersion + 1 })),
}));
