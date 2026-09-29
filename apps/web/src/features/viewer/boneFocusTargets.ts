import type { Bone } from 'three';
import { Vector3 } from 'three';
import type { Vec3 } from '@dt/avatar-core';
import { useAvatarRuntimeStore } from '../../store/avatarRuntimeStore';
import { getFocusTargets } from './focusTargets';
import type { FocusPreset, FocusTarget, FocusTargetProvider } from './focusTargets';

export type BonePositions = ReadonlyMap<string, Vec3>;

const REQUIRED = ['head', 'spine_03', 'pelvis', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'foot_l', 'foot_r', 'ball_l', 'ball_r'];

function mean(points: Vec3[]): Vec3 {
  const out: Vec3 = [0, 0, 0];
  for (const p of points) for (let k = 0; k < 3; k++) out[k] = out[k]! + p[k]! / points.length;
  return out;
}

/**
 * Focus targets from posed bone world positions (pure). Returns the height-fraction fallback when a bone is missing.
 * face = above the head bone (skull centre), upper = spine_03, lower = pelvis + thighs + calves,
 * feet = centre of the foot and ball bones (the shoe area), full = whole body from its height.
 */
export function computeBoneFocusTargets(
  bones: BonePositions,
  heightM: number,
): Record<FocusPreset, FocusTarget> {
  const fallback = getFocusTargets(heightM);
  if (REQUIRED.some((name) => !bones.has(name))) return fallback;
  const p = (name: string): Vec3 => bones.get(name)!;
  const h = Math.max(1.4, Math.min(2.1, heightM));
  const scale = h / 1.75;
  const head = p('head');
  const feet = mean([p('foot_l'), p('foot_r'), p('ball_l'), p('ball_r')]);
  return {
    full: { target: [0, h * 0.5, 0], distance: h * 2.05 },
    face: { target: [head[0], head[1] + 0.05 * scale, head[2] + 0.015], distance: 0.85 * scale },
    upper: { target: p('spine_03'), distance: 1.3 * scale },
    lower: {
      target: mean([p('pelvis'), p('thigh_l'), p('thigh_r'), p('calf_l'), p('calf_r')]),
      distance: 1.35 * scale,
    },
    feet: { target: feet, distance: 1.0 },
  };
}

function worldPositions(bones: ReadonlyMap<string, Bone>): BonePositions {
  const out = new Map<string, Vec3>();
  const v = new Vector3();
  for (const name of REQUIRED) {
    const bone = bones.get(name);
    if (!bone) continue;
    bone.getWorldPosition(v);
    out.set(name, [v.x, v.y, v.z]);
  }
  return out;
}

/** Provider reading the live skeleton in the avatar runtime store (falls back before the avatar has loaded). */
export const boneFocusTargetProvider: FocusTargetProvider = {
  getTargets: (heightM) =>
    computeBoneFocusTargets(worldPositions(useAvatarRuntimeStore.getState().bones), heightM),
};
