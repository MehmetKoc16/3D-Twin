import type { PoseDef } from '@dt/avatar-core';
import { Quaternion } from 'three';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** Reject malformed downloads before any skeleton rotation is applied. */
export function parsePoseDef(value: unknown, expectedId: string): PoseDef {
  if (
    !isRecord(value) ||
    value.id !== expectedId ||
    !isRecord(value.label) ||
    typeof value.label.tr !== 'string' ||
    typeof value.label.en !== 'string' ||
    !isRecord(value.bones)
  ) {
    throw new Error(`Invalid pose ${expectedId}`);
  }
  for (const [name, quaternion] of Object.entries(value.bones)) {
    if (
      !name ||
      !Array.isArray(quaternion) ||
      quaternion.length !== 4 ||
      !quaternion.every(
        (component) => typeof component === 'number' && Number.isFinite(component),
      ) ||
      Math.abs(Math.hypot(...quaternion) - 1) > 0.002
    ) {
      throw new Error(`Invalid quaternion for ${name} in ${expectedId}`);
    }
  }
  return value as unknown as PoseDef;
}

export function easeInOut(t: number): number {
  const clamped = Math.max(0, Math.min(1, t));
  return clamped * clamped * (3 - 2 * clamped);
}

/** three.js slerp handles the equivalent q/-q pair via the shortest arc. */
export function blendQuaternion(
  from: Quaternion,
  to: Quaternion,
  t: number,
  out = new Quaternion(),
): Quaternion {
  return out
    .copy(from)
    .slerp(to, Math.max(0, Math.min(1, t)))
    .normalize();
}
