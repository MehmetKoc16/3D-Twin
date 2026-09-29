import { useEffect, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import type { PoseDef } from '@dt/avatar-core';
import { Quaternion } from 'three';
import { usePoseStore } from '../../store/poseStore';
import { useAvatarRuntimeStore } from '../../store/avatarRuntimeStore';
import { blendQuaternion, easeInOut, parsePoseDef } from './poseEngine';

const durationSeconds = 0.35;
const identity = new Quaternion();
const cache = new Map<string, Promise<PoseDef>>();

function loadPose(id: string): Promise<PoseDef> {
  let request = cache.get(id);
  if (!request) {
    request = fetch(`/assets/poses/${id}.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`Could not load pose ${id}: ${response.status}`);
        return response.json() as Promise<unknown>;
      })
      .then((data) => parsePoseDef(data, id))
      .catch((error: unknown) => {
        cache.delete(id);
        throw error;
      });
    cache.set(id, request);
  }
  return request;
}

interface Transition {
  from: Map<string, Quaternion>;
  to: Map<string, Quaternion>;
  elapsed: number;
  complete: boolean;
}

/** Mounted inside Canvas; writes the current pose even when the rest skeleton is rebuilt. */
export function PoseDriver(): null {
  const poseId = usePoseStore((state) => state.poseId);
  const transition = useRef<Transition | null>(null);

  useEffect(() => {
    let cancelled = false;
    void loadPose(poseId)
      .then((pose) => {
        if (cancelled) return;
        const from = new Map<string, Quaternion>();
        for (const [name, bone] of useAvatarRuntimeStore.getState().bones) {
          from.set(name, bone.quaternion.clone());
        }
        transition.current = {
          from,
          to: new Map(Object.entries(pose.bones).map(([name, q]) => [name, new Quaternion(...q)])),
          elapsed: 0,
          complete: false,
        };
      })
      .catch((error: unknown) => {
        if (!cancelled) console.error(error);
      });
    return () => {
      cancelled = true;
    };
  }, [poseId]);

  useFrame((_, delta) => {
    const current = transition.current;
    if (!current) return;
    current.elapsed = Math.min(durationSeconds, current.elapsed + delta);
    const factor = easeInOut(current.elapsed / durationSeconds);
    for (const [name, bone] of useAvatarRuntimeStore.getState().bones) {
      const target = current.to.get(name) ?? identity;
      if (current.complete) {
        bone.quaternion.copy(target);
      } else {
        let source = current.from.get(name);
        if (!source) {
          source = bone.quaternion.clone();
          current.from.set(name, source);
        }
        blendQuaternion(source, target, factor, bone.quaternion);
      }
    }
    current.complete = current.elapsed >= durationSeconds;
  });

  return null;
}
