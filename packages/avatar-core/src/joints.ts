import type { JointRef, RigDef, Vec3 } from './contracts';
import { AvatarCoreError } from './errors';

export interface JointPair {
  head: Vec3;
  tail: Vec3;
}

/**
 * Validates a rig: unique bone names, parents exist, no cycles, and (when `vertexCount` is given) every vertex
 * index is a valid integer < vertexCount. Throws AvatarCoreError.
 */
export function validateRig(rig: RigDef, vertexCount?: number): void {
  const names = new Set<string>();
  for (const b of rig.bones) {
    if (names.has(b.name)) throw new AvatarCoreError(`rig: duplicate bone "${b.name}"`);
    names.add(b.name);
  }
  for (const b of rig.bones) {
    if (b.parent !== null && !names.has(b.parent))
      throw new AvatarCoreError(`rig: bone "${b.name}" has unknown parent "${b.parent}"`);
    if (b.parent === b.name) throw new AvatarCoreError(`rig: bone "${b.name}" is its own parent`);
    for (const ref of [b.head, b.tail]) checkRef(b.name, ref, vertexCount);
  }
  boneOrder(rig); // detects cycles
}

function checkRef(bone: string, ref: JointRef, vertexCount: number | undefined): void {
  const checkVert = (v: number): void => {
    if (!Number.isInteger(v) || v < 0 || (vertexCount !== undefined && v >= vertexCount))
      throw new AvatarCoreError(`rig: bone "${bone}" references invalid vertex ${v}`);
  };
  switch (ref.strategy) {
    case 'MEAN':
      if (ref.verts.length === 0)
        throw new AvatarCoreError(`rig: bone "${bone}" has an empty MEAN joint`);
      ref.verts.forEach(checkVert);
      return;
    case 'VERTEX':
      return checkVert(ref.vert);
    case 'FIXED':
      if (!ref.position.every(Number.isFinite))
        throw new AvatarCoreError(`rig: bone "${bone}" has a non-finite FIXED position`);
      return;
  }
}

/**
 * Bone names in topological order (every parent before its children). Stable: among ready bones the original
 * rig order is kept. Throws on unknown parents and cycles.
 */
export function boneOrder(rig: RigDef): string[] {
  const byName = new Map(rig.bones.map((b) => [b.name, b]));
  const state = new Map<string, 0 | 1 | 2>(); // 1 = visiting, 2 = done
  const order: string[] = [];
  const visit = (name: string, path: string[]): void => {
    const s = state.get(name);
    if (s === 2) return;
    if (s === 1)
      throw new AvatarCoreError(`rig: cycle in parent chain: ${[...path, name].join(' -> ')}`);
    state.set(name, 1);
    const bone = byName.get(name);
    if (!bone) throw new AvatarCoreError(`rig: unknown bone "${name}"`);
    if (bone.parent !== null) {
      if (!byName.has(bone.parent))
        throw new AvatarCoreError(`rig: bone "${name}" has unknown parent "${bone.parent}"`);
      visit(bone.parent, [...path, name]);
    }
    state.set(name, 2);
    order.push(name);
  };
  for (const b of rig.bones) visit(b.name, []);
  return order;
}

function resolve(ref: JointRef, p: Float32Array | Float64Array | number[], out: Vec3): void {
  switch (ref.strategy) {
    case 'FIXED':
      out[0] = ref.position[0];
      out[1] = ref.position[1];
      out[2] = ref.position[2];
      return;
    case 'VERTEX': {
      const o = ref.vert * 3;
      out[0] = p[o]!;
      out[1] = p[o + 1]!;
      out[2] = p[o + 2]!;
      return;
    }
    case 'MEAN': {
      let x = 0;
      let y = 0;
      let z = 0;
      for (const v of ref.verts) {
        const o = v * 3;
        x += p[o]!;
        y += p[o + 1]!;
        z += p[o + 2]!;
      }
      const n = ref.verts.length;
      out[0] = x / n;
      out[1] = y / n;
      out[2] = z / n;
      return;
    }
  }
}

/**
 * Bone head/tail positions from the morphed mesh (meters, world space). The map iterates in rig order.
 * Vertex indices are in the combined (render + joint point) index space and are bounds-checked.
 */
export function computeJoints(rig: RigDef, positions: Float32Array): Map<string, JointPair> {
  validateRig(rig, Math.floor(positions.length / 3));
  const out = new Map<string, JointPair>();
  for (const b of rig.bones) {
    const pair: JointPair = { head: [0, 0, 0], tail: [0, 0, 0] };
    resolve(b.head, positions, pair.head);
    resolve(b.tail, positions, pair.tail);
    out.set(b.name, pair);
  }
  return out;
}
