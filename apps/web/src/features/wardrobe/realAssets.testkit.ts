// @ts-expect-error Node types are not in the browser app's tsconfig; Vitest supplies this module.
import { existsSync, readFileSync } from 'node:fs';

/** Shared loaders for the tests that run against the real body and garment assets (skipped when not built). */

export const root = ['public/assets', 'apps/web/public/assets'].find(
  (d) => existsSync(`${d}/garments/index.json`) && existsSync(`${d}/body/manifest.json`),
);

export interface Glb {
  json: {
    meshes: { primitives: { attributes: Record<string, number>; indices?: number }[] }[];
    accessors: {
      bufferView: number;
      byteOffset?: number;
      componentType: number;
      count: number;
      type: string;
    }[];
    bufferViews: { byteOffset?: number; byteLength: number; byteStride?: number }[];
  };
  bin: DataView;
}

export function parseGlb(buffer: Uint8Array): Glb {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const jsonLength = view.getUint32(12, true);
  const json = JSON.parse(
    new TextDecoder().decode(buffer.subarray(20, 20 + jsonLength)),
  ) as Glb['json'];
  const binStart = 20 + jsonLength + 8;
  return {
    json,
    bin: new DataView(buffer.buffer, buffer.byteOffset + binStart, buffer.byteLength - binStart),
  };
}

const COMPONENTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

function readAccessor(glb: Glb, index: number): Float64Array {
  const accessor = glb.json.accessors[index]!;
  const view = glb.json.bufferViews[accessor.bufferView]!;
  const n = COMPONENTS[accessor.type]!;
  const size = accessor.componentType === 5126 || accessor.componentType === 5125 ? 4 : 2;
  const stride = view.byteStride ?? size * n;
  const out = new Float64Array(accessor.count * n);
  for (let i = 0; i < accessor.count; i++) {
    for (let k = 0; k < n; k++) {
      const offset = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0) + i * stride + k * size;
      out[i * n + k] =
        accessor.componentType === 5126
          ? glb.bin.getFloat32(offset, true)
          : accessor.componentType === 5125
            ? glb.bin.getUint32(offset, true)
            : glb.bin.getUint16(offset, true);
    }
  }
  return out;
}

export function primitive(glb: Glb): {
  position: Float32Array;
  indices: Uint32Array;
  joints?: Float64Array;
  weights?: Float64Array;
} {
  const attributes = glb.json.meshes[0]!.primitives[0]!.attributes;
  const p = glb.json.meshes[0]!.primitives[0]!;
  return {
    position: Float32Array.from(readAccessor(glb, attributes.POSITION!)),
    indices: Uint32Array.from(readAccessor(glb, p.indices!)),
    ...(attributes.JOINTS_0 !== undefined
      ? { joints: readAccessor(glb, attributes.JOINTS_0) }
      : {}),
    ...(attributes.WEIGHTS_0 !== undefined
      ? { weights: readAccessor(glb, attributes.WEIGHTS_0) }
      : {}),
  };
}

export function bytes(path: string): ArrayBuffer {
  const b = readFileSync(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
