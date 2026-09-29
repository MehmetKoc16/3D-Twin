/** Minimal GLB reader for tests: returns POSITION (Float32Array) and indices of the first primitive. */
export interface GlbMesh {
  positions: Float32Array;
  indices: Uint32Array;
}

interface Accessor {
  bufferView: number;
  byteOffset?: number;
  componentType: number;
  count: number;
  type: string;
}
interface BufferView {
  byteOffset?: number;
  byteStride?: number;
}
interface GltfJson {
  accessors: Accessor[];
  bufferViews: BufferView[];
  meshes: { primitives: { attributes: Record<string, number>; indices?: number }[] }[];
}

const COMPONENTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

function readAccessor(json: GltfJson, bin: DataView, index: number): number[] {
  const acc = json.accessors[index]!;
  const view = json.bufferViews[acc.bufferView]!;
  const n = COMPONENTS[acc.type]!;
  const size = acc.componentType === 5126 || acc.componentType === 5125 ? 4 : 2;
  const stride = view.byteStride ?? n * size;
  const base = (view.byteOffset ?? 0) + (acc.byteOffset ?? 0);
  const out: number[] = [];
  for (let i = 0; i < acc.count; i++)
    for (let c = 0; c < n; c++) {
      const o = base + i * stride + c * size;
      out.push(
        acc.componentType === 5126
          ? bin.getFloat32(o, true)
          : acc.componentType === 5125
            ? bin.getUint32(o, true)
            : bin.getUint16(o, true),
      );
    }
  return out;
}

export function parseGlbMesh(buffer: ArrayBuffer): GlbMesh {
  const dv = new DataView(buffer);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a GLB');
  let offset = 12;
  let json: GltfJson | undefined;
  let bin: DataView | undefined;
  while (offset < buffer.byteLength) {
    const len = dv.getUint32(offset, true);
    const type = dv.getUint32(offset + 4, true);
    const start = offset + 8;
    if (type === 0x4e4f534a)
      json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, start, len))) as GltfJson;
    else if (type === 0x004e4942) bin = new DataView(buffer, start, len);
    offset = start + len;
  }
  if (!json || !bin) throw new Error('GLB chunks missing');
  const prim = json.meshes[0]!.primitives[0]!;
  return {
    positions: Float32Array.from(readAccessor(json, bin, prim.attributes.POSITION!)),
    indices: Uint32Array.from(readAccessor(json, bin, prim.indices!)),
  };
}
