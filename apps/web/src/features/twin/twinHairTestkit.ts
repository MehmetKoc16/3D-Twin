import { DataTexture, NoColorSpace } from 'three';
import { writeTestGlb } from './twinAccessoryTestkit';
import { HAIR_FORMAT, type TwinHairInfo, type TwinHairModel } from './twinHair';

/** Synthetic, image-free hair fixtures for the unit tests (nothing here derives from a person). */

export const hairInfo: TwinHairInfo = {
  nodeName: 'dtHair',
  format: HAIR_FORMAT,
  colorHex: '#2a1e18',
  rootHex: '#1f1611',
  tipHex: '#3a2a20',
  cardCount: 1,
};

/** JSON-level extras as the pipeline writes them. */
export const hairExtras = {
  format: HAIR_FORMAT,
  colorHex: '#2a1e18',
  rootHex: '#1f1611',
  tipHex: '#3a2a20',
  cardCount: 1,
};

/** The glTF fragments `parseTwinHair` reads: node -> mesh -> primitive -> material with `extras.dtHair`. */
export function hairJson(overrides: { extras?: unknown; node?: unknown } = {}) {
  return {
    asset: { extras: { dtHairNode: overrides.node ?? 'dtHair' } },
    nodes: [
      { name: 'twin', mesh: 0 },
      { name: 'dtHair', mesh: 1 },
    ],
    meshes: [{ primitives: [{ material: 0 }] }, { primitives: [{ material: 1 }] }],
    materials: [
      { name: 'twin' },
      { name: 'dtHair', extras: { dtHair: 'extras' in overrides ? overrides.extras : hairExtras } },
    ],
  };
}

/** One hair card (a triangle) skinned to the last bone (`spine_01`) of the fake rig of `twinTestkit`. */
export function fakeHairModel(): TwinHairModel & { disposed: { count: number } } {
  const disposed = { count: 0 };
  const atlas = new DataTexture(new Uint8Array([255, 0, 128, 255]), 1, 1);
  atlas.colorSpace = NoColorSpace;
  return {
    info: hairInfo,
    vertexCount: 3,
    position: Float32Array.from([0, 1.5, 0.1, 0.1, 1.5, 0.1, 0, 1.6, 0.1]),
    normal: Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uv: Float32Array.from([0, 0, 1, 0, 0, 1]),
    index: Uint32Array.from([0, 1, 2]),
    skinIndex: Uint16Array.from([2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0]),
    skinWeight: Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
    atlas,
    disposed,
    dispose: () => {
      disposed.count++;
      atlas.dispose();
    },
  };
}

// 1x1 PNG
const PNG = Uint8Array.from(
  atob(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  )
    .split('')
    .map((c) => c.charCodeAt(0)),
);

/**
 * A GLB with a one-bone skin (`Root`), a body triangle (node `twin`) and, unless `hair` is false, a hair card (node
 * `dtHair`) on the SAME skin with an embedded atlas image and `asset.extras.dtHairNode`.
 */
export function hairGlb(
  options: { hair?: boolean; node?: string; extras?: unknown } = {},
): ArrayBuffer {
  const withHair = options.hair !== false;
  const f32 = (...v: number[]) => new Uint8Array(Float32Array.from(v).buffer);
  const u16 = (...v: number[]) => new Uint8Array(Uint16Array.from(v).buffer);
  const u32 = (...v: number[]) => new Uint8Array(Uint32Array.from(v).buffer);
  const arrays: Uint8Array[] = [
    f32(0, 0, 0, 1, 0, 0, 0, 1, 0), // 0 body position
    f32(0, 0, 1, 0, 0, 1, 0, 0, 1), // 1 normal
    u16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), // 2 joints
    f32(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0), // 3 weights
    u32(0, 1, 2), // 4 indices
    f32(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), // 5 inverse bind (Root head at the origin)
    f32(0, 1.5, 0, 0.1, 1.5, 0, 0, 1.6, 0), // 6 hair position
    f32(0, 0, 0, 1, 0, 1), // 7 hair uv
    PNG, // 8 atlas image
  ];
  let offset = 0;
  const views = arrays.map((a) => {
    const view = { buffer: 0, byteOffset: offset, byteLength: a.byteLength };
    offset += Math.ceil(a.byteLength / 4) * 4;
    return view;
  });
  const bin = new Uint8Array(offset);
  arrays.forEach((a, i) => bin.set(a, views[i]!.byteOffset));
  const body = {
    attributes: { POSITION: 0, NORMAL: 1, JOINTS_0: 2, WEIGHTS_0: 3 },
    indices: 4,
    material: 0,
  };
  const hair = {
    attributes: { POSITION: 6, NORMAL: 1, TEXCOORD_0: 7, JOINTS_0: 2, WEIGHTS_0: 3 },
    indices: 4,
    material: 1,
  };
  return writeTestGlb(
    {
      asset: {
        version: '2.0',
        ...(withHair ? { extras: { dtHairNode: options.node ?? 'dtHair' } } : {}),
      },
      scene: 0,
      scenes: [{ nodes: [0, 1, ...(withHair ? [2] : [])] }],
      nodes: [
        { name: 'Root' },
        { name: 'twin', mesh: 0, skin: 0 },
        ...(withHair ? [{ name: 'dtHair', mesh: 1, skin: 0 }] : []),
      ],
      skins: [{ joints: [0], inverseBindMatrices: 5 }],
      meshes: [{ primitives: [body] }, ...(withHair ? [{ primitives: [hair] }] : [])],
      materials: [
        { name: 'twin', pbrMetallicRoughness: { baseColorFactor: [0.8, 0.6, 0.5, 1] } },
        ...(withHair
          ? [
              {
                name: 'dtHair',
                doubleSided: true,
                alphaMode: 'MASK',
                alphaCutoff: 0.5,
                pbrMetallicRoughness: { baseColorTexture: { index: 0 } },
                extras: { dtHair: 'extras' in options ? options.extras : hairExtras },
              },
            ]
          : []),
      ],
      textures: [{ source: 0 }],
      images: [{ bufferView: 8, mimeType: 'image/png' }],
      buffers: [{ byteLength: bin.length }],
      bufferViews: views,
      accessors: [
        {
          bufferView: 0,
          componentType: 5126,
          count: 3,
          type: 'VEC3',
          min: [0, 0, 0],
          max: [1, 1, 0],
        },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 2, componentType: 5123, count: 3, type: 'VEC4' },
        { bufferView: 3, componentType: 5126, count: 3, type: 'VEC4' },
        { bufferView: 4, componentType: 5125, count: 3, type: 'SCALAR' },
        { bufferView: 5, componentType: 5126, count: 1, type: 'MAT4' },
        {
          bufferView: 6,
          componentType: 5126,
          count: 3,
          type: 'VEC3',
          min: [0, 1.5, 0],
          max: [0.1, 1.6, 0],
        },
        { bufferView: 7, componentType: 5126, count: 3, type: 'VEC2' },
      ],
    },
    bin,
  );
}

/** Rewrites the JSON chunk of a GLB (the BIN chunk is kept), e.g. to drop `asset.extras` in a test. */
export function editGlbJson(
  buffer: ArrayBuffer,
  edit: (json: Record<string, unknown>) => void,
): ArrayBuffer {
  const view = new DataView(buffer);
  const jsonLength = view.getUint32(12, true);
  const json = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLength)),
  ) as Record<string, unknown>;
  edit(json);
  const binLength = view.getUint32(20 + jsonLength, true);
  return writeTestGlb(json, new Uint8Array(buffer, 28 + jsonLength, binLength));
}
