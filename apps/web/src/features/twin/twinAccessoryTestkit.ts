/** Procedural, image-free glasses triangle for tests; never derived from a person's scan. */
export const accessoryParams = {
  lensShape: 'round',
  outerRadius: 0.024,
  bridgeWidth: 0.018,
  frameWidth: 0.135,
  thickness: 0.0012,
  templeLength: 0.14,
  colour: '#b9b4ad',
  verticalOffset: 0,
};

export function writeTestGlb(json: Record<string, unknown>, bin: Uint8Array): ArrayBuffer {
  const raw = new TextEncoder().encode(JSON.stringify(json));
  const encoded = new Uint8Array(Math.ceil(raw.length / 4) * 4).fill(32);
  encoded.set(raw);
  const padded = new Uint8Array(Math.ceil(bin.length / 4) * 4);
  padded.set(bin);
  const out = new ArrayBuffer(28 + encoded.length + padded.length);
  const view = new DataView(out);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, out.byteLength, true);
  view.setUint32(12, encoded.length, true);
  view.setUint32(16, 0x4e4f534a, true);
  new Uint8Array(out, 20, encoded.length).set(encoded);
  view.setUint32(20 + encoded.length, padded.length, true);
  view.setUint32(24 + encoded.length, 0x004e4942, true);
  new Uint8Array(out, 28 + encoded.length).set(padded);
  return out;
}

export function accessoryBytes(
  options: { lenses?: boolean; external?: boolean; invalidWeights?: boolean } = {},
): ArrayBuffer {
  const position = new Float32Array([-0.02, 0.13, 0.11, 0.02, 0.13, 0.11, 0, 0.15, 0.11]);
  const normal = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  const joints = new Uint16Array(12);
  const weights = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
  if (options.invalidWeights) {
    weights[0] = 0.5;
    weights[1] = 0.5;
  }
  const indices = new Uint32Array([0, 1, 2]);
  const inverse = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  const arrays = [position, normal, joints, weights, indices, inverse];
  let offset = 0;
  const views = arrays.map((array) => {
    const result = { buffer: 0, byteOffset: offset, byteLength: array.byteLength };
    offset += array.byteLength;
    return result;
  });
  const bin = new Uint8Array(offset);
  arrays.forEach((array, i) => bin.set(new Uint8Array(array.buffer), views[i]!.byteOffset));
  const primitive = {
    attributes: { POSITION: 0, NORMAL: 1, JOINTS_0: 2, WEIGHTS_0: 3 },
    indices: 4,
    material: 0,
  };
  return writeTestGlb(
    {
      asset: {
        version: '2.0',
        extras: {
          dtAccessory: {
            id: 'glasses',
            bone: 'head',
            coordinateSpace: 'head-local',
            params: accessoryParams,
          },
        },
      },
      scene: 0,
      scenes: [{ nodes: [0, 1] }],
      nodes: [{ name: 'head' }, { name: 'glasses', mesh: 0, skin: 0 }],
      skins: [{ joints: [0], inverseBindMatrices: 5 }],
      meshes: [
        { primitives: options.lenses ? [primitive, { ...primitive, material: 1 }] : [primitive] },
      ],
      materials: [
        {
          pbrMetallicRoughness: {
            baseColorFactor: [0.485, 0.456, 0.418, 1],
            metallicFactor: 1,
            roughnessFactor: 0.3,
          },
        },
        {
          alphaMode: 'BLEND',
          doubleSided: true,
          pbrMetallicRoughness: {
            baseColorFactor: [0.9, 0.96, 1, 0.035],
            metallicFactor: 0,
            roughnessFactor: 0.08,
          },
        },
      ],
      buffers: [
        {
          byteLength: bin.length,
          ...(options.external ? { uri: 'https://example.invalid/accessory.bin' } : {}),
        },
      ],
      bufferViews: views,
      accessors: [
        {
          bufferView: 0,
          componentType: 5126,
          count: 3,
          type: 'VEC3',
          min: [-0.02, 0.13, 0.11],
          max: [0.02, 0.15, 0.11],
        },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 2, componentType: 5123, count: 3, type: 'VEC4' },
        { bufferView: 3, componentType: 5126, count: 3, type: 'VEC4' },
        { bufferView: 4, componentType: 5125, count: 3, type: 'SCALAR' },
        { bufferView: 5, componentType: 5126, count: 1, type: 'MAT4' },
      ],
    },
    bin,
  );
}

/** Adds the synthetic accessory to a synthetic or CC0 bundle in memory. */
export function appendTestAccessory(
  buffer: ArrayBuffer,
  accessory = accessoryBytes(),
): ArrayBuffer {
  const view = new DataView(buffer);
  const jsonLength = view.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLength))) as {
    asset: { extras: { dtTwin: { accessories?: unknown[] } } };
    bufferViews: { buffer: number; byteOffset?: number; byteLength: number }[];
    buffers: { byteLength: number }[];
  };
  const length = json.buffers[0]!.byteLength;
  const start = Math.ceil(length / 4) * 4;
  const bin = new Uint8Array(start + accessory.byteLength);
  bin.set(new Uint8Array(buffer, 28 + jsonLength, length));
  bin.set(new Uint8Array(accessory), start);
  const index = json.bufferViews.length;
  json.bufferViews.push({ buffer: 0, byteOffset: start, byteLength: accessory.byteLength });
  json.buffers[0]!.byteLength = bin.length;
  json.asset.extras.dtTwin.accessories = [
    { id: 'glasses', bone: 'head', mesh: { bufferView: index }, params: accessoryParams },
  ];
  return writeTestGlb(json, bin);
}
