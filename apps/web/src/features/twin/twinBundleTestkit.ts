/** Synthetic, image-free bundle for unit tests. No personal data or renderer required. */
export function bundleExtras() {
  return {
    version: 1,
    twin: {
      version: 1,
      boneOrder: ['Root'],
      fittedMacros: {},
      fittedModifiers: {},
      measurementsCm: { height: 170 },
      mapping: { file: 'mh2twin.bin', twinVertexCount: 2, renderVertexCount: 10 },
    },
    skinToneHex: '#c49a7a',
    mh2twin: { bufferView: 0, count: 2, componentType: 'uint32' },
    provenance: { shape: 'synthetic unit test', license: 'CC0', createdAt: '2026-10-01T00:00:00Z' },
  };
}

export function bundleBytes(external = false): ArrayBuffer {
  const raw = new TextEncoder().encode(
    JSON.stringify({
      asset: { version: '2.0', extras: { dtTwin: bundleExtras() } },
      scenes: [{ nodes: [] }],
      scene: 0,
      buffers: [
        { byteLength: 8, ...(external ? { uri: 'https://example.invalid/private.bin' } : {}) },
      ],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 8 }],
    }),
  );
  const json = new Uint8Array(Math.ceil(raw.length / 4) * 4).fill(32);
  json.set(raw);
  const out = new ArrayBuffer(28 + json.length + 8);
  const view = new DataView(out);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, out.byteLength, true);
  view.setUint32(12, json.length, true);
  view.setUint32(16, 0x4e4f534a, true);
  new Uint8Array(out, 20, json.length).set(json);
  view.setUint32(20 + json.length, 8, true);
  view.setUint32(24 + json.length, 0x004e4942, true);
  view.setUint32(28 + json.length, 1, true);
  view.setUint32(32 + json.length, 3, true);
  return out;
}
