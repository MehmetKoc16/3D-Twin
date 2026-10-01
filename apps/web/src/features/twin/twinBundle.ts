import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { Mesh, MeshStandardMaterial } from 'three';
import { parseTwinDef, TwinFormatError, type TwinDef } from './twinDef';
import { parseMapping, validateMapping } from './twinMapping';

interface BundleParser {
  json: { asset?: { extras?: { dtTwin?: unknown } }; bufferViews?: unknown[] };
  getDependency(type: 'bufferView', index: number): Promise<ArrayBuffer>;
}

export interface TwinBundle {
  def: TwinDef;
  mapping: Uint32Array;
  skinToneHex: string;
}

const record = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

/** The contract lives on asset.extras, not the scene's userData. */
export async function parseTwinBundle(parser: BundleParser): Promise<TwinBundle | null> {
  const raw = parser.json.asset?.extras?.dtTwin;
  if (raw === undefined) return null;
  if (!record(raw)) throw new TwinFormatError('json', 'invalid dtTwin extras');
  if (raw.version !== 1) throw new TwinFormatError('version', 'unsupported dtTwin version');
  const def = parseTwinDef(raw.twin);
  if (typeof raw.skinToneHex !== 'string' || !/^#[\da-f]{6}$/i.test(raw.skinToneHex))
    throw new TwinFormatError('json', 'dtTwin.skinToneHex must be #rrggbb');
  const provenance = raw.provenance;
  if (
    !record(provenance) ||
    typeof provenance.shape !== 'string' ||
    !provenance.shape ||
    typeof provenance.license !== 'string' ||
    !provenance.license ||
    typeof provenance.createdAt !== 'string' ||
    !/^\d{4}-\d{2}-\d{2}T/.test(provenance.createdAt) ||
    !Number.isFinite(Date.parse(provenance.createdAt))
  )
    throw new TwinFormatError('json', 'invalid dtTwin provenance');
  const map = raw.mh2twin;
  if (
    !record(map) ||
    map.componentType !== 'uint32' ||
    !Number.isSafeInteger(map.bufferView) ||
    (map.bufferView as number) < 0 ||
    (map.bufferView as number) >= (parser.json.bufferViews?.length ?? 0) ||
    !Number.isSafeInteger(map.count) ||
    (map.count as number) <= 0
  )
    throw new TwinFormatError('mapping', 'invalid dtTwin.mh2twin');
  let buffer: ArrayBuffer;
  try {
    buffer = await parser.getDependency('bufferView', map.bufferView as number);
  } catch {
    throw new TwinFormatError('mapping', 'unreadable mapping bufferView');
  }
  if (buffer.byteLength !== (map.count as number) * 4)
    throw new TwinFormatError('mapping', 'mapping byte length does not match count');
  const mapping = parseMapping(buffer);
  if (def.mapping)
    validateMapping(mapping, def.mapping.twinVertexCount, def.mapping.renderVertexCount);
  return { def, mapping, skinToneHex: raw.skinToneHex };
}

/** Inspect only the JSON header before invoking the loader; legacy GLBs need no second parse. */
export async function loadTwinBundle(buffer: ArrayBuffer): Promise<TwinBundle | null> {
  const view = new DataView(buffer);
  if (buffer.byteLength < 20 || view.getUint32(0, true) !== 0x46546c67)
    throw new TwinFormatError('glb', 'not a binary glTF file');
  if (view.getUint32(4, true) !== 2) return null; // legacy validation remains in buildPack / model loader
  if (
    view.getUint32(8, true) !== buffer.byteLength ||
    view.getUint32(16, true) !== 0x4e4f534a ||
    view.getUint32(12, true) > buffer.byteLength - 20
  )
    throw new TwinFormatError('glb', 'invalid GLB header');
  let rawJson: unknown;
  try {
    rawJson = JSON.parse(
      new TextDecoder().decode(new Uint8Array(buffer, 20, view.getUint32(12, true))),
    );
  } catch {
    throw new TwinFormatError('json', 'invalid GLB JSON');
  }
  if (!record(rawJson)) throw new TwinFormatError('json', 'GLB JSON must be an object');
  const json = rawJson;
  if (!record(json.asset) || !record(json.asset.extras) || !('dtTwin' in json.asset.extras))
    return null;
  // Bundles are self-contained. Never resolve a URL from a picked bundle.
  for (const list of [json.buffers, json.images]) {
    if (Array.isArray(list) && list.some((v: unknown) => record(v) && v.uri !== undefined))
      throw new TwinFormatError('glb', 'twin.glb must embed buffers and images');
  }
  const gltf = await new GLTFLoader().parseAsync(buffer.slice(0), '').catch((error: unknown) => {
    throw new TwinFormatError(
      'glb',
      error instanceof Error ? error.message : 'twin.glb could not be parsed',
    );
  });
  try {
    return await parseTwinBundle(gltf.parser);
  } finally {
    gltf.scene.traverse((child) => {
      if (!(child instanceof Mesh)) return;
      child.geometry.dispose();
      for (const material of Array.isArray(child.material) ? child.material : [child.material]) {
        if (material instanceof MeshStandardMaterial) material.map?.dispose();
        material.dispose();
      }
    });
  }
}
