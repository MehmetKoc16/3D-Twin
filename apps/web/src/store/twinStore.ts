import { del, get, set } from 'idb-keyval';
import { create } from 'zustand';
import {
  parseTwinJson,
  TwinFormatError,
  type TwinDef,
  type TwinErrorCode,
} from '../features/twin/twinDef';
import { parseMapping } from '../features/twin/twinMapping';
import { loadTwinBundle } from '../features/twin/twinBundle';

/**
 * The realistic twin: the user's own rigged scan (`rigged.glb` + `twin.json` + optional `mh2twin.bin`, produced locally
 * by tools/twin-lab/rig). PRIVACY: the files are picked with a file input, kept in memory and in this browser's
 * IndexedDB (`twin:*` keys) and can be removed again. Nothing is bundled, fetched or uploaded.
 */

export type ModelMode = 'standard' | 'twin';
export type TwinStatus = 'empty' | 'partial' | 'ready' | 'error';

/** IndexedDB keys (idb-keyval). */
export const TWIN_KEYS = {
  mode: 'twin:mode',
  glb: 'twin:glb',
  json: 'twin:json',
  mapping: 'twin:mapping',
  names: 'twin:names',
} as const;

export const MAX_TWIN_BYTES = 200 * 1024 * 1024;

export interface TwinPack {
  def: TwinDef;
  /** Present for a single-file bundle; legacy packs use the scan material colour. */
  skinToneHex?: string;
  /** Bundle flag: the twin's own hands are MakeHuman hands, keep them (no mannequin-hand swap). */
  hasMakeHumanHands?: boolean;
  /** rigged.glb bytes. */
  glb: ArrayBuffer;
  /** Twin vertex -> nearest body render vertex; null when no mapping file was given. */
  mapping: Uint32Array | null;
  names: { glb: string; json: string; mapping: string | null };
}

export interface TwinError {
  code: TwinErrorCode | 'unknown' | 'files';
  detail?: string;
}

/** What the running twin rig reports back (shown in the twin panel). */
export interface TwinRuntimeInfo {
  vertices: number;
  triangles: number;
  hiddenTriangles: number;
  /** Rest-frame translation applied to the twin, cm. */
  offsetCm: [number, number, number];
  /** Largest rest-head difference after that translation, mm. */
  residualMm: number;
}

interface PendingFiles {
  glb?: { name: string; buffer: ArrayBuffer };
  json?: { name: string; text: string };
  mapping?: { name: string; buffer: ArrayBuffer };
}

interface TwinState {
  mode: ModelMode;
  status: TwinStatus;
  error: TwinError | null;
  pack: TwinPack | null;
  pending: PendingFiles;
  runtime: TwinRuntimeInfo | null;
  /** Incremented whenever the pack is replaced or removed, so the rig reloads. */
  revision: number;
  hydrated: boolean;
  /** Takes the picked files (any subset of rigged.glb, twin.json, mh2twin.bin) and builds the pack once complete. */
  loadFiles: (files: readonly File[]) => Promise<void>;
  setMode: (mode: ModelMode) => void;
  remove: () => Promise<void>;
  /** The rig could not show the twin: report the reason and fall back to the standard model. */
  fail: (error: TwinError) => void;
  setRuntime: (runtime: TwinRuntimeInfo | null) => void;
  hydrate: () => Promise<void>;
}

async function persist(entries: [string, unknown][]): Promise<void> {
  try {
    await Promise.all(entries.map(([key, value]) => set(key, value)));
  } catch {
    /* storage unavailable (private mode / quota): the session still works */
  }
}

function toError(error: unknown): TwinError {
  if (error instanceof TwinFormatError) return { code: error.code, detail: error.message };
  return { code: 'unknown', detail: error instanceof Error ? error.message : String(error) };
}

const GLB_MAGIC = 0x46546c67; // "glTF"

/** Builds a pack from the raw pieces; throws `TwinFormatError`. */
export function buildPack(pending: PendingFiles): TwinPack | null {
  if (!pending.glb || !pending.json) return null;
  if (
    pending.glb.buffer.byteLength < 20 ||
    new DataView(pending.glb.buffer).getUint32(0, true) !== GLB_MAGIC
  )
    throw new TwinFormatError('glb', 'rigged.glb is not a binary glTF file');
  const def = parseTwinJson(pending.json.text);
  let mapping: Uint32Array | null = null;
  if (pending.mapping) {
    mapping = parseMapping(pending.mapping.buffer);
    if (def.mapping && mapping.length !== def.mapping.twinVertexCount)
      throw new TwinFormatError(
        'mapping',
        `mh2twin.bin has ${mapping.length} entries, twin.json expects ${def.mapping.twinVertexCount}`,
      );
  }
  return {
    def,
    glb: pending.glb.buffer,
    mapping,
    names: {
      glb: pending.glb.name,
      json: pending.json.name,
      mapping: pending.mapping?.name ?? null,
    },
  };
}

async function buildPickedPack(pending: PendingFiles): Promise<TwinPack | null> {
  if (pending.glb) {
    const bundle = await loadTwinBundle(pending.glb.buffer);
    if (bundle)
      return {
        ...bundle,
        glb: pending.glb.buffer,
        names: {
          glb: pending.glb.name,
          json: 'asset.extras.dtTwin.twin',
          mapping: 'embedded mh2twin',
        },
      };
  }
  return buildPack(pending);
}

const empty = {
  status: 'empty' as TwinStatus,
  error: null,
  pack: null,
  pending: {} as PendingFiles,
  runtime: null,
};

export const useTwinStore = create<TwinState>((update, read) => ({
  mode: 'standard',
  ...empty,
  revision: 0,
  hydrated: false,
  loadFiles: async (files) => {
    const pending: PendingFiles = { ...read().pending };
    try {
      for (const file of files) {
        if (file.size > MAX_TWIN_BYTES)
          throw new TwinFormatError('glb', `${file.name} is too large`);
        const name = file.name.toLowerCase();
        if (name.endsWith('.glb'))
          pending.glb = { name: file.name, buffer: await file.arrayBuffer() };
        else if (name.endsWith('.json'))
          pending.json = { name: file.name, text: await file.text() };
        else if (name.endsWith('.bin'))
          pending.mapping = { name: file.name, buffer: await file.arrayBuffer() };
        else
          throw new TwinFormatError(
            'glb',
            `${file.name}: expected rigged.glb, twin.json or mh2twin.bin`,
          );
      }
      const pack = await buildPickedPack(pending);
      if (!pack) {
        update({ pending, status: 'partial', error: null });
        return;
      }
      update((state) => ({
        pending: {},
        pack,
        status: 'ready',
        error: null,
        runtime: null,
        mode: 'twin',
        revision: state.revision + 1,
      }));
      await persist([
        [TWIN_KEYS.glb, pack.glb],
        [TWIN_KEYS.json, pack.skinToneHex ? null : pending.json?.text],
        [TWIN_KEYS.mapping, pack.skinToneHex ? null : (pending.mapping?.buffer ?? null)],
        [TWIN_KEYS.names, pack.names],
        [TWIN_KEYS.mode, 'twin'],
      ]);
    } catch (error) {
      update({ pending, status: 'error', error: toError(error) });
    }
  },
  setMode: (mode) => {
    if (mode === 'twin' && !read().pack) return;
    update({ mode });
    void persist([[TWIN_KEYS.mode, mode]]);
  },
  remove: async () => {
    update((state) => ({ ...empty, mode: 'standard', revision: state.revision + 1 }));
    try {
      await Promise.all(Object.values(TWIN_KEYS).map((key) => del(key)));
    } catch {
      /* ignore */
    }
  },
  fail: (error) => {
    update((state) => ({
      mode: 'standard',
      status: 'error',
      error,
      runtime: null,
      revision: state.revision,
    }));
    void persist([[TWIN_KEYS.mode, 'standard']]);
  },
  setRuntime: (runtime) => update({ runtime }),
  hydrate: async () => {
    if (read().hydrated) return;
    update({ hydrated: true });
    try {
      const [glb, json, mapping, names, mode] = await Promise.all([
        get<unknown>(TWIN_KEYS.glb),
        get<unknown>(TWIN_KEYS.json),
        get<unknown>(TWIN_KEYS.mapping),
        get<unknown>(TWIN_KEYS.names),
        get<unknown>(TWIN_KEYS.mode),
      ]);
      if (read().pack || Object.keys(read().pending).length > 0) return; // the user picked files meanwhile
      if (!(glb instanceof ArrayBuffer)) return;
      const stored = names as
        Partial<Record<'glb' | 'json' | 'mapping', string | null>> | undefined;
      const pack = await buildPickedPack({
        glb: { name: stored?.glb ?? 'rigged.glb', buffer: glb },
        ...(typeof json === 'string'
          ? { json: { name: stored?.json ?? 'twin.json', text: json } }
          : {}),
        ...(mapping instanceof ArrayBuffer
          ? { mapping: { name: stored?.mapping ?? 'mh2twin.bin', buffer: mapping } }
          : {}),
      });
      if (!pack) return;
      if (read().pack || Object.keys(read().pending).length > 0) return;
      update((state) => ({
        pack,
        status: 'ready',
        error: null,
        mode: mode === 'twin' ? 'twin' : 'standard',
        revision: state.revision + 1,
      }));
    } catch {
      /* nothing stored, unreadable, or storage unavailable */
    }
  },
}));

/** True while the twin is the model shown (a twin is loaded and selected). */
export function isTwinActive(state: Pick<TwinState, 'mode' | 'pack'>): boolean {
  return state.mode === 'twin' && state.pack !== null;
}
