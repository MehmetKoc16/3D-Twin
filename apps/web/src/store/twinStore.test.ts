import { beforeEach, describe, expect, it, vi } from 'vitest';

const db = new Map<string, unknown>();
let failing = false;
vi.mock('idb-keyval', () => ({
  get: vi.fn(async (key: string) => {
    if (failing) throw new Error('storage unavailable');
    return db.get(key);
  }),
  set: vi.fn(async (key: string, value: unknown) => {
    if (failing) throw new Error('storage unavailable');
    db.set(key, value);
  }),
  del: vi.fn(async (key: string) => void db.delete(key)),
}));

import { buildPack, isTwinActive, TWIN_KEYS, useTwinStore } from './twinStore';
import { bundleBytes } from '../features/twin/twinBundleTestkit';

const pristine = useTwinStore.getState();
beforeEach(() => {
  db.clear();
  failing = false;
  useTwinStore.setState({ ...pristine, hydrated: false }, true);
});

const twinJson = JSON.stringify({
  version: 1,
  boneOrder: ['Root', 'pelvis'],
  fittedMacros: { gender: 0.5 },
  fittedModifiers: {},
  measurementsCm: { height: 170 },
  mapping: { file: 'mh2twin.bin', twinVertexCount: 2, renderVertexCount: 10 },
});

function glbBytes(): ArrayBuffer {
  const bytes = new Uint8Array(32);
  new DataView(bytes.buffer).setUint32(0, 0x46546c67, true); // "glTF"
  return bytes.buffer;
}

function file(name: string, content: BlobPart): File {
  return new File([content], name);
}

const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

describe('twin store', () => {
  it('loads one twin.glb and restores its embedded definition, mapping and skin tone from IndexedDB', async () => {
    await useTwinStore.getState().loadFiles([file('twin.glb', bundleBytes())]);
    expect(useTwinStore.getState().status).toBe('ready');
    expect(useTwinStore.getState().pack?.skinToneHex).toBe('#c49a7a');
    expect(useTwinStore.getState().pack?.mapping).toEqual(Uint32Array.from([1, 3]));
    expect(db.get(TWIN_KEYS.json)).toBeNull();
    expect(db.get(TWIN_KEYS.mapping)).toBeNull();
    useTwinStore.setState({ ...pristine, hydrated: false }, true);
    await useTwinStore.getState().hydrate();
    expect(useTwinStore.getState().mode).toBe('twin');
    expect(useTwinStore.getState().pack?.skinToneHex).toBe('#c49a7a');
    expect(useTwinStore.getState().pack?.names.glb).toBe('twin.glb');
  });
  it('starts empty in the standard model and refuses to show a twin that is not loaded', () => {
    const s = useTwinStore.getState();
    expect(s.mode).toBe('standard');
    expect(s.status).toBe('empty');
    s.setMode('twin');
    expect(useTwinStore.getState().mode).toBe('standard');
    expect(isTwinActive(useTwinStore.getState())).toBe(false);
  });

  it('waits for both required files, then loads, switches to the twin and persists', async () => {
    await useTwinStore.getState().loadFiles([file('rigged.glb', glbBytes())]);
    expect(useTwinStore.getState().status).toBe('partial');
    expect(useTwinStore.getState().pack).toBeNull();
    await useTwinStore.getState().loadFiles([file('twin.json', twinJson)]);
    const s = useTwinStore.getState();
    expect(s.status).toBe('ready');
    expect(s.pack?.def.measurementsCm.height).toBe(170);
    expect(s.pack?.mapping).toBeNull(); // optional
    expect(s.mode).toBe('twin');
    expect(isTwinActive(s)).toBe(true);
    await flush();
    expect(db.get(TWIN_KEYS.mode)).toBe('twin');
    expect(db.get(TWIN_KEYS.glb)).toBeInstanceOf(ArrayBuffer);
    expect(db.get(TWIN_KEYS.json)).toBe(twinJson);
    for (const key of db.keys()) expect(key.startsWith('twin:')).toBe(true);
  });

  it('takes the three files in one pick, whatever the order', async () => {
    const mapping = new Uint8Array(8);
    await useTwinStore
      .getState()
      .loadFiles([
        file('mh2twin.bin', mapping),
        file('twin.json', twinJson),
        file('rigged.glb', glbBytes()),
      ]);
    const s = useTwinStore.getState();
    expect(s.status).toBe('ready');
    expect(s.pack?.mapping?.length).toBe(2);
    expect(s.pack?.names).toEqual({ glb: 'rigged.glb', json: 'twin.json', mapping: 'mh2twin.bin' });
  });

  it('reports a bad twin.json, a non-glb file and a mapping of the wrong size', async () => {
    await useTwinStore
      .getState()
      .loadFiles([file('rigged.glb', glbBytes()), file('twin.json', '{"version":3}')]);
    expect(useTwinStore.getState().status).toBe('error');
    expect(useTwinStore.getState().error?.code).toBe('version');
    expect(useTwinStore.getState().pack).toBeNull();

    useTwinStore.setState({ ...pristine, hydrated: false }, true);
    await useTwinStore
      .getState()
      .loadFiles([file('rigged.glb', new Uint8Array(40)), file('twin.json', twinJson)]);
    expect(useTwinStore.getState().error?.code).toBe('glb');

    useTwinStore.setState({ ...pristine, hydrated: false }, true);
    await useTwinStore
      .getState()
      .loadFiles([
        file('rigged.glb', glbBytes()),
        file('twin.json', twinJson),
        file('mh2twin.bin', new Uint8Array(12)),
      ]);
    expect(useTwinStore.getState().error?.code).toBe('mapping'); // 3 entries, twin.json says 2
    expect(useTwinStore.getState().pack).toBeNull();

    useTwinStore.setState({ ...pristine, hydrated: false }, true);
    await useTwinStore.getState().loadFiles([file('photo.png', 'x')]);
    expect(useTwinStore.getState().status).toBe('error');
  });

  it('switching back and forth keeps the files; remove clears everything including the stored copy', async () => {
    await useTwinStore
      .getState()
      .loadFiles([file('rigged.glb', glbBytes()), file('twin.json', twinJson)]);
    useTwinStore.getState().setMode('standard');
    expect(useTwinStore.getState().pack).not.toBeNull();
    useTwinStore.getState().setMode('twin');
    expect(useTwinStore.getState().mode).toBe('twin');
    await flush();
    await useTwinStore.getState().remove();
    const s = useTwinStore.getState();
    expect(s.pack).toBeNull();
    expect(s.mode).toBe('standard');
    expect(s.status).toBe('empty');
    expect(db.size).toBe(0);
  });

  it('a runtime failure falls back to the standard model and keeps the message', async () => {
    await useTwinStore
      .getState()
      .loadFiles([file('rigged.glb', glbBytes()), file('twin.json', twinJson)]);
    useTwinStore.getState().fail({ code: 'mismatch', detail: 'rest heads differ by 40 mm' });
    const s = useTwinStore.getState();
    expect(s.mode).toBe('standard');
    expect(s.status).toBe('error');
    expect(s.error?.code).toBe('mismatch');
    expect(s.pack).not.toBeNull();
  });

  it('hydrate restores the stored twin and the chosen model', async () => {
    await useTwinStore
      .getState()
      .loadFiles([
        file('rigged.glb', glbBytes()),
        file('twin.json', twinJson),
        file('mh2twin.bin', new Uint8Array(8)),
      ]);
    await flush();
    useTwinStore.setState({ ...pristine, hydrated: false }, true);
    expect(useTwinStore.getState().pack).toBeNull();
    await useTwinStore.getState().hydrate();
    const s = useTwinStore.getState();
    expect(s.status).toBe('ready');
    expect(s.mode).toBe('twin');
    expect(s.pack?.mapping?.length).toBe(2);
    expect(s.pack?.names.glb).toBe('rigged.glb');
  });

  it('hydrate ignores corrupt storage and survives unavailable storage (private mode)', async () => {
    db.set(TWIN_KEYS.glb, glbBytes());
    db.set(TWIN_KEYS.json, 'not json');
    await useTwinStore.getState().hydrate();
    expect(useTwinStore.getState().pack).toBeNull();

    useTwinStore.setState({ ...pristine, hydrated: false }, true);
    failing = true;
    await useTwinStore
      .getState()
      .loadFiles([file('rigged.glb', glbBytes()), file('twin.json', twinJson)]);
    expect(useTwinStore.getState().status).toBe('ready'); // the session still works without persistence
    await useTwinStore.getState().hydrate();
  });

  it('buildPack needs both required pieces', () => {
    expect(buildPack({})).toBeNull();
    expect(buildPack({ json: { name: 'twin.json', text: twinJson } })).toBeNull();
  });
});
