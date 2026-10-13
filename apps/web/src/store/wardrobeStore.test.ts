import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { GarmentTemplateDef, StoreItemDef } from '@dt/avatar-core';

const db = new Map<string, unknown>();
vi.mock('idb-keyval', () => ({
  get: vi.fn(async (key: string) => db.get(key)),
  set: vi.fn(async (key: string, value: unknown) => void db.set(key, value)),
  del: vi.fn(async (key: string) => void db.delete(key)),
  keys: vi.fn(async () => [...db.keys()]),
  getMany: vi.fn(async (keys: string[]) => keys.map((k) => db.get(k))),
}));

import {
  ITEM_KEY_PREFIX,
  isTemplateDef,
  migrateWorn,
  useWardrobeStore,
  WORN_KEY,
} from './wardrobeStore';

function template(
  id: string,
  category: GarmentTemplateDef['category'],
  kind: GarmentTemplateDef['kind'],
): GarmentTemplateDef {
  return {
    id,
    kind,
    category,
    label: { tr: id, en: id },
    license: 'CC0-1.0',
    mesh: `${id}.glb`,
    nativeMeasures: {},
    defaultEase: {},
    layer: 1,
    baseColor: '#808080',
  };
}

const templates = [
  template('tshirt', 'top', 'tshirt'),
  template('sweatshirt', 'top', 'sweatshirt'),
  template('pants', 'bottom', 'pants'),
  { ...template('jacket', 'top', 'longsleeve'), layer: 3 },
  { ...template('coat', 'top', 'longsleeve'), layer: 2, slot: 'outer' as const },
];

function item(id: string, templateId: string): StoreItemDef {
  return {
    id,
    name: id,
    templateId,
    color: '#123456',
    sizes: ['S', 'M'],
    chart: { chest: [90, 100] },
    selectedSize: 'S',
  };
}

beforeEach(() => {
  db.clear();
  useWardrobeStore.setState({
    templates,
    catalogStatus: 'ready',
    items: [],
    worn: {},
    heatmap: false,
    hydrated: false,
  });
});

describe('wardrobeStore', () => {
  it('keeps the top while wearing, replacing, removing and deleting outerwear', async () => {
    const s = useWardrobeStore.getState();
    for (const [id, templateId] of [
      ['tee', 'tshirt'],
      ['j', 'jacket'],
      ['c', 'coat'],
    ])
      await s.saveItem(item(id!, templateId!));
    s.wear('tee');
    s.wear('j');
    expect(useWardrobeStore.getState().worn).toEqual({ top: 'tee', outer: 'j' });
    s.wear('c');
    expect(useWardrobeStore.getState().worn).toEqual({ top: 'tee', outer: 'c' });
    s.takeOff('outer');
    expect(useWardrobeStore.getState().worn.top).toBe('tee');
    s.wear('j');
    await s.deleteItem('j');
    expect(useWardrobeStore.getState().worn.top).toBe('tee');
    expect(useWardrobeStore.getState().worn.outer).toBeUndefined();
    s.takeOff('top');
    s.wear('c');
    expect(useWardrobeStore.getState().worn.outer).toBe('c');
    expect(useWardrobeStore.getState().worn.top).toBeUndefined();
  });

  it('migrates legacy jackets from top to outer and persists the new slot map', async () => {
    db.set(`${ITEM_KEY_PREFIX}j`, item('j', 'jacket'));
    db.set(`${ITEM_KEY_PREFIX}p`, item('p', 'pants'));
    db.set(WORN_KEY, { top: 'j', bottom: 'p' });
    await useWardrobeStore.getState().hydrate();
    expect(useWardrobeStore.getState().worn).toEqual({ outer: 'j', bottom: 'p' });
    expect(db.get(WORN_KEY)).toEqual({ outer: 'j', bottom: 'p' });
  });

  it('preserves the saved outfit when catalogue loading fails, then migrates on retry', async () => {
    db.set(`${ITEM_KEY_PREFIX}j`, item('j', 'jacket'));
    db.set(WORN_KEY, { top: 'j' });
    useWardrobeStore.setState({ catalogStatus: 'idle', templates: [] });
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
    try {
      await useWardrobeStore.getState().hydrate();
      expect(db.get(WORN_KEY)).toEqual({ top: 'j' });
      expect(useWardrobeStore.getState().hydrated).toBe(false);
      useWardrobeStore.setState({ templates, catalogStatus: 'ready' });
      await useWardrobeStore.getState().hydrate();
      expect(useWardrobeStore.getState().worn).toEqual({ outer: 'j' });
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('preserves saved top + outer and gives an explicit outer priority over a legacy top jacket', () => {
    const state = {
      templates,
      items: [item('tee', 'tshirt'), item('j', 'jacket'), item('c', 'coat')],
    };
    expect(migrateWorn(state, { top: 'tee', outer: 'j' })).toEqual({ top: 'tee', outer: 'j' });
    expect(migrateWorn(state, { top: 'j', outer: 'c', shoes: 'missing' })).toEqual({ outer: 'c' });
    expect(migrateWorn(state, null)).toEqual({});
    expect(migrateWorn(state, { bottom: 'tee' })).toEqual({});
  });

  it('supports a slot override and rejects malformed catalogue slots', () => {
    expect(isTemplateDef({ ...templates[0], slot: 'outer' })).toBe(true);
    expect(isTemplateDef({ ...templates[0], slot: 'invalid' })).toBe(false);
    expect(isTemplateDef({ ...templates[0], layer: Infinity })).toBe(false);
  });

  it('moves an edited worn item into its new slot and persists it', async () => {
    const s = useWardrobeStore.getState();
    await s.saveItem(item('a', 'tshirt'));
    s.wear('a');
    await s.saveItem(item('a', 'jacket'));
    expect(useWardrobeStore.getState().worn).toEqual({ outer: 'a' });
    expect(db.get(WORN_KEY)).toEqual({ outer: 'a' });
  });
  it('wears one item per category: a second top replaces the first', async () => {
    const s = useWardrobeStore.getState();
    await s.saveItem(item('a', 'tshirt'));
    await s.saveItem(item('b', 'sweatshirt'));
    await s.saveItem(item('c', 'pants'));
    s.wear('a');
    s.wear('c');
    expect(useWardrobeStore.getState().worn).toEqual({ top: 'a', bottom: 'c' });
    useWardrobeStore.getState().wear('b');
    expect(useWardrobeStore.getState().worn.top).toBe('b');
    expect(useWardrobeStore.getState().worn.bottom).toBe('c');
    useWardrobeStore.getState().takeOff('top');
    expect(useWardrobeStore.getState().worn.top).toBeUndefined();
  });

  it('persists items under the wardrobe: prefix and removes them on delete', async () => {
    await useWardrobeStore.getState().saveItem(item('a', 'tshirt'));
    expect(db.has(`${ITEM_KEY_PREFIX}a`)).toBe(true);
    expect([...db.keys()].every((k) => k.startsWith('wardrobe:'))).toBe(true);
    useWardrobeStore.getState().wear('a');
    await useWardrobeStore.getState().deleteItem('a');
    expect(db.has(`${ITEM_KEY_PREFIX}a`)).toBe(false);
    expect(useWardrobeStore.getState().items).toEqual([]);
    expect(useWardrobeStore.getState().worn.top).toBeUndefined(); // deleting a worn item takes it off
  });

  it('remembers the selected size and ignores unknown sizes', async () => {
    await useWardrobeStore.getState().saveItem(item('a', 'tshirt'));
    useWardrobeStore.getState().selectSize('a', 'M');
    expect(useWardrobeStore.getState().items[0]!.selectedSize).toBe('M');
    expect((db.get(`${ITEM_KEY_PREFIX}a`) as StoreItemDef).selectedSize).toBe('M');
    useWardrobeStore.getState().selectSize('a', 'XXL');
    expect(useWardrobeStore.getState().items[0]!.selectedSize).toBe('M');
  });

  it('updating an item keeps its place; editing does not duplicate it', async () => {
    await useWardrobeStore.getState().saveItem(item('a', 'tshirt'));
    await useWardrobeStore.getState().saveItem({ ...item('a', 'tshirt'), name: 'renamed' });
    expect(useWardrobeStore.getState().items).toHaveLength(1);
    expect(useWardrobeStore.getState().items[0]!.name).toBe('renamed');
  });

  it('does not wear an item whose template is unknown', async () => {
    await useWardrobeStore.getState().saveItem(item('x', 'missing-template'));
    useWardrobeStore.getState().wear('x');
    expect(useWardrobeStore.getState().worn).toEqual({});
  });

  it('hydrates saved items and the worn map, dropping stale entries', async () => {
    db.set(`${ITEM_KEY_PREFIX}a`, item('a', 'tshirt'));
    db.set(`${ITEM_KEY_PREFIX}bad`, { nonsense: true });
    db.set(WORN_KEY, { top: 'a', bottom: 'ghost' });
    await useWardrobeStore.getState().hydrate();
    const state = useWardrobeStore.getState();
    expect(state.items.map((i) => i.id)).toEqual(['a']);
    expect(state.worn).toEqual({ top: 'a' });
    expect(state.hydrated).toBe(true);
  });
});
