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

import { ITEM_KEY_PREFIX, useWardrobeStore, WORN_KEY } from './wardrobeStore';

function template(id: string, category: GarmentTemplateDef['category'], kind: GarmentTemplateDef['kind']): GarmentTemplateDef {
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

const templates = [template('tshirt', 'top', 'tshirt'), template('sweatshirt', 'top', 'sweatshirt'), template('pants', 'bottom', 'pants')];

function item(id: string, templateId: string): StoreItemDef {
  return { id, name: id, templateId, color: '#123456', sizes: ['S', 'M'], chart: { chest: [90, 100] }, selectedSize: 'S' };
}

beforeEach(() => {
  db.clear();
  useWardrobeStore.setState({ templates, catalogStatus: 'ready', items: [], worn: {}, heatmap: false, hydrated: false });
});

describe('wardrobeStore', () => {
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
