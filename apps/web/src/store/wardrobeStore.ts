import { del, get, getMany, keys, set } from 'idb-keyval';
import { create } from 'zustand';
import {
  garmentSlot,
  type GarmentCategory,
  type GarmentSlot,
  type GarmentTemplateDef,
  type StoreItemDef,
} from '@dt/avatar-core';

/** Storage keys: every wardrobe record lives below the `wardrobe:` prefix in IndexedDB (idb-keyval). */
export const ITEM_KEY_PREFIX = 'wardrobe:item:';
export const WORN_KEY = 'wardrobe:worn';

export type CatalogStatus = 'idle' | 'loading' | 'ready' | 'error';
export type WornMap = Partial<Record<GarmentSlot, string>>;

interface WardrobeState {
  templates: GarmentTemplateDef[];
  catalogStatus: CatalogStatus;
  items: StoreItemDef[];
  /** Item id worn per outfit slot (one item per slot). */
  worn: WornMap;
  heatmap: boolean;
  hydrated: boolean;
  loadCatalog: () => Promise<void>;
  hydrate: () => Promise<void>;
  saveItem: (item: StoreItemDef) => Promise<void>;
  deleteItem: (id: string) => Promise<void>;
  wear: (itemId: string) => void;
  takeOff: (category: GarmentSlot) => void;
  selectSize: (itemId: string, size: string) => void;
  setHeatmap: (value: boolean) => void;
}

const SLOTS: readonly GarmentSlot[] = ['top', 'bottom', 'shoes', 'outer'];

export function isTemplateDef(value: unknown): value is GarmentTemplateDef {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Partial<GarmentTemplateDef>;
  return (
    typeof v.id === 'string' &&
    typeof v.mesh === 'string' &&
    typeof v.layer === 'number' &&
    Number.isFinite(v.layer) &&
    (v.slot === undefined || SLOTS.includes(v.slot)) &&
    typeof v.baseColor === 'string' &&
    typeof v.label?.tr === 'string' &&
    typeof v.label?.en === 'string' &&
    (v.category === 'top' || v.category === 'bottom' || v.category === 'shoes') &&
    typeof v.nativeMeasures === 'object' &&
    v.nativeMeasures !== null &&
    typeof v.defaultEase === 'object' &&
    v.defaultEase !== null
  );
}

export function isStoreItem(value: unknown): value is StoreItemDef {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Partial<StoreItemDef>;
  return (
    typeof v.id === 'string' &&
    typeof v.name === 'string' &&
    typeof v.templateId === 'string' &&
    typeof v.color === 'string' &&
    typeof v.selectedSize === 'string' &&
    Array.isArray(v.sizes) &&
    v.sizes.every((s) => typeof s === 'string') &&
    typeof v.chart === 'object' &&
    v.chart !== null
  );
}

async function persist(action: () => Promise<unknown>): Promise<void> {
  try {
    await action();
  } catch {
    /* storage unavailable (private mode / quota): the session still works */
  }
}

/** The fitting category stays independent of the outfit slot. */
export function categoryOfItem(
  state: Pick<WardrobeState, 'templates' | 'items'>,
  itemId: string,
): GarmentCategory | undefined {
  const item = state.items.find((i) => i.id === itemId);
  return state.templates.find((t) => t.id === item?.templateId)?.category;
}

/** The outfit slot an item belongs to, from its template (including legacy catalogue defaults). */
export function slotOfItem(
  state: Pick<WardrobeState, 'templates' | 'items'>,
  itemId: string,
): GarmentSlot | undefined {
  const item = state.items.find((i) => i.id === itemId);
  const template = state.templates.find((t) => t.id === item?.templateId);
  return template ? garmentSlot(template) : undefined;
}

/** Re-key legacy category maps by the current catalogue. Explicit outer entries win over legacy top entries. */
export function migrateWorn(
  state: Pick<WardrobeState, 'templates' | 'items'>,
  saved: unknown,
): WornMap {
  const worn: WornMap = {};
  if (typeof saved !== 'object' || saved === null) return worn;
  for (const slot of SLOTS) {
    const id = (saved as Record<string, unknown>)[slot];
    if (typeof id !== 'string') continue;
    const target = slotOfItem(state, id);
    if (target && (target === slot || (slot === 'top' && target === 'outer'))) worn[target] = id;
  }
  return worn;
}

let hydrating: Promise<void> | null = null;

export const useWardrobeStore = create<WardrobeState>((update, read) => ({
  templates: [],
  catalogStatus: 'idle',
  items: [],
  worn: {},
  heatmap: false,
  hydrated: false,

  loadCatalog: async () => {
    if (read().catalogStatus === 'loading' || read().catalogStatus === 'ready') return;
    update({ catalogStatus: 'loading' });
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}assets/garments/index.json`);
      if (!response.ok) throw new Error(`garment catalogue: ${response.status}`);
      const data = (await response.json()) as { garments?: unknown };
      const garments = Array.isArray(data.garments) ? data.garments.filter(isTemplateDef) : [];
      update({ templates: garments, catalogStatus: 'ready' });
    } catch {
      update({ catalogStatus: 'error' });
    }
  },

  hydrate: () => {
    if (read().hydrated) return Promise.resolve();
    hydrating ??= (async () => {
      await read().loadCatalog();
      let stored: StoreItemDef[];
      let savedWorn: unknown;
      try {
        const all = await keys();
        const itemKeys = all.filter(
          (k): k is string => typeof k === 'string' && k.startsWith(ITEM_KEY_PREFIX),
        );
        const values = await getMany<unknown>(itemKeys);
        stored = values.filter(isStoreItem);
        savedWorn = await get<unknown>(WORN_KEY);
      } catch {
        stored = [];
      }
      const current = read();
      const merged = [
        ...stored.filter((s) => !current.items.some((i) => i.id === s.id)),
        ...current.items,
      ];
      const context = { templates: current.templates, items: merged };
      const valid = { ...migrateWorn(context, savedWorn), ...migrateWorn(context, current.worn) };
      const catalogReady = current.catalogStatus === 'ready';
      update({ items: merged, worn: valid, hydrated: catalogReady });
      // A failed catalogue fetch cannot establish slots; keep the saved outfit available for a later retry.
      if (catalogReady) await persist(() => set(WORN_KEY, valid));
    })().finally(() => {
      hydrating = null;
    });
    return hydrating;
  },

  saveItem: async (item) => {
    update((state) => {
      const exists = state.items.some((i) => i.id === item.id);
      const items = exists
        ? state.items.map((i) => (i.id === item.id ? item : i))
        : [...state.items, item];
      const worn = { ...state.worn };
      const previousSlot = SLOTS.find((slot) => worn[slot] === item.id);
      if (previousSlot) {
        delete worn[previousSlot];
        const nextSlot = slotOfItem({ templates: state.templates, items }, item.id);
        if (nextSlot) worn[nextSlot] = item.id;
      }
      return { items, worn };
    });
    await persist(() => set(`${ITEM_KEY_PREFIX}${item.id}`, item));
    await persist(() => set(WORN_KEY, read().worn));
  },

  deleteItem: async (id) => {
    const category = slotOfItem(read(), id);
    update((state) => ({
      items: state.items.filter((i) => i.id !== id),
      worn:
        category && state.worn[category] === id
          ? { ...state.worn, [category]: undefined }
          : state.worn,
    }));
    const worn = read().worn;
    await persist(async () => {
      await del(`${ITEM_KEY_PREFIX}${id}`);
      await set(WORN_KEY, worn);
    });
  },

  wear: (itemId) => {
    const category = slotOfItem(read(), itemId);
    if (!category) return;
    update((state) => ({ worn: { ...state.worn, [category]: itemId } }));
    const worn = read().worn;
    void persist(() => set(WORN_KEY, worn));
  },

  takeOff: (category) => {
    update((state) => ({ worn: { ...state.worn, [category]: undefined } }));
    const worn = read().worn;
    void persist(() => set(WORN_KEY, worn));
  },

  selectSize: (itemId, size) => {
    const item = read().items.find((i) => i.id === itemId);
    if (!item || !item.sizes.includes(size) || item.selectedSize === size) return;
    const next = { ...item, selectedSize: size };
    update((state) => ({ items: state.items.map((i) => (i.id === itemId ? next : i)) }));
    void persist(() => set(`${ITEM_KEY_PREFIX}${itemId}`, next));
  },

  setHeatmap: (heatmap) => update({ heatmap }),
}));
