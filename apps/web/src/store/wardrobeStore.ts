import { del, get, getMany, keys, set } from 'idb-keyval';
import { create } from 'zustand';
import type { GarmentCategory, GarmentTemplateDef, StoreItemDef } from '@dt/avatar-core';

/** Storage keys: every wardrobe record lives below the `wardrobe:` prefix in IndexedDB (idb-keyval). */
export const ITEM_KEY_PREFIX = 'wardrobe:item:';
export const WORN_KEY = 'wardrobe:worn';

export type CatalogStatus = 'idle' | 'loading' | 'ready' | 'error';
export type WornMap = Partial<Record<GarmentCategory, string>>;

interface WardrobeState {
  templates: GarmentTemplateDef[];
  catalogStatus: CatalogStatus;
  items: StoreItemDef[];
  /** Item id worn per category (one item per category). */
  worn: WornMap;
  heatmap: boolean;
  hydrated: boolean;
  loadCatalog: () => Promise<void>;
  hydrate: () => Promise<void>;
  saveItem: (item: StoreItemDef) => Promise<void>;
  deleteItem: (id: string) => Promise<void>;
  wear: (itemId: string) => void;
  takeOff: (category: GarmentCategory) => void;
  selectSize: (itemId: string, size: string) => void;
  setHeatmap: (value: boolean) => void;
}

const CATEGORIES: readonly GarmentCategory[] = ['top', 'bottom', 'shoes'];

export function isTemplateDef(value: unknown): value is GarmentTemplateDef {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Partial<GarmentTemplateDef>;
  return (
    typeof v.id === 'string' &&
    typeof v.mesh === 'string' &&
    typeof v.layer === 'number' &&
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

/** The category an item belongs to, from its template. */
export function categoryOfItem(
  state: Pick<WardrobeState, 'templates' | 'items'>,
  itemId: string,
): GarmentCategory | undefined {
  const item = state.items.find((i) => i.id === itemId);
  return state.templates.find((t) => t.id === item?.templateId)?.category;
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
    hydrating ??= (async () => {
      await read().loadCatalog();
      let stored: StoreItemDef[];
      const worn: WornMap = {};
      try {
        const all = await keys();
        const itemKeys = all.filter((k): k is string => typeof k === 'string' && k.startsWith(ITEM_KEY_PREFIX));
        const values = await getMany<unknown>(itemKeys);
        stored = values.filter(isStoreItem);
        const savedWorn = await get<unknown>(WORN_KEY);
        if (typeof savedWorn === 'object' && savedWorn !== null) {
          for (const category of CATEGORIES) {
            const id = (savedWorn as Record<string, unknown>)[category];
            if (typeof id === 'string') worn[category] = id;
          }
        }
      } catch {
        stored = [];
      }
      const current = read();
      const merged = [...stored.filter((s) => !current.items.some((i) => i.id === s.id)), ...current.items];
      const valid: WornMap = {};
      for (const category of CATEGORIES) {
        const id = current.worn[category] ?? worn[category];
        if (id && categoryOfItem({ templates: current.templates, items: merged }, id) === category) valid[category] = id;
      }
      update({ items: merged, worn: valid, hydrated: true });
    })();
    return hydrating;
  },

  saveItem: async (item) => {
    update((state) => {
      const exists = state.items.some((i) => i.id === item.id);
      return { items: exists ? state.items.map((i) => (i.id === item.id ? item : i)) : [...state.items, item] };
    });
    await persist(() => set(`${ITEM_KEY_PREFIX}${item.id}`, item));
  },

  deleteItem: async (id) => {
    const category = categoryOfItem(read(), id);
    update((state) => ({
      items: state.items.filter((i) => i.id !== id),
      worn: category && state.worn[category] === id ? { ...state.worn, [category]: undefined } : state.worn,
    }));
    const worn = read().worn;
    await persist(async () => {
      await del(`${ITEM_KEY_PREFIX}${id}`);
      await set(WORN_KEY, worn);
    });
  },

  wear: (itemId) => {
    const category = categoryOfItem(read(), itemId);
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
