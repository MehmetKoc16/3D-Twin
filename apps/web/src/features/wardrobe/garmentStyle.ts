import type { GarmentTemplateDef } from '@dt/avatar-core';

/**
 * How a top template was authored. A `tucked` template ends at the waist, inside the trousers (its hem is a
 * construction edge that must never be visible next to a bottom); an `untucked` one hangs freely over them.
 * The catalogue (`index.json`) does not carry this, so it is kept here, keyed by template id. Unknown templates
 * are `untucked`; bottoms and shoes are never tucked.
 */
export type GarmentStyle = 'tucked' | 'untucked';

const STYLE_BY_TEMPLATE: Readonly<Record<string, GarmentStyle>> = {
  // toigo_basic_tucked_t-shirt
  tshirt: 'tucked',
  sweatshirt: 'untucked',
};

export function garmentStyle(def: Pick<GarmentTemplateDef, 'id' | 'category'>): GarmentStyle {
  return def.category === 'top' ? (STYLE_BY_TEMPLATE[def.id] ?? 'untucked') : 'untucked';
}

/** True for a top that is worn tucked into a bottom (only meaningful while a bottom is worn). */
export function isTuckedTop(def: Pick<GarmentTemplateDef, 'id' | 'category'>): boolean {
  return garmentStyle(def) === 'tucked';
}
