import type { BodyPartCategory, BodyPartDef } from '@dt/avatar-core';

/** `public/assets/parts/index.json`. */
export interface PartsIndex {
  version: 1;
  parts: BodyPartDef[];
  defaults: Partial<Record<BodyPartCategory, string>>;
}

export const PART_CATEGORIES: readonly BodyPartCategory[] = ['eyes', 'eyebrows', 'eyelashes', 'hair'];
const ALPHA_MODES = ['OPAQUE', 'MASK', 'BLEND'];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function fail(message: string): never {
  throw new Error(`parts index: ${message}`);
}

function isScaleRef(value: unknown): boolean {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    Number.isInteger(value[0]) &&
    Number.isInteger(value[1]) &&
    (value[0] as number) >= 0 &&
    (value[1] as number) >= 0 &&
    typeof value[2] === 'number' &&
    Number.isFinite(value[2]) &&
    value[2] > 0
  );
}

function validatePart(value: unknown, position: number): BodyPartDef {
  if (!isRecord(value)) return fail(`parts[${position}] is not an object`);
  const id = value.id;
  if (typeof id !== 'string' || id === '') return fail(`parts[${position}] has no id`);
  if (!PART_CATEGORIES.includes(value.category as BodyPartCategory)) return fail(`${id}: unknown category`);
  const label = value.label;
  if (!isRecord(label) || typeof label.tr !== 'string' || typeof label.en !== 'string')
    return fail(`${id}: label needs tr and en`);
  if (typeof value.mesh !== 'string' || typeof value.binding !== 'string') return fail(`${id}: mesh / binding missing`);
  if (value.deleteVerts !== undefined && typeof value.deleteVerts !== 'string') return fail(`${id}: bad deleteVerts`);
  const material = value.material;
  if (
    !isRecord(material) ||
    !ALPHA_MODES.includes(material.alphaMode as string) ||
    typeof material.doubleSided !== 'boolean' ||
    typeof material.tintable !== 'boolean'
  )
    return fail(`${id}: bad material`);
  if (material.alphaMode === 'MASK') {
    const cutoff = material.alphaCutoff;
    if (typeof cutoff !== 'number' || !(cutoff > 0 && cutoff < 1)) return fail(`${id}: MASK needs an alphaCutoff in (0, 1)`);
  }
  const refs = value.scaleRefs;
  if (refs !== undefined) {
    if (!isRecord(refs) || !isScaleRef(refs.x) || !isScaleRef(refs.y) || !isScaleRef(refs.z))
      return fail(`${id}: bad scaleRefs`);
  }
  if (value.category === 'eyes') {
    const iris = value.irisUv;
    if (
      !isRecord(iris) ||
      !Array.isArray(iris.center) ||
      iris.center.length !== 2 ||
      !iris.center.every((c) => typeof c === 'number' && c >= 0 && c <= 1) ||
      typeof iris.radius !== 'number' ||
      !(iris.radius > 0 && iris.radius <= 0.5)
    )
      return fail(`${id}: eyes need a valid irisUv`);
  }
  return value as unknown as BodyPartDef;
}

/** Validates a parsed `index.json`; throws with a message naming the offending part. */
export function validatePartsIndex(raw: unknown): PartsIndex {
  if (!isRecord(raw)) return fail('not an object');
  if (raw.version !== 1) return fail(`unsupported version ${String(raw.version)}`);
  if (!Array.isArray(raw.parts)) return fail('parts is not an array');
  const parts = raw.parts.map(validatePart);
  const ids = new Set<string>();
  for (const part of parts) {
    if (ids.has(part.id)) return fail(`duplicate id ${part.id}`);
    ids.add(part.id);
  }
  const defaults: Partial<Record<BodyPartCategory, string>> = {};
  if (raw.defaults !== undefined) {
    if (!isRecord(raw.defaults)) return fail('defaults is not an object');
    for (const [category, id] of Object.entries(raw.defaults)) {
      const part = parts.find((p) => p.id === id);
      if (!PART_CATEGORIES.includes(category as BodyPartCategory) || !part || part.category !== category)
        return fail(`default ${category} -> ${String(id)} is not a part of that category`);
      defaults[category as BodyPartCategory] = part.id;
    }
  }
  return { version: 1, parts, defaults };
}

export function partsOf(index: PartsIndex, category: BodyPartCategory): BodyPartDef[] {
  return index.parts.filter((p) => p.category === category);
}

export function findPart(index: PartsIndex, id: string | null | undefined): BodyPartDef | undefined {
  return id ? index.parts.find((p) => p.id === id) : undefined;
}

export function partsBaseUrl(): string {
  return `${import.meta.env.BASE_URL}assets/parts/`;
}

let cached: Promise<PartsIndex> | null = null;

/** Fetches and validates the parts catalogue once. */
export function loadPartsIndex(baseUrl = partsBaseUrl()): Promise<PartsIndex> {
  cached ??= (async () => {
    const response = await fetch(`${baseUrl}index.json`);
    if (!response.ok) throw new Error(`parts index: HTTP ${response.status}`);
    return validatePartsIndex(await response.json());
  })().catch((error: unknown) => {
    cached = null;
    throw error;
  });
  return cached;
}
