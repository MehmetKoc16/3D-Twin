import { footLengthCmFromShoe } from '@dt/avatar-core';
import type {
  GarmentKind,
  GarmentMeasureId,
  GarmentTemplateDef,
  StoreItemDef,
} from '@dt/avatar-core';

/** Pure model of the size-chart editor (no React): drafts hold the raw text of every cell. */

/** Measures that are full girths (a flat / laid-out measurement is doubled). */
export const GIRTH_MEASURES: readonly GarmentMeasureId[] = ['chest', 'waist', 'hip', 'thigh'];

/** Shoe inner length = foot length of the EU size + this allowance (the middle of the "regular" fit band). */
export const SHOE_ALLOWANCE_CM = 1.4;
export const MAX_SIZES = 10;
export const MAX_NAME_LENGTH = 60;

const TOP_KINDS: readonly GarmentKind[] = ['tshirt', 'longsleeve', 'sweatshirt'];
const BOTTOM_KINDS: readonly GarmentKind[] = ['pants', 'jeans', 'shorts'];

/** Plausible range of a full (not flat) value, in cm. */
export const MEASURE_RANGES: Record<GarmentMeasureId, readonly [min: number, max: number]> = {
  chest: [60, 200],
  waist: [50, 200],
  hip: [60, 210],
  length: [20, 160],
  sleeve: [5, 90],
  shoulder: [25, 70],
  inseam: [10, 100],
  thigh: [30, 120],
  footLength: [15, 40],
};

export type ShoeMode = 'eu' | 'cm';

export function isTopKind(kind: GarmentKind): boolean {
  return TOP_KINDS.includes(kind);
}

export function isBottomKind(kind: GarmentKind): boolean {
  return BOTTOM_KINDS.includes(kind);
}

/** Chart rows offered for a garment kind, in display order. */
export function measuresForKind(kind: GarmentKind): GarmentMeasureId[] {
  if (isTopKind(kind)) return ['chest', 'waist', 'length', 'sleeve', 'shoulder'];
  if (isBottomKind(kind)) return ['waist', 'hip', 'inseam', 'thigh'];
  return ['footLength'];
}

/** Rows that must be filled in. */
export function requiredMeasures(kind: GarmentKind): GarmentMeasureId[] {
  if (isTopKind(kind)) return ['chest'];
  if (isBottomKind(kind)) return ['waist'];
  return ['footLength'];
}

export function isGirth(id: GarmentMeasureId): boolean {
  return GIRTH_MEASURES.includes(id);
}

export interface ChartDraft {
  name: string;
  storeUrl: string;
  templateId: string;
  color: string;
  sizes: string[];
  /** Raw cell text, one entry per size (missing / short arrays count as empty cells). */
  chart: Partial<Record<GarmentMeasureId, string[]>>;
  /** Measurements were taken flat (laid out): girth values are doubled on save. */
  flat: boolean;
  selectedSize: string;
  /** Shoes only: `eu` derives the inner length from EU size labels, `cm` reads the row as typed. */
  shoeMode: ShoeMode;
}

export interface ValidationIssue {
  /** Key below `wardrobe.errors.` in the locale files. */
  code:
    | 'nameRequired'
    | 'nameTooLong'
    | 'urlInvalid'
    | 'templateUnknown'
    | 'colorInvalid'
    | 'noSizes'
    | 'tooManySizes'
    | 'emptySize'
    | 'duplicateSize'
    | 'euSizeInvalid'
    | 'missingRequired'
    | 'incompleteRow'
    | 'invalidNumber'
    | 'outOfRange'
    | 'notIncreasing'
    | 'selectedSizeMissing';
  measure?: GarmentMeasureId;
  size?: string;
  params?: Record<string, string | number>;
}

export type DraftResult =
  { ok: true; item: StoreItemDef } | { ok: false; issues: ValidationIssue[] };

/** Accepts "96", "96.5" and "96,5"; anything else (including empty text) is undefined. */
export function parseCm(text: string | undefined): number | undefined {
  if (text === undefined) return undefined;
  const cleaned = text.trim().replace(',', '.');
  if (!/^\d+(\.\d+)?$/.test(cleaned)) return undefined;
  const value = Number(cleaned);
  return Number.isFinite(value) ? value : undefined;
}

export function roundCm(value: number): number {
  return Math.round(value * 10) / 10;
}

/** Inner length (cm) of a shoe given its EU size label, or undefined when the label is not a plausible EU size. */
export function shoeInnerLengthCm(euLabel: string): number | undefined {
  const size = parseCm(euLabel);
  if (size === undefined || size < 30 || size > 52) return undefined;
  return roundCm(footLengthCmFromShoe({ system: 'EU', size }) + SHOE_ALLOWANCE_CM);
}

export function defaultSizeLabels(kind: GarmentKind): string[] {
  return measuresForKind(kind).includes('footLength')
    ? ['39', '40', '41', '42', '43', '44']
    : ['S', 'M', 'L'];
}

export function isHexColor(value: string): boolean {
  return /^#[0-9a-fA-F]{6}$/.test(value);
}

export function createDraft(template: GarmentTemplateDef): ChartDraft {
  const sizes = defaultSizeLabels(template.kind);
  return {
    name: '',
    storeUrl: '',
    templateId: template.id,
    color: template.baseColor,
    sizes,
    chart: {},
    flat: false,
    selectedSize: sizes[Math.min(1, sizes.length - 1)] ?? '',
    shoeMode: template.category === 'shoes' ? 'eu' : 'cm',
  };
}

function formatCell(value: number | undefined): string {
  return value === undefined ? '' : String(value);
}

/** Restores an editable draft from a saved item (values are already full girths, so flat is off). */
export function itemToDraft(item: StoreItemDef, template: GarmentTemplateDef): ChartDraft {
  const chart: ChartDraft['chart'] = {};
  for (const id of measuresForKind(template.kind)) {
    const row = item.chart[id];
    if (row) chart[id] = item.sizes.map((_, i) => formatCell(row[i]));
  }
  let shoeMode: ShoeMode = 'cm';
  if (template.category === 'shoes') {
    const row = item.chart.footLength ?? [];
    const derived = item.sizes.map((label) => shoeInnerLengthCm(label));
    if (
      item.sizes.length > 0 &&
      derived.every((v, i) => v !== undefined && Math.abs(v - (row[i] ?? Infinity)) < 0.06)
    ) {
      shoeMode = 'eu';
    }
  }
  return {
    name: item.name,
    storeUrl: item.storeUrl ?? '',
    templateId: item.templateId,
    color: item.color,
    sizes: [...item.sizes],
    chart,
    flat: false,
    selectedSize: item.selectedSize,
    shoeMode,
  };
}

/** The footLength row of a shoe draft in EU mode (empty text for labels that are not EU sizes). */
export function euRow(sizes: readonly string[]): string[] {
  return sizes.map((label) => formatCell(shoeInnerLengthCm(label)));
}

export function setCell(
  draft: ChartDraft,
  id: GarmentMeasureId,
  sizeIndex: number,
  text: string,
): ChartDraft {
  const row = [...(draft.chart[id] ?? [])];
  while (row.length < draft.sizes.length) row.push('');
  row[sizeIndex] = text;
  return { ...draft, chart: { ...draft.chart, [id]: row } };
}

export function addSize(draft: ChartDraft, label: string): ChartDraft {
  if (draft.sizes.length >= MAX_SIZES) return draft;
  const chart: ChartDraft['chart'] = {};
  for (const [id, row] of Object.entries(draft.chart) as [GarmentMeasureId, string[]][]) {
    chart[id] = [...row, ''];
  }
  const sizes = [...draft.sizes, label];
  return { ...draft, sizes, chart, selectedSize: draft.selectedSize || label };
}

export function removeSize(draft: ChartDraft, index: number): ChartDraft {
  if (index < 0 || index >= draft.sizes.length) return draft;
  const sizes = draft.sizes.filter((_, i) => i !== index);
  const chart: ChartDraft['chart'] = {};
  for (const [id, row] of Object.entries(draft.chart) as [GarmentMeasureId, string[]][]) {
    chart[id] = row.filter((_, i) => i !== index);
  }
  const removed = draft.sizes[index];
  const selectedSize = draft.selectedSize === removed ? (sizes[0] ?? '') : draft.selectedSize;
  return { ...draft, sizes, chart, selectedSize };
}

/** Renames a size label; the selected size follows its label. */
export function renameSize(draft: ChartDraft, index: number, label: string): ChartDraft {
  const previous = draft.sizes[index];
  if (previous === undefined) return draft;
  const sizes = draft.sizes.map((s, i) => (i === index ? label : s));
  return {
    ...draft,
    sizes,
    selectedSize: draft.selectedSize === previous ? label : draft.selectedSize,
  };
}

function isValidUrl(text: string): boolean {
  try {
    const url = new URL(text);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch {
    return false;
  }
}

export function newItemId(): string {
  const random = globalThis.crypto?.randomUUID?.();
  return `item-${random ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`}`;
}

/** Validates a draft and turns it into a StoreItemDef (girths doubled when measured flat). */
export function draftToItem(
  draft: ChartDraft,
  template: GarmentTemplateDef | undefined,
  id: string = newItemId(),
): DraftResult {
  const issues: ValidationIssue[] = [];
  const name = draft.name.trim();
  if (name.length === 0) issues.push({ code: 'nameRequired' });
  else if (name.length > MAX_NAME_LENGTH)
    issues.push({ code: 'nameTooLong', params: { max: MAX_NAME_LENGTH } });
  const url = draft.storeUrl.trim();
  if (url.length > 0 && !isValidUrl(url)) issues.push({ code: 'urlInvalid' });
  if (!template) {
    issues.push({ code: 'templateUnknown' });
    return { ok: false, issues };
  }
  if (!isHexColor(draft.color)) issues.push({ code: 'colorInvalid' });

  const labels = draft.sizes.map((s) => s.trim());
  if (labels.length === 0) issues.push({ code: 'noSizes' });
  if (labels.length > MAX_SIZES) issues.push({ code: 'tooManySizes', params: { max: MAX_SIZES } });
  const seen = new Set<string>();
  labels.forEach((label) => {
    if (label.length === 0) issues.push({ code: 'emptySize' });
    else if (seen.has(label.toLowerCase())) issues.push({ code: 'duplicateSize', size: label });
    seen.add(label.toLowerCase());
  });
  if (labels.length > 0 && !labels.includes(draft.selectedSize.trim()))
    issues.push({ code: 'selectedSizeMissing' });

  const chart: StoreItemDef['chart'] = {};
  const shoeEu = template.category === 'shoes' && draft.shoeMode === 'eu';
  for (const measure of measuresForKind(template.kind)) {
    const required = requiredMeasures(template.kind).includes(measure);
    let cells: (string | undefined)[];
    if (measure === 'footLength' && shoeEu) {
      cells = labels.map((label) => {
        const value = shoeInnerLengthCm(label);
        if (value === undefined && label.length > 0)
          issues.push({ code: 'euSizeInvalid', size: label });
        return value === undefined ? undefined : String(value);
      });
    } else {
      cells = labels.map((_, i) => draft.chart[measure]?.[i]);
    }
    const filled = cells.filter((c) => c !== undefined && c.trim().length > 0).length;
    if (filled === 0) {
      if (required && labels.length > 0) issues.push({ code: 'missingRequired', measure });
      continue;
    }
    if (filled < labels.length) {
      issues.push({ code: 'incompleteRow', measure });
      continue;
    }
    const factor = draft.flat && isGirth(measure) ? 2 : 1;
    const [min, max] = MEASURE_RANGES[measure];
    const values: number[] = [];
    let rowOk = true;
    cells.forEach((cell, i) => {
      const parsed = parseCm(cell);
      if (parsed === undefined) {
        issues.push({ code: 'invalidNumber', measure, size: labels[i] ?? '' });
        rowOk = false;
        return;
      }
      const value = roundCm(parsed * factor);
      if (value < min || value > max) {
        issues.push({ code: 'outOfRange', measure, size: labels[i] ?? '', params: { min, max } });
        rowOk = false;
      }
      values.push(value);
    });
    if (!rowOk) continue;
    if (values.some((v, i) => i > 0 && v < (values[i - 1] ?? v) - 0.05)) {
      issues.push({ code: 'notIncreasing', measure });
      continue;
    }
    chart[measure] = values;
  }
  if (issues.length > 0) return { ok: false, issues };
  const item: StoreItemDef = {
    id,
    name,
    templateId: template.id,
    color: draft.color.toLowerCase(),
    sizes: labels,
    chart,
    selectedSize: draft.selectedSize.trim(),
  };
  if (url.length > 0) item.storeUrl = url;
  return { ok: true, item };
}
