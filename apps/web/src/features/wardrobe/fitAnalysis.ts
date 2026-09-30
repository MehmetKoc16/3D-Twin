import {
  analyzeFit,
  bodyMeasuresForGarment,
  type FitReport,
  type FitVerdict,
  type GarmentMeasureId,
  type GarmentTemplateDef,
  type MeasureId,
  type StoreItemDef,
} from '@dt/avatar-core';

/** Measures that are lengths along the body (the wording "short / long" instead of "tight / loose"). */
const LENGTH_MEASURES: readonly GarmentMeasureId[] = ['length', 'sleeve', 'inseam'];

export function isLengthMeasure(id: GarmentMeasureId): boolean {
  return LENGTH_MEASURES.includes(id);
}

/** Measures whose body value is a reference (body + the template's own length offset) rather than a raw body value. */
const REFERENCE_MEASURES: readonly GarmentMeasureId[] = ['sleeve', 'inseam'];

/**
 * Fit report of one item for the current body. The store chart's sleeve / inseam are compared with the template's
 * intended length on this body (a short-sleeve tee has a sleeve of about 20 cm although the arm is 60 cm long), so
 * those regions read "short / regular / long" relative to the template and are not judged against the raw arm.
 */
export function analyzeItemFit(
  item: StoreItemDef,
  template: GarmentTemplateDef,
  achievedCm: Partial<Record<MeasureId, number>>,
): FitReport | undefined {
  if (!item.sizes.includes(item.selectedSize)) return undefined;
  const body = bodyMeasuresForGarment(achievedCm);
  for (const id of REFERENCE_MEASURES) {
    const bodyValue = body[id];
    if (bodyValue !== undefined) body[id] = bodyValue + (template.defaultEase[id] ?? 0);
  }
  return analyzeFit(item, template, body);
}

/**
 * Regions shown as information only (no verdict): the sleeve of a short-sleeve template, whose native sleeve is far
 * shorter than the arm, says nothing about how the shirt fits.
 */
export function isInfoOnly(template: GarmentTemplateDef, id: GarmentMeasureId): boolean {
  return id === 'sleeve' && (template.defaultEase.sleeve ?? 0) < -15;
}

const SEVERITY: Record<FitVerdict, number> = {
  regular: 0,
  snug: 1,
  loose: 2,
  oversized: 3,
  tight: 4,
};

/** Sort key: worst regions first in the UI. */
export function verdictSeverity(verdict: FitVerdict): number {
  return SEVERITY[verdict];
}

/** Tailwind classes of a verdict chip (the meaning is always also written as text). */
export const VERDICT_STYLE: Record<FitVerdict, string> = {
  tight: 'border-red-400/60 bg-red-500/15 text-red-200',
  snug: 'border-amber-400/60 bg-amber-400/15 text-amber-100',
  regular: 'border-emerald-400/60 bg-emerald-400/15 text-emerald-100',
  loose: 'border-sky-400/60 bg-sky-400/15 text-sky-100',
  oversized: 'border-indigo-400/60 bg-indigo-400/15 text-indigo-100',
};

export function formatEase(easeCm: number): string {
  const rounded = Math.round(easeCm * 10) / 10;
  return `${rounded > 0 ? '+' : ''}${rounded.toFixed(1)} cm`;
}
