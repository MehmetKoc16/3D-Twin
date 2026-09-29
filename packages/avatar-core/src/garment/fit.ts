import type { MeasureId } from '../contracts';
import type {
  FitReport,
  FitVerdict,
  GarmentMeasureId,
  GarmentTemplateDef,
  StoreItemDef,
} from '../garmentContracts';

type Measures = Partial<Record<GarmentMeasureId, number>>;
const GIRTHS: readonly GarmentMeasureId[] = ['chest', 'waist', 'hip', 'thigh'];
const SEVERITY: Record<FitVerdict, number> = {
  regular: 0,
  snug: 1,
  loose: 2,
  oversized: 3,
  tight: 4,
};

/**
 * Ease thresholds in cm. Chest: <0 tight, [0,4) snug, [4,12] regular,
 * (12,20] loose, >20 oversized. Waist/hip use 0/3/10/18 and thigh 0/2/7/12.
 * Lengths use garment minus body: <-2 tight, [-2,-0.5) snug, [-0.5,2] regular,
 * (2,5] loose, >5 oversized. Shoe inner length minus foot: <0.5 tight,
 * [0.5,1) snug, [1,1.8] regular, (1.8,2.5] loose, >2.5 oversized.
 */
export function verdictForEase(id: GarmentMeasureId, easeCm: number): FitVerdict {
  if (id === 'footLength') {
    if (easeCm < 0.5) return 'tight';
    if (easeCm < 1) return 'snug';
    if (easeCm <= 1.8) return 'regular';
    if (easeCm <= 2.5) return 'loose';
    return 'oversized';
  }
  const [snug, regular, loose] =
    id === 'chest'
      ? [4, 12, 20]
      : id === 'waist' || id === 'hip'
        ? [3, 10, 18]
        : id === 'thigh'
          ? [2, 7, 12]
          : [-0.5, 2, 5];
  if (GIRTHS.includes(id)) {
    if (easeCm < 0) return 'tight';
    if (easeCm < snug) return 'snug';
  } else {
    if (easeCm < -2) return 'tight';
    if (easeCm < snug) return 'snug';
  }
  if (easeCm <= regular) return 'regular';
  if (easeCm <= loose) return 'loose';
  return 'oversized';
}

/** Returns the regular-fit size nearest to default girth ease, excluding sizes tight in any known girth. */
export function recommendSize(
  item: StoreItemDef,
  template: GarmentTemplateDef,
  bodyCm: Measures,
): string | undefined {
  let best: string | undefined;
  let bestScore = Infinity;
  for (let s = 0; s < item.sizes.length; s++) {
    let score = 0;
    let count = 0;
    let tight = false;
    for (const id of GIRTHS) {
      if (id === 'thigh') continue;
      const body = bodyCm[id];
      const garment = item.chart[id]?.[s];
      if (
        body === undefined ||
        garment === undefined ||
        !Number.isFinite(body) ||
        !Number.isFinite(garment)
      )
        continue;
      const ease = garment - body;
      if (ease < 0) {
        tight = true;
        break;
      }
      const target = template.defaultEase[id] ?? (id === 'chest' ? 8 : 6.5);
      score += Math.abs(ease - target);
      count++;
    }
    if (!tight && count > 0 && score < bestScore) {
      best = item.sizes[s];
      bestScore = score;
    }
  }
  return best;
}

export function analyzeFit(
  item: StoreItemDef,
  template: GarmentTemplateDef,
  bodyCm: Measures,
): FitReport {
  const sizeIndex = item.sizes.indexOf(item.selectedSize);
  if (sizeIndex < 0) throw new Error(`garment fit: unknown size ${item.selectedSize}`);
  const regions: FitReport['regions'] = [];
  for (const id of Object.keys(item.chart) as GarmentMeasureId[]) {
    const body = bodyCm[id];
    const garment = item.chart[id]?.[sizeIndex];
    if (
      body === undefined ||
      garment === undefined ||
      !Number.isFinite(body) ||
      !Number.isFinite(garment)
    )
      continue;
    const easeCm = garment - body;
    regions.push({
      id,
      bodyCm: body,
      garmentCm: garment,
      easeCm,
      verdict: verdictForEase(id, easeCm),
    });
  }
  const girths = regions.filter((r) => GIRTHS.includes(r.id));
  const meaningful = girths.length ? girths : regions;
  const overall = meaningful.reduce<FitVerdict>(
    (worst, r) => (SEVERITY[r.verdict] > SEVERITY[worst] ? r.verdict : worst),
    'regular',
  );
  const recommendedSize = recommendSize(item, template, bodyCm);
  return {
    itemId: item.id,
    size: item.selectedSize,
    regions,
    overall,
    ...(recommendedSize === undefined ? {} : { recommendedSize }),
  };
}

/** Translate achieved avatar measurements (cm) into the store-chart vocabulary. */
export function bodyMeasuresForGarment(achievedCm: Partial<Record<MeasureId, number>>): Measures {
  const result: Measures = {};
  const same = ['chest', 'waist', 'hip', 'shoulder', 'inseam', 'thigh', 'footLength'] as const;
  for (const id of same) {
    const value = achievedCm[id];
    if (value !== undefined) result[id] = value;
  }
  if (achievedCm.armLength !== undefined) result.sleeve = achievedCm.armLength;
  return result;
}
