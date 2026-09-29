import type { ShoeSystem } from './contracts';

/**
 * Approximate shoe-size table (typical unisex / mainstream brand charts). Brands differ by up to a full size,
 * so treat every value as an estimate. Foot length follows the sanity relation EU ~ 1.5 * foot_cm + 2
 * (Paris point = 2/3 cm), rounded to 0.1 cm. US_M = UK + 1, US_W = UK + 2.5 (adult sizes).
 * Sizes between rows are linearly interpolated; outside EU 35..48 the end segments are extrapolated linearly.
 */
const TABLE: ReadonlyArray<{ eu: number; uk: number; usM: number; usW: number; footCm: number }> = [
  { eu: 35, uk: 2.5, usM: 3.5, usW: 5, footCm: 22.0 },
  { eu: 36, uk: 3.5, usM: 4.5, usW: 6, footCm: 22.7 },
  { eu: 37, uk: 4, usM: 5, usW: 6.5, footCm: 23.3 },
  { eu: 38, uk: 5, usM: 6, usW: 7.5, footCm: 24.0 },
  { eu: 39, uk: 6, usM: 7, usW: 8.5, footCm: 24.7 },
  { eu: 40, uk: 6.5, usM: 7.5, usW: 9, footCm: 25.3 },
  { eu: 41, uk: 7.5, usM: 8.5, usW: 10, footCm: 26.0 },
  { eu: 42, uk: 8, usM: 9, usW: 10.5, footCm: 26.7 },
  { eu: 43, uk: 9, usM: 10, usW: 11.5, footCm: 27.3 },
  { eu: 44, uk: 9.5, usM: 10.5, usW: 12, footCm: 28.0 },
  { eu: 45, uk: 10.5, usM: 11.5, usW: 13, footCm: 28.7 },
  { eu: 46, uk: 11, usM: 12, usW: 13.5, footCm: 29.3 },
  { eu: 47, uk: 12, usM: 13, usW: 14.5, footCm: 30.0 },
  { eu: 48, uk: 12.5, usM: 13.5, usW: 15, footCm: 30.7 },
];

type Row = (typeof TABLE)[number];
const SIZE_KEY: Record<ShoeSystem, (r: Row) => number> = {
  EU: (r) => r.eu,
  UK: (r) => r.uk,
  US_M: (r) => r.usM,
  US_W: (r) => r.usW,
};

function assertFinite(x: number, what: string): void {
  if (!Number.isFinite(x)) throw new RangeError(`${what} must be a finite number`);
}

/** Piecewise-linear map through (xs[i], ys[i]) with linear extrapolation on both ends. xs must be increasing. */
function interpolate(xs: number[], ys: number[], x: number): number {
  const n = xs.length;
  let i = 0;
  while (i < n - 2 && x > xs[i + 1]!) i++;
  const x0 = xs[i]!;
  const x1 = xs[i + 1]!;
  return ys[i]! + ((ys[i + 1]! - ys[i]!) * (x - x0)) / (x1 - x0);
}

/** Foot length in cm for a shoe size (approximate, brand dependent). */
export function footLengthCmFromShoe(shoe: { system: ShoeSystem; size: number }): number {
  assertFinite(shoe.size, 'shoe size');
  const key = SIZE_KEY[shoe.system];
  return interpolate(
    TABLE.map(key),
    TABLE.map((r) => r.footCm),
    shoe.size,
  );
}

/** Shoe size (continuous, not rounded) in `system` for a foot length in cm. Inverse of footLengthCmFromShoe. */
export function shoeFromFootLengthCm(cm: number, system: ShoeSystem): number {
  assertFinite(cm, 'foot length');
  const key = SIZE_KEY[system];
  return interpolate(
    TABLE.map((r) => r.footCm),
    TABLE.map(key),
    cm,
  );
}

/** Converts a size between systems via foot length (continuous result). */
export function convertShoeSize(
  shoe: { system: ShoeSystem; size: number },
  to: ShoeSystem,
): number {
  return shoeFromFootLengthCm(footLengthCmFromShoe(shoe), to);
}

/** Rounds to the nearest half size, the granularity retailers use. */
export function roundShoeSize(size: number): number {
  return Math.round(size * 2) / 2;
}
