import type { ShoeSystem } from '@dt/avatar-core';

// Approximate common adult sizing. Brand-specific fit varies.
// TODO(wave2): replace with @dt/avatar-core footLengthCmFromShoe
export const shoeSizes: { EU: number; UK: number; US_M: number; US_W: number; footCm: number }[] = [
  { EU: 35, UK: 2.5, US_M: 3.5, US_W: 5, footCm: 22.0 },
  { EU: 36, UK: 3, US_M: 4, US_W: 5.5, footCm: 22.5 },
  { EU: 37, UK: 4, US_M: 5, US_W: 6.5, footCm: 23.5 },
  { EU: 38, UK: 5, US_M: 6, US_W: 7.5, footCm: 24.0 },
  { EU: 39, UK: 5.5, US_M: 6.5, US_W: 8, footCm: 24.5 },
  { EU: 40, UK: 6.5, US_M: 7.5, US_W: 9, footCm: 25.5 },
  { EU: 41, UK: 7, US_M: 8, US_W: 9.5, footCm: 26.0 },
  { EU: 42, UK: 8, US_M: 9, US_W: 10.5, footCm: 27.0 },
  { EU: 43, UK: 9, US_M: 10, US_W: 11.5, footCm: 27.5 },
  { EU: 44, UK: 9.5, US_M: 10.5, US_W: 12, footCm: 28.0 },
  { EU: 45, UK: 10.5, US_M: 11.5, US_W: 13, footCm: 29.0 },
  { EU: 46, UK: 11, US_M: 12, US_W: 13.5, footCm: 29.5 },
  { EU: 47, UK: 12, US_M: 13, US_W: 14.5, footCm: 30.5 },
  { EU: 48, UK: 13, US_M: 14, US_W: 15.5, footCm: 31.0 },
];

export function shoeRow(system: ShoeSystem, size: number) {
  return shoeSizes.reduce((closest, row) => Math.abs(row[system] - size) < Math.abs(closest[system] - size) ? row : closest);
}

export function footLengthCmFromShoe(system: ShoeSystem, size: number): number {
  return shoeRow(system, size).footCm;
}

export function convertShoeSize(size: number, from: ShoeSystem, to: ShoeSystem): number {
  return shoeRow(from, size)[to];
}

