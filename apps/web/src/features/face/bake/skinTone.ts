import { at } from '../util';

/** Robust skin tone estimation in linear RGB. */

export type Rgb = readonly [number, number, number];

export function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

export function linearToSrgb(c: number): number {
  const v = Math.max(0, Math.min(1, c));
  return v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055;
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? at(sorted, mid) : (at(sorted, mid - 1) + at(sorted, mid)) / 2;
}

/** Channel-wise median of linear RGB samples. */
export function medianLinearRgb(samples: readonly Rgb[]): Rgb {
  return [
    median(samples.map((s) => s[0])),
    median(samples.map((s) => s[1])),
    median(samples.map((s) => s[2])),
  ];
}

export function linearRgbToHex(rgb: Rgb): string {
  const hex = (c: number) => Math.round(linearToSrgb(c) * 255).toString(16).padStart(2, '0');
  return `#${hex(rgb[0])}${hex(rgb[1])}${hex(rgb[2])}`;
}

export function hexToLinearRgb(hex: string): Rgb {
  const n = Number.parseInt(hex.slice(1), 16);
  return [srgbToLinear(((n >> 16) & 255) / 255), srgbToLinear(((n >> 8) & 255) / 255), srgbToLinear((n & 255) / 255)];
}

export interface PixelSource {
  data: ArrayLike<number>;
  width: number;
  height: number;
}

/**
 * Samples small windows around the given landmark indices (normalized coordinates, 3 floats per landmark)
 * and returns the median linear RGB, or undefined if nothing could be sampled.
 */
export function sampleLandmarkColors(
  image: PixelSource,
  landmarks: ArrayLike<number>,
  indices: readonly number[],
  radiusPx: number,
): Rgb | undefined {
  const samples: Rgb[] = [];
  const r = Math.max(0, Math.round(radiusPx));
  for (const index of indices) {
    const cx = Math.round(at(landmarks, index * 3) * image.width);
    const cy = Math.round(at(landmarks, index * 3 + 1) * image.height);
    for (let dy = -r; dy <= r; dy += 1) {
      for (let dx = -r; dx <= r; dx += 1) {
        const x = cx + dx;
        const y = cy + dy;
        if (x < 0 || y < 0 || x >= image.width || y >= image.height) continue;
        const o = (y * image.width + x) * 4;
        samples.push([
          srgbToLinear(at(image.data, o) / 255),
          srgbToLinear(at(image.data, o + 1) / 255),
          srgbToLinear(at(image.data, o + 2) / 255),
        ]);
      }
    }
  }
  return samples.length ? medianLinearRgb(samples) : undefined;
}
