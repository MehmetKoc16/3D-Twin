import { at } from '../util';

/** Delighting: flatten low-frequency luminance (directional light, shadows) in linear space. */

export function luminanceLinear(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Multiplicative gain that pulls a pixel with blurred luminance `blurred` toward the region `mean`. */
export function delightGain(blurred: number, mean: number, strength: number): number {
  const ratio = mean / Math.max(blurred, 1e-4);
  const gain = 1 + (ratio - 1) * strength;
  return Math.max(0.5, Math.min(2, gain));
}

/** Separable box blur (edge-clamped by shrinking the window), radius in pixels. */
export function boxBlur(src: Float32Array, width: number, height: number, radius: number): Float32Array {
  const pass = (
    input: Float32Array, output: Float32Array, length: number, lines: number, stride: number, lineStride: number,
  ) => {
    for (let line = 0; line < lines; line += 1) {
      const base = line * lineStride;
      let sum = 0;
      let count = 0;
      for (let i = 0; i < Math.min(radius, length); i += 1) {
        sum += at(input, base + i * stride);
        count += 1;
      }
      for (let i = 0; i < length; i += 1) {
        const add = i + radius;
        if (add < length) {
          sum += at(input, base + add * stride);
          count += 1;
        }
        const drop = i - radius - 1;
        if (drop >= 0) {
          sum -= at(input, base + drop * stride);
          count -= 1;
        }
        output[base + i * stride] = sum / count;
      }
    }
  };
  const tmp = new Float32Array(width * height);
  const out = new Float32Array(width * height);
  pass(src, tmp, width, height, 1, width);
  pass(tmp, out, height, width, width, 1);
  return out;
}

/** Mask-weighted (normalized) blur: pixels with zero weight do not contribute to their neighbours. */
export function maskedBlur(
  values: Float32Array, weights: Float32Array, width: number, height: number, radius: number,
): Float32Array {
  const weighted = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) weighted[i] = at(values, i) * at(weights, i);
  const num = boxBlur(weighted, width, height, radius);
  const den = boxBlur(weights, width, height, radius);
  const out = new Float32Array(values.length);
  for (let i = 0; i < out.length; i += 1) out[i] = at(den, i) > 1e-6 ? at(num, i) / at(den, i) : 0;
  return out;
}
