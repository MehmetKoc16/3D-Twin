import { at } from '../util';
import { delightGain, luminanceLinear, maskedBlur } from './delight';
import { smoothstep } from './mask';
import { linearToSrgb, srgbToLinear, type Rgb } from './skinTone';

export interface ProcessInput {
  /** RGBA bytes (sRGB, straight alpha) of the warped photo crop; alpha > 0 where the photo covers. */
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  /** Mask alpha 0..1 per pixel (feathered oval, forehead fade). */
  maskAlpha: Float32Array;
  /** Signed inside distance per pixel to the oval boundary (px). */
  inside: Float32Array;
  feather: number;
  skinLinear: Rgb;
  /** 0 = no delighting, 1 = full flattening. */
  strength: number;
  blurRadius: number;
}

/**
 * In-place: delight (linear space), colour-match the border toward the skin tone, then write the final
 * alpha = mask * photo coverage. Output stays straight-alpha sRGB.
 */
export function processFaceCrop(input: ProcessInput): void {
  const { rgba, width, height, maskAlpha, inside, feather, skinLinear, strength, blurRadius } = input;
  const count = width * height;
  const linear = new Float32Array(count * 3);
  const luma = new Float32Array(count);
  const weight = new Float32Array(count);
  let weightSum = 0;
  let lumaSum = 0;
  for (let i = 0; i < count; i += 1) {
    const r = srgbToLinear(at(rgba, i * 4) / 255);
    const g = srgbToLinear(at(rgba, i * 4 + 1) / 255);
    const b = srgbToLinear(at(rgba, i * 4 + 2) / 255);
    linear[i * 3] = r;
    linear[i * 3 + 1] = g;
    linear[i * 3 + 2] = b;
    luma[i] = luminanceLinear(r, g, b);
    weight[i] = at(maskAlpha, i) * (at(rgba, i * 4 + 3) / 255);
    weightSum += at(weight, i);
    lumaSum += at(luma, i) * at(weight, i);
  }
  const mean = weightSum > 0 ? lumaSum / weightSum : 0;
  const blurred = strength > 0 ? maskedBlur(luma, weight, width, height, blurRadius) : luma;
  const borderBand = Math.max(feather * 3, 1);

  for (let i = 0; i < count; i += 1) {
    const alpha = at(maskAlpha, i) * (at(rgba, i * 4 + 3) / 255);
    if (alpha <= 0) {
      rgba[i * 4 + 3] = 0;
      continue;
    }
    const gain = strength > 0 ? delightGain(at(blurred, i), mean, strength) : 1;
    const borderWeight = 0.6 * (1 - smoothstep(0, borderBand, at(inside, i)));
    for (let c = 0; c < 3; c += 1) {
      const lit = at(linear, i * 3 + c) * gain;
      const matched = lit + (skinLinear[c as 0 | 1 | 2] - lit) * borderWeight;
      rgba[i * 4 + c] = Math.round(linearToSrgb(matched) * 255);
    }
    rgba[i * 4 + 3] = Math.round(alpha * 255);
  }
}
