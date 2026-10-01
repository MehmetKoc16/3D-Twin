import { Vector2, type Texture } from 'three';
import type { TwinModel } from './twinModel';

export const NEUTRAL_SKIN_TONE = '#c99a7e';

export interface TexturePixels {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

type Color3 = [number, number, number];

/** Oklab from sRGB bytes: distances are perceptual, rather than RGB channel differences. */
export function perceptualColor(rgb: readonly number[]): Color3 {
  const linear = rgb.map((byte) => {
    const c = byte! / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  const [r, g, b] = linear as Color3;
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

export function colorDistance(a: readonly number[], b: readonly number[]): number {
  return Math.hypot(a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!);
}

export function hexBytes(hex: string): Color3 {
  return [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16)) as Color3;
}

function median(values: number[]): number {
  values.sort((a, b) => a - b);
  const middle = Math.floor(values.length / 2);
  return values.length % 2 ? values[middle]! : (values[middle - 1]! + values[middle]!) / 2;
}

/** UVs here are image coordinates (glTF's flipY=false), including the texture's UV transform. */
export function sampleSkinTone(
  pixels: TexturePixels,
  uv: ArrayLike<number>,
  joints: ArrayLike<number>,
  weights: ArrayLike<number>,
  names: readonly string[],
): string {
  const texels = new Set<number>();
  for (let v = 0; v < uv.length / 2; v++) {
    let forearm = false;
    for (let j = 0; j < 4; j++)
      if (/^lowerarm_[lr]$/.test(names[joints[v * 4 + j]!] ?? '') && weights[v * 4 + j]! > 0.5)
        forearm = true;
    if (!forearm || !Number.isFinite(uv[v * 2]) || !Number.isFinite(uv[v * 2 + 1])) continue;
    const x = Math.min(pixels.width - 1, Math.max(0, Math.floor(uv[v * 2]! * pixels.width)));
    const y = Math.min(pixels.height - 1, Math.max(0, Math.floor(uv[v * 2 + 1]! * pixels.height)));
    texels.add((y * pixels.width + x) * 4);
  }
  const samples: Color3[] = [];
  for (const offset of texels) {
    const rgb: Color3 = [pixels.data[offset]!, pixels.data[offset + 1]!, pixels.data[offset + 2]!];
    const lightness = perceptualColor(rgb)[0];
    // Transparent atlas padding, clipped highlights and deep shadows cannot identify skin.
    if (pixels.data[offset + 3]! >= 128 && lightness > 0.12 && lightness < 0.97) samples.push(rgb);
  }
  if (!samples.length) return NEUTRAL_SKIN_TONE;
  const center = [0, 1, 2].map((c) => median(samples.map((s) => s[c]!))) as Color3;
  const lab = perceptualColor(center);
  const distances = samples.map((rgb) => colorDistance(perceptualColor(rgb), lab));
  const cutoff = Math.max(0.06, median([...distances]) * 3);
  const retained = samples.filter((_, i) => distances[i]! <= cutoff);
  if (!retained.length) return NEUTRAL_SKIN_TONE;
  return `#${[0, 1, 2]
    .map((c) =>
      Math.round(median(retained.map((s) => s[c]!)))
        .toString(16)
        .padStart(2, '0'),
    )
    .join('')}`;
}

/** Read only in the browser; an unavailable/tainted image uses the neutral fallback. */
export function readTexturePixels(texture: Texture | null): TexturePixels | null {
  if (!texture) return null;
  const image: unknown = texture.image;
  if (!image || typeof image !== 'object' || !('width' in image) || !('height' in image))
    return null;
  const width = Number(image.width),
    height = Number(image.height);
  if (!width || !height) return null;
  if ('data' in image && image.data instanceof Uint8Array)
    return { width, height, data: new Uint8ClampedArray(image.data) };
  if ('data' in image && image.data instanceof Uint8ClampedArray)
    return { width, height, data: image.data.slice() };
  if (typeof document === 'undefined') return null;
  try {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) return null;
    context.drawImage(image as CanvasImageSource, 0, 0);
    return { width, height, data: context.getImageData(0, 0, width, height).data };
  } catch {
    return null;
  }
}

export function textureUVs(uv: Float32Array | null, texture: Texture | null): Float32Array | null {
  if (!uv || !texture) return uv;
  if (texture.matrixAutoUpdate) texture.updateMatrix();
  const out = new Float32Array(uv.length);
  const point = new Vector2();
  for (let v = 0; v < uv.length / 2; v++) {
    texture.transformUv(point.set(uv[v * 2]!, uv[v * 2 + 1]!));
    out.set([point.x, point.y], v * 2);
  }
  return out;
}

// The original GLB buffer survives standard/twin switches; weak keys retain no removed personal data.
const toneCache = new WeakMap<object, string>();
export function resolveSkinTone(
  model: TwinModel,
  names: readonly string[],
  key: object = model,
): string {
  const cached = toneCache.get(key);
  if (cached) return cached;
  const pixels = readTexturePixels(model.material.map);
  const uv = textureUVs(model.uv, model.material.map);
  const tone =
    pixels && uv
      ? sampleSkinTone(pixels, uv, model.skinIndex, model.skinWeight, names)
      : NEUTRAL_SKIN_TONE;
  toneCache.set(key, tone);
  return tone;
}
