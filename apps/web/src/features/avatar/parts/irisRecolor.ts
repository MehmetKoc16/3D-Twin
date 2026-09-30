/**
 * Iris recolouring of the eye texture (pure pixel math, no DOM). The upstream texture has a neutral grey iris; the
 * pixels inside the iris circle are replaced by `colour * (luminance / reference luminance)`, so the radial fibre
 * detail of the grey iris survives, the pupil stays dark and the sclera outside the circle is untouched. The circle
 * edge fades smoothly. Pixels are sRGB-encoded RGBA bytes, `y` runs top to bottom (glTF texture space, v down).
 */

export interface IrisRegion {
  /** Circle centre in texture space, 0..1 (u right, v down). */
  center: readonly [number, number];
  /** Radius as a fraction of the texture width. */
  radius: number;
}

/** Inside this normalised radius the recolour is fully applied, it fades out to `FADE_END`. */
export const FADE_START = 0.97;
export const FADE_END = 1.12;
/** The reference luminance is measured on the iris ring (outside the pupil, inside the limbus). */
const REF_INNER = 0.4;
const REF_OUTER = 0.9;

export function parseHex(hex: string): [number, number, number] {
  const n = Number.parseInt(hex.replace('#', ''), 16);
  if (!Number.isFinite(n)) return [128, 128, 128];
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Rec. 709 luminance of sRGB-encoded 0..255 channels, 0..1 (a perceptual stand-in, not linear light). */
export function luma(r: number, g: number, b: number): number {
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
}

function smoothstep(lo: number, hi: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - lo) / (hi - lo)));
  return t * t * (3 - 2 * t);
}

/** Mean luminance of the iris ring of the source texture (the value that maps to the chosen colour). */
export function irisReferenceLuma(
  source: ArrayLike<number>,
  width: number,
  height: number,
  iris: IrisRegion,
): number {
  const cx = iris.center[0] * width;
  const cy = iris.center[1] * height;
  const radius = iris.radius * width;
  let sum = 0;
  let count = 0;
  const y0 = Math.max(0, Math.floor(cy - radius));
  const y1 = Math.min(height - 1, Math.ceil(cy + radius));
  const x0 = Math.max(0, Math.floor(cx - radius));
  const x1 = Math.min(width - 1, Math.ceil(cx + radius));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / radius;
      if (d < REF_INNER || d > REF_OUTER) continue;
      const o = (y * width + x) * 4;
      sum += luma(source[o]!, source[o + 1]!, source[o + 2]!);
      count++;
    }
  }
  return count > 0 ? Math.max(0.05, sum / count) : 0.5;
}

/**
 * Writes the recoloured texture into `out` (same size as `source`). Only pixels within `FADE_END` of the iris
 * centre are touched; the rest is copied. `referenceLuma` defaults to `irisReferenceLuma(source)`.
 */
export function recolorIris(
  source: ArrayLike<number>,
  out: Uint8ClampedArray,
  width: number,
  height: number,
  iris: IrisRegion,
  colorHex: string,
  referenceLuma: number = irisReferenceLuma(source, width, height, iris),
): void {
  for (let i = 0; i < width * height * 4; i++) out[i] = source[i]!;
  const [cr, cg, cb] = parseHex(colorHex);
  const cx = iris.center[0] * width;
  const cy = iris.center[1] * height;
  const radius = iris.radius * width;
  const reach = radius * FADE_END;
  const y0 = Math.max(0, Math.floor(cy - reach));
  const y1 = Math.min(height - 1, Math.ceil(cy + reach));
  const x0 = Math.max(0, Math.floor(cx - reach));
  const x1 = Math.min(width - 1, Math.ceil(cx + reach));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / radius;
      const weight = 1 - smoothstep(FADE_START, FADE_END, d);
      if (weight <= 0) continue;
      const o = (y * width + x) * 4;
      const r = source[o]!;
      const g = source[o + 1]!;
      const b = source[o + 2]!;
      // keep the grey iris' luminance detail: multiply the colour by its ratio to the ring's mean
      const ratio = Math.min(1.8, luma(r, g, b) / referenceLuma);
      out[o] = r + (Math.min(255, cr * ratio) - r) * weight;
      out[o + 1] = g + (Math.min(255, cg * ratio) - g) * weight;
      out[o + 2] = b + (Math.min(255, cb * ratio) - b) * weight;
    }
  }
}

/**
 * Tint factor of an alpha-masked neutral hair / brow texture: the mean linear value of the covered texels, so that
 * `material.color = picked colour / factor` shows the picked colour on average instead of a darker one.
 */
export function meanCoveredLinear(pixels: ArrayLike<number>, alphaCutoff: number): number {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i + 3]! / 255 < alphaCutoff) continue;
    const c = Math.max(pixels[i]!, pixels[i + 1]!, pixels[i + 2]!) / 255;
    sum += c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    count++;
  }
  return count > 0 ? Math.max(0.02, sum / count) : 1;
}
