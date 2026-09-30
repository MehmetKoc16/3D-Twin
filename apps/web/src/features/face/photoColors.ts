import { at } from './util';
import { linearRgbToHex, medianLinearRgb, srgbToLinear, type PixelSource, type Rgb } from './bake/skinTone';
import { landmarkBox } from './quality';

/** MediaPipe iris landmarks: centre first, then the four ring points (468-472 and 473-477). */
export const IRIS_LANDMARKS: readonly (readonly number[])[] = [
  [468, 469, 470, 471, 472],
  [473, 474, 475, 476, 477],
];
/** Landmarks along both eyebrows (upper and lower contour of each). */
export const BROW_LANDMARKS: readonly number[] = [
  70, 63, 105, 66, 107, 46, 53, 52, 65, 55, 300, 293, 334, 296, 336, 276, 283, 282, 295, 285,
];

/** Colours read from the selfie (sRGB hex). */
export interface PhotoColors {
  iris?: string;
  brow?: string;
}

function lumaOf(data: ArrayLike<number>, o: number): number {
  return 0.2126 * at(data, o) + 0.7152 * at(data, o + 1) + 0.0722 * at(data, o + 2);
}

function percentile(sorted: number[], p: number): number {
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(p * sorted.length)))] ?? 0;
}

function pixelLinear(data: ArrayLike<number>, o: number): Rgb {
  return [srgbToLinear(at(data, o) / 255), srgbToLinear(at(data, o + 1) / 255), srgbToLinear(at(data, o + 2) / 255)];
}

/**
 * Iris colour of one or both eyes: the median colour of the pixels in the iris ring, without the pupil (inner
 * radius), the lid-shaded upper part, and the darkest / brightest luminance tails (pupil edge, catch lights).
 */
export function sampleIrisColor(image: PixelSource, landmarks: ArrayLike<number>): string | undefined {
  const candidates: { o: number; luminance: number }[] = [];
  for (const ring of IRIS_LANDMARKS) {
    const [centre, ...edge] = ring;
    if (centre === undefined || landmarks.length < (Math.max(...ring) + 1) * 3) continue;
    const cx = at(landmarks, centre * 3) * image.width;
    const cy = at(landmarks, centre * 3 + 1) * image.height;
    const radius =
      edge.reduce(
        (sum, i) => sum + Math.hypot(at(landmarks, i * 3) * image.width - cx, at(landmarks, i * 3 + 1) * image.height - cy),
        0,
      ) / Math.max(1, edge.length);
    if (!(radius >= 2)) continue; // too small to sample
    for (let y = Math.floor(cy - radius); y <= Math.ceil(cy + radius); y++) {
      for (let x = Math.floor(cx - radius); x <= Math.ceil(cx + radius); x++) {
        if (x < 0 || y < 0 || x >= image.width || y >= image.height) continue;
        const d = Math.hypot(x - cx, y - cy) / radius;
        if (d < 0.4 || d > 0.85) continue; // outside the pupil, inside the limbus
        if (y - cy < -0.45 * radius) continue; // the upper iris is shaded by the lid and the lashes
        const o = (y * image.width + x) * 4;
        candidates.push({ o, luminance: lumaOf(image.data, o) });
      }
    }
  }
  if (candidates.length < 12) return undefined;
  const sorted = candidates.map((c) => c.luminance).sort((a, b) => a - b);
  const low = percentile(sorted, 0.15);
  const high = percentile(sorted, 0.85);
  const kept = candidates.filter((c) => c.luminance >= low && c.luminance <= high);
  if (kept.length === 0) return undefined;
  return linearRgbToHex(medianLinearRgb(kept.map((c) => pixelLinear(image.data, c.o))));
}

/**
 * Eyebrow (and hair) colour: around the brow contour landmarks the skin between the hairs is lighter, so the darkest
 * third of the pixels is kept and its median colour returned.
 */
export function sampleBrowColor(image: PixelSource, landmarks: ArrayLike<number>): string | undefined {
  if (landmarks.length < 478 * 3) return undefined;
  const box = landmarkBox(landmarks);
  const faceHeightPx = (box.maxY - box.minY) * image.height;
  const r = Math.max(1, Math.round(faceHeightPx * 0.012));
  const pixels: { o: number; luminance: number }[] = [];
  for (const index of BROW_LANDMARKS) {
    const cx = Math.round(at(landmarks, index * 3) * image.width);
    const cy = Math.round(at(landmarks, index * 3 + 1) * image.height);
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const x = cx + dx;
        const y = cy + dy;
        if (x < 0 || y < 0 || x >= image.width || y >= image.height) continue;
        const o = (y * image.width + x) * 4;
        pixels.push({ o, luminance: lumaOf(image.data, o) });
      }
    }
  }
  if (pixels.length < 12) return undefined;
  const cutoff = percentile(
    pixels.map((p) => p.luminance).sort((a, b) => a - b),
    0.15,
  );
  const dark = pixels.filter((p) => p.luminance <= cutoff);
  return linearRgbToHex(medianLinearRgb(dark.map((p) => pixelLinear(image.data, p.o))));
}

/** Decodes the photo at up to `maxSide` pixels (an iris is only a few pixels wide, so this is larger than the skin read). */
export function readPhotoPixelsForColors(image: ImageBitmap, maxSide = 2048): ImageData {
  const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(image.width * scale));
  canvas.height = Math.max(1, Math.round(image.height * scale));
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('2D canvas is not available');
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  return context.getImageData(0, 0, canvas.width, canvas.height);
}

/** Iris and eyebrow colours of a detected selfie; entries are missing when they could not be sampled. */
export function samplePhotoColors(image: ImageBitmap, landmarks: ArrayLike<number>): PhotoColors {
  const pixels = readPhotoPixelsForColors(image);
  return { iris: sampleIrisColor(pixels, landmarks), brow: sampleBrowColor(pixels, landmarks) };
}
