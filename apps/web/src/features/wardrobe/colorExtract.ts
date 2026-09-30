/** Dominant colours of a product photo: simple deterministic k-means in RGB. The image stays on the device. */

export interface ColorExtractOptions {
  /** Number of clusters (default 4). */
  k?: number;
  /** Ignore pixels close to the image border colour (a plain studio background). Default true. */
  dropBackground?: boolean;
}

interface Cluster {
  rgb: [number, number, number];
  count: number;
}

const SAMPLE_LIMIT = 6000;
const BACKGROUND_DISTANCE = 32;

function toHex(rgb: readonly [number, number, number]): string {
  return `#${rgb
    .map((c) =>
      Math.max(0, Math.min(255, Math.round(c)))
        .toString(16)
        .padStart(2, '0'),
    )
    .join('')}`;
}

function distance2(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const dr = a[0]! - b[0]!;
  const dg = a[1]! - b[1]!;
  const db = a[2]! - b[2]!;
  return dr * dr + dg * dg + db * db;
}

/** 0 (grey) .. 1 (fully saturated) chroma of an RGB colour. */
function chroma(rgb: ArrayLike<number>): number {
  const max = Math.max(rgb[0]!, rgb[1]!, rgb[2]!);
  const min = Math.min(rgb[0]!, rgb[1]!, rgb[2]!);
  return max === 0 ? 0 : (max - min) / max;
}

/** Mean colour of the outermost ring of pixels: the background of a typical product shot. */
function borderColor(
  data: ArrayLike<number>,
  width: number,
  height: number,
): [number, number, number] {
  let r = 0;
  let g = 0;
  let b = 0;
  let n = 0;
  const add = (x: number, y: number): void => {
    const o = (y * width + x) * 4;
    r += data[o]!;
    g += data[o + 1]!;
    b += data[o + 2]!;
    n++;
  };
  for (let x = 0; x < width; x++) {
    add(x, 0);
    add(x, height - 1);
  }
  for (let y = 1; y < height - 1; y++) {
    add(0, y);
    add(width - 1, y);
  }
  return n > 0 ? [r / n, g / n, b / n] : [0, 0, 0];
}

/**
 * Returns up to `k` swatches (hex, best first) of an RGBA pixel buffer. Pixels that are transparent or close to the
 * border colour are ignored, so a garment on a white background yields the garment colour. Clusters are ranked by
 * pixel share, boosted for saturated colours. Deterministic (farthest-point initialisation, no randomness).
 */
export function dominantColors(
  data: ArrayLike<number>,
  width: number,
  height: number,
  options: ColorExtractOptions = {},
): string[] {
  const k = Math.max(1, Math.min(8, options.k ?? 4));
  const total = width * height;
  if (total <= 0 || data.length < total * 4) return [];
  const background =
    options.dropBackground === false ? undefined : borderColor(data, width, height);
  const stride = Math.max(1, Math.floor(total / SAMPLE_LIMIT));

  const collect = (skipBackground: boolean): number[][] => {
    const points: number[][] = [];
    for (let p = 0; p < total; p += stride) {
      const o = p * 4;
      if (data[o + 3]! < 128) continue;
      const rgb = [data[o]!, data[o + 1]!, data[o + 2]!];
      if (
        skipBackground &&
        background &&
        distance2(rgb, background) < BACKGROUND_DISTANCE * BACKGROUND_DISTANCE
      )
        continue;
      points.push(rgb);
    }
    return points;
  };
  let points = collect(true);
  // a photo that is (almost) only background: fall back to every opaque pixel
  if (points.length < Math.max(8, total / stride / 50)) points = collect(false);
  if (points.length === 0) return [];

  // farthest-point initialisation starting from the mean
  const mean: [number, number, number] = [0, 0, 0];
  for (const p of points) {
    mean[0] += p[0]!;
    mean[1] += p[1]!;
    mean[2] += p[2]!;
  }
  mean[0] /= points.length;
  mean[1] /= points.length;
  mean[2] /= points.length;
  const centers: [number, number, number][] = [];
  let seed = mean;
  for (let c = 0; c < Math.min(k, points.length); c++) {
    let best = points[0]!;
    let bestDistance = -1;
    for (const p of points) {
      const nearest =
        c === 0 ? distance2(p, seed) : Math.min(...centers.map((center) => distance2(p, center)));
      if (nearest > bestDistance) {
        bestDistance = nearest;
        best = p;
      }
    }
    centers.push([best[0]!, best[1]!, best[2]!]);
    seed = centers[centers.length - 1]!;
  }

  const clusters: Cluster[] = centers.map((rgb) => ({ rgb, count: 0 }));
  for (let iteration = 0; iteration < 12; iteration++) {
    const sums = clusters.map(() => [0, 0, 0, 0]);
    for (const p of points) {
      let best = 0;
      let bestDistance = Infinity;
      clusters.forEach((cluster, i) => {
        const d = distance2(p, cluster.rgb);
        if (d < bestDistance) {
          bestDistance = d;
          best = i;
        }
      });
      const s = sums[best]!;
      s[0] = s[0]! + p[0]!;
      s[1] = s[1]! + p[1]!;
      s[2] = s[2]! + p[2]!;
      s[3] = s[3]! + 1;
    }
    let moved = 0;
    clusters.forEach((cluster, i) => {
      const s = sums[i]!;
      cluster.count = s[3]!;
      if (s[3]! === 0) return;
      const next: [number, number, number] = [s[0]! / s[3]!, s[1]! / s[3]!, s[2]! / s[3]!];
      moved += distance2(next, cluster.rgb);
      cluster.rgb = next;
    });
    if (moved < 0.5) break;
  }

  return clusters
    .filter((c) => c.count > 0)
    .sort((a, b) => b.count * (1 + chroma(b.rgb)) - a.count * (1 + chroma(a.rgb)))
    .map((c) => toHex(c.rgb));
}

/** The single best colour, or undefined for an empty / fully transparent image. */
export function dominantColor(
  data: ArrayLike<number>,
  width: number,
  height: number,
  options?: ColorExtractOptions,
): string | undefined {
  return dominantColors(data, width, height, options)[0];
}

export const MAX_COLOR_IMAGE_BYTES = 15 * 1024 * 1024;
const ANALYSIS_SIZE = 96;

/** Decodes a local image on a small canvas and returns its swatches. Nothing is uploaded or stored. */
export async function swatchesFromImageFile(file: Blob): Promise<string[]> {
  if (!file.type.startsWith('image/') || file.size > MAX_COLOR_IMAGE_BYTES)
    throw new Error('unsupported image');
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  try {
    const scale = Math.min(1, ANALYSIS_SIZE / Math.max(bitmap.width, bitmap.height));
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('no 2d context');
    context.drawImage(bitmap, 0, 0, width, height);
    return dominantColors(context.getImageData(0, 0, width, height).data, width, height);
  } finally {
    bitmap.close();
  }
}
