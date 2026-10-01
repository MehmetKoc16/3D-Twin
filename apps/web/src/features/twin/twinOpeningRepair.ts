import { CanvasTexture, SRGBColorSpace, type MeshStandardMaterial, type Texture } from 'three';
import {
  colorDistance,
  hexBytes,
  perceptualColor,
  readTexturePixels,
  textureUVs,
  type TexturePixels,
} from './twinSkinTone';

export const OPENING_BAND_M = 0.04;

export interface GarmentSurface {
  positions: Float32Array;
  index: ArrayLike<number>;
}

/** Weld UV seams before counting boundary edges: a seam inside the fabric is not an opening. */
export function openingEdges(surface: GarmentSurface): [number, number][] {
  const { positions, index } = surface;
  const weld = new Map<string, number>();
  const representatives = new Uint32Array(positions.length / 3);
  for (let v = 0; v < representatives.length; v++) {
    const key = [0, 1, 2].map((axis) => Math.round(positions[v * 3 + axis]! * 100000)).join(',');
    if (!weld.has(key)) weld.set(key, v);
    representatives[v] = weld.get(key)!;
  }
  const edges = new Map<string, { a: number; b: number; count: number }>();
  for (let t = 0; t < index.length; t += 3)
    for (let j = 0; j < 3; j++) {
      const a = representatives[index[t + j]!]!,
        b = representatives[index[t + ((j + 1) % 3)]!]!;
      if (a === b) continue;
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      const edge = edges.get(key);
      if (edge) edge.count++;
      else edges.set(key, { a, b, count: 1 });
    }
  return [...edges.values()].filter((edge) => edge.count === 1).map(({ a, b }) => [a, b]);
}

/** Exact segment distance, with a grid built only around garment openings, never the whole scan. */
export function openingBand(
  positions: Float32Array,
  covered: Uint8Array,
  surfaces: readonly GarmentSurface[],
): Float32Array {
  const segments: number[][] = [];
  const grid = new Map<string, Set<number>>();
  const cell = (n: number): number => Math.floor(n / OPENING_BAND_M);
  const key = (x: number, y: number, z: number): string => `${x},${y},${z}`;
  for (const surface of surfaces)
    for (const [a, b] of openingEdges(surface)) {
      const segment = [0, 1, 2]
        .map((axis) => surface.positions[a * 3 + axis]!)
        .concat([0, 1, 2].map((axis) => surface.positions[b * 3 + axis]!));
      const id = segments.length;
      segments.push(segment);
      const length = Math.hypot(
        segment[3]! - segment[0]!,
        segment[4]! - segment[1]!,
        segment[5]! - segment[2]!,
      );
      const steps = Math.max(1, Math.ceil(length / (OPENING_BAND_M / 2)));
      for (let step = 0; step <= steps; step++) {
        const point = [0, 1, 2].map((axis) =>
          cell(segment[axis]! + ((segment[axis + 3]! - segment[axis]!) * step) / steps),
        );
        for (let dx = -1; dx <= 1; dx++)
          for (let dy = -1; dy <= 1; dy++)
            for (let dz = -1; dz <= 1; dz++) {
              const k = key(point[0]! + dx, point[1]! + dy, point[2]! + dz);
              if (!grid.has(k)) grid.set(k, new Set());
              grid.get(k)!.add(id);
            }
      }
    }
  const band = new Float32Array(covered.length);
  for (let v = 0; v < covered.length; v++) {
    if (covered[v]) continue;
    const point = [positions[v * 3]!, positions[v * 3 + 1]!, positions[v * 3 + 2]!];
    let distance = OPENING_BAND_M;
    for (const id of grid.get(key(cell(point[0]!), cell(point[1]!), cell(point[2]!))) ?? []) {
      const segment = segments[id]!;
      const delta = [0, 1, 2].map((axis) => segment[axis + 3]! - segment[axis]!);
      const length2 = delta.reduce((sum, value) => sum + value * value, 0);
      const dot = delta.reduce(
        (sum, value, axis) => sum + (point[axis]! - segment[axis]!) * value,
        0,
      );
      const t = length2 ? Math.max(0, Math.min(1, dot / length2)) : 0;
      distance = Math.min(
        distance,
        Math.hypot(...point.map((value, axis) => value - segment[axis]! - t * delta[axis]!)),
      );
    }
    // Full paint in the first 15 mm, then a smooth fade to zero at 40 mm.
    const t = Math.max(0, Math.min(1, (OPENING_BAND_M - distance) / (OPENING_BAND_M - 0.015)));
    band[v] = t * t * (3 - 2 * t);
  }
  return band;
}

/** Rasterize selected scan triangles only: atlas padding and unrelated UV islands stay untouched. */
export function openingTexels(
  uv: Float32Array,
  index: ArrayLike<number>,
  band: Float32Array,
  width: number,
  height: number,
): Map<number, number> {
  const selected = new Map<number, number>();
  for (let t = 0; t < index.length; t += 3) {
    const vertices = [index[t]!, index[t + 1]!, index[t + 2]!];
    if (!vertices.some((v) => band[v]! > 0)) continue;
    const x = vertices.map((v) => uv[v * 2]! * width);
    const y = vertices.map((v) => uv[v * 2 + 1]! * height);
    const denominator = (y[1]! - y[2]!) * (x[0]! - x[2]!) + (x[2]! - x[1]!) * (y[0]! - y[2]!);
    if (!Number.isFinite(denominator) || Math.abs(denominator) < 1e-8) continue;
    const minX = Math.max(0, Math.floor(Math.min(...x))),
      maxX = Math.min(width - 1, Math.ceil(Math.max(...x)));
    const minY = Math.max(0, Math.floor(Math.min(...y))),
      maxY = Math.min(height - 1, Math.ceil(Math.max(...y)));
    for (let py = minY; py <= maxY; py++)
      for (let px = minX; px <= maxX; px++) {
        const a =
          ((y[1]! - y[2]!) * (px + 0.5 - x[2]!) + (x[2]! - x[1]!) * (py + 0.5 - y[2]!)) /
          denominator;
        const b =
          ((y[2]! - y[0]!) * (px + 0.5 - x[2]!) + (x[0]! - x[2]!) * (py + 0.5 - y[2]!)) /
          denominator;
        const c = 1 - a - b;
        if (Math.min(a, b, c) < -1e-6) continue;
        const strength =
          a * band[vertices[0]!]! + b * band[vertices[1]!]! + c * band[vertices[2]!]!;
        if (strength <= 0) continue;
        const pixel = py * width + px;
        selected.set(pixel, Math.max(strength, selected.get(pixel) ?? 0));
      }
  }
  return selected;
}

/** Always copy the original albedo; garment changes cannot accumulate retouching. */
export function repaintOpeningPixels(
  original: TexturePixels,
  texels: ReadonlyMap<number, number>,
  skinToneHex: string,
): { data: Uint8ClampedArray; painted: number } {
  const data = original.data.slice();
  const skin = hexBytes(skinToneHex);
  const skinLab = perceptualColor(skin);
  let painted = 0;
  for (const [pixel, strength] of texels) {
    const offset = pixel * 4;
    if (data[offset + 3]! < 128) continue;
    const rgb = [data[offset]!, data[offset + 1]!, data[offset + 2]!];
    const distance = colorDistance(perceptualColor(rgb), skinLab);
    // Preserve small skin variation; even similar-lightness grey fabric differs in Oklab chroma.
    const t = Math.max(0, Math.min(1, (distance - 0.04) / 0.04));
    const blend = Math.max(0, Math.min(1, strength)) * t * t * (3 - 2 * t);
    if (blend <= 0) continue;
    for (let channel = 0; channel < 3; channel++)
      data[offset + channel] = rgb[channel]! * (1 - blend) + skin[channel]! * blend;
    painted++;
  }
  return { data, painted };
}

/** Owns the temporary texture only; the model retains its original map for exact restoration/disposal. */
export class TwinOpeningRepair {
  private readonly original: Texture | null;
  private readonly pixels: TexturePixels | null;
  private readonly uv: Float32Array | null;
  private copy: CanvasTexture | null = null;
  painted = 0;

  constructor(
    private readonly material: MeshStandardMaterial,
    uv: Float32Array | null,
    private readonly tone: string,
  ) {
    this.original = material.map;
    this.pixels = uv ? readTexturePixels(this.original) : null;
    this.uv = textureUVs(uv, this.original);
  }

  get originalTexture(): boolean {
    return this.material.map === this.original;
  }

  update(
    positions: Float32Array,
    index: ArrayLike<number>,
    covered: Uint8Array,
    surfaces: readonly GarmentSurface[],
  ): void {
    this.restore();
    if (!surfaces.length || !this.pixels || !this.uv || typeof document === 'undefined') return;
    const band = openingBand(positions, covered, surfaces);
    const texels = openingTexels(this.uv, index, band, this.pixels.width, this.pixels.height);
    if (!texels.size) return;
    const result = repaintOpeningPixels(this.pixels, texels, this.tone);
    if (!result.painted) return;
    const canvas = document.createElement('canvas');
    canvas.width = this.pixels.width;
    canvas.height = this.pixels.height;
    const context = canvas.getContext('2d');
    if (!context) return;
    const image = context.createImageData(canvas.width, canvas.height);
    image.data.set(result.data);
    context.putImageData(image, 0, 0);
    this.copy = new CanvasTexture(canvas);
    this.copy.flipY = false;
    this.copy.colorSpace = SRGBColorSpace;
    // The texel set uses transformed UVs; keep the same transform/sampler on the replacement map.
    if (this.original) {
      this.copy.channel = this.original.channel;
      this.copy.wrapS = this.original.wrapS;
      this.copy.wrapT = this.original.wrapT;
      this.copy.magFilter = this.original.magFilter;
      this.copy.minFilter = this.original.minFilter;
      this.copy.anisotropy = this.original.anisotropy;
      this.copy.matrixAutoUpdate = false;
      this.copy.matrix.copy(this.original.matrix);
    }
    this.material.map = this.copy;
    this.material.needsUpdate = true;
    this.painted = result.painted;
  }

  restore(): void {
    if (this.copy) {
      this.material.map = this.original;
      this.material.needsUpdate = true;
      this.copy.dispose();
      this.copy = null;
    }
    this.painted = 0;
  }
}
