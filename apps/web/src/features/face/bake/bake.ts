import type { FaceMapDef } from '@dt/avatar-core';
import { assembleBakeGeometry } from './assemble';
import { buildMask, sortByAngle, type Point } from './mask';
import { processFaceCrop } from './process';
import { hexToLinearRgb, linearRgbToHex, sampleLandmarkColors, type PixelSource } from './skinTone';
import { warpPhotoToUv } from './warp';
import { landmarkBox } from '../quality';

export const BAKE_SIZE = 2048;

/** MediaPipe inner-lip contour in ring order (the mouth slit is a hole of the head UV island). */
export const INNER_LIP_RING: readonly number[] = [
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
];

export interface BakeOptions {
  size?: number;
  /** Delighting strength 0..1 (default 0.6). */
  delightStrength?: number;
  /** Feather width as a fraction of the face height in UV pixels (default 0.04). */
  featherFraction?: number;
}

export interface BakeResult {
  overlayCanvas: HTMLCanvasElement;
  skinToneHex: string;
  uvBounds: { min: [number, number]; max: [number, number] };
}

function uvLookup(faceMap: FaceMapDef): Map<number, [number, number]> {
  const map = new Map<number, [number, number]>();
  for (const lm of faceMap.landmarks) map.set(lm.index, lm.uv);
  return map;
}

function regionPoints(indices: number[], uvs: Map<number, [number, number]>, size: number): Point[] {
  const points: Point[] = [];
  for (const index of indices) {
    const uv = uvs.get(index);
    if (uv) points.push([uv[0] * size, uv[1] * size]);
  }
  return points;
}

/** Reads the photo pixels at a reduced size (for skin sampling only). */
export function readPhotoPixels(image: ImageBitmap, maxSide = 1024): ImageData {
  const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(image.width * scale));
  canvas.height = Math.max(1, Math.round(image.height * scale));
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('2D canvas is not available');
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  return context.getImageData(0, 0, canvas.width, canvas.height);
}

/** Estimated skin tone (hex) from the cheek landmark neighbourhoods of the photo. */
export function estimateSkinTone(
  pixels: PixelSource,
  landmarks: ArrayLike<number>,
  faceMap: FaceMapDef,
): string {
  const box = landmarkBox(landmarks);
  const facePx = (box.maxY - box.minY) * pixels.height;
  const cheeks = [...faceMap.regions.leftCheek, ...faceMap.regions.rightCheek];
  const rgb = sampleLandmarkColors(pixels, landmarks, cheeks, Math.max(1, facePx * 0.01));
  return linearRgbToHex(rgb ?? [0.4, 0.25, 0.2]);
}

export async function bakeFace(
  image: ImageBitmap,
  landmarks: ArrayLike<number>,
  faceMap: FaceMapDef,
  options: BakeOptions = {},
): Promise<BakeResult> {
  const size = options.size ?? BAKE_SIZE;
  const uvs = uvLookup(faceMap);
  const skinToneHex = estimateSkinTone(readPhotoPixels(image), landmarks, faceMap);

  const geometry = assembleBakeGeometry(faceMap, landmarks, size, size);
  const warped = warpPhotoToUv(image, geometry, size);

  // Crop to the oval bounds (plus a margin) so per-pixel work stays small.
  const oval = regionPoints(faceMap.faceOval, uvs, size);
  if (oval.length < 3) throw new Error('face oval is not bound in the face map');
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of oval) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const faceHeight = maxY - minY;
  const feather = Math.max(2, faceHeight * (options.featherFraction ?? 0.04));
  const margin = Math.ceil(feather * 2);
  const x0 = Math.max(0, Math.floor(minX - margin));
  const y0 = Math.max(0, Math.floor(minY - margin));
  const x1 = Math.min(size, Math.ceil(maxX + margin));
  const y1 = Math.min(size, Math.ceil(maxY + margin));
  const width = x1 - x0;
  const height = y1 - y0;
  if (width <= 0 || height <= 0) throw new Error('empty face region');

  const localOval: Point[] = oval.map(([x, y]) => [x - x0, y - y0]);
  const foreheadPts = regionPoints(faceMap.regions.forehead, uvs, size).map(([x, y]): Point => [x - x0, y - y0]);
  const eyeGroups = [faceMap.regions.leftEye, faceMap.regions.rightEye].map((ids) =>
    regionPoints(ids, uvs, size).map(([x, y]): Point => [x - x0, y - y0]),
  );
  const eyePts = eyeGroups.flat();
  // The head UV island is rotated: derive "toward the top of the head" from the eyes -> forehead direction.
  let forehead: { origin: Point; axis: Point; topT: number; browT: number } | undefined;
  if (foreheadPts.length && eyePts.length) {
    const mean = (pts: readonly Point[]): Point => [
      pts.reduce((sum, p) => sum + p[0], 0) / pts.length,
      pts.reduce((sum, p) => sum + p[1], 0) / pts.length,
    ];
    const origin = mean(eyePts);
    const top = mean(foreheadPts);
    const length = Math.hypot(top[0] - origin[0], top[1] - origin[1]);
    if (length > 1) {
      const axis: Point = [(top[0] - origin[0]) / length, (top[1] - origin[1]) / length];
      const topT = Math.max(...foreheadPts.map((p) => (p[0] - origin[0]) * axis[0] + (p[1] - origin[1]) * axis[1]));
      if (topT > 1) forehead = { origin, axis, topT, browT: 0.4 * topT };
    }
  }
  // Eye openings and the mouth slit are holes of the head UV island: never paint them from the photo.
  const holeGrow = Math.max(1.5, faceHeight * 0.006);
  const holeFeather = Math.max(2, faceHeight * 0.012);
  const holes: { polygon: readonly Point[]; grow: number; feather: number }[] = [];
  for (const group of eyeGroups) if (group.length >= 3) holes.push({ polygon: sortByAngle(group), grow: holeGrow, feather: holeFeather });
  const slit = INNER_LIP_RING.map((index) => uvs.get(index)).filter((uv): uv is [number, number] => uv !== undefined);
  if (slit.length >= 3) {
    holes.push({
      polygon: slit.map(([u, v]): Point => [u * size - x0, v * size - y0]),
      grow: holeGrow,
      feather: holeFeather,
    });
  }

  const mask = buildMask({ width, height, polygon: localOval, feather, forehead, holes });
  const context = warped.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('2D canvas is not available');
  const crop = context.getImageData(x0, y0, width, height);
  processFaceCrop({
    rgba: crop.data,
    width,
    height,
    maskAlpha: mask.alpha,
    inside: mask.inside,
    feather,
    skinLinear: hexToLinearRgb(skinToneHex),
    strength: options.delightStrength ?? 0.6,
    blurRadius: Math.max(4, Math.round(width / 6)),
  });

  const overlayCanvas = document.createElement('canvas');
  overlayCanvas.width = size;
  overlayCanvas.height = size;
  const overlayContext = overlayCanvas.getContext('2d');
  if (!overlayContext) throw new Error('2D canvas is not available');
  overlayContext.putImageData(crop, x0, y0);

  return {
    overlayCanvas,
    skinToneHex,
    uvBounds: { min: [x0 / size, y0 / size], max: [x1 / size, y1 / size] },
  };
}
