import { at } from './util';

/** Photo quality heuristics on MediaPipe landmarks (normalized coordinates, 478 * 3 floats). */

export type QualityHint = 'noFace' | 'multipleFaces' | 'tooSmall' | 'notFrontal' | 'tooDark';

export const LANDMARK_COUNT = 478;
const NOSE_TIP = 1;
const LEFT_EDGE = 234;
const RIGHT_EDGE = 454;
const FOREHEAD_TOP = 10;
const CHIN = 152;

export const MAX_YAW_DEG = 20;
export const MIN_FACE_HEIGHT_FRACTION = 0.22;
export const MIN_FACE_HEIGHT_PX = 160;
export const MIN_LUMA = 0.05;

export interface FaceBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** Bounding box of all landmarks in normalized coordinates. */
export function landmarkBox(landmarks: ArrayLike<number>): FaceBox {
  const count = Math.floor(landmarks.length / 3);
  const box: FaceBox = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  for (let i = 0; i < count; i += 1) {
    const x = at(landmarks, i * 3);
    const y = at(landmarks, i * 3 + 1);
    if (x < box.minX) box.minX = x;
    if (x > box.maxX) box.maxX = x;
    if (y < box.minY) box.minY = y;
    if (y > box.maxY) box.maxY = y;
  }
  return box;
}

/** Approximate head yaw in degrees from the nose position between the two face edges (0 = frontal). */
export function estimateYawDeg(landmarks: ArrayLike<number>): number {
  const nose = at(landmarks, NOSE_TIP * 3);
  const left = at(landmarks, LEFT_EDGE * 3);
  const right = at(landmarks, RIGHT_EDGE * 3);
  const width = right - left;
  if (!Number.isFinite(width) || Math.abs(width) < 1e-6) return 0;
  const offset = ((nose - left) / width - 0.5) * 2; // -1..1
  const clamped = Math.max(-1, Math.min(1, offset));
  return (Math.asin(clamped) * 180) / Math.PI;
}

/** Face height (forehead top to chin) as a fraction of the image height. */
export function faceHeightFraction(landmarks: ArrayLike<number>): number {
  return Math.abs(at(landmarks, CHIN * 3 + 1) - at(landmarks, FOREHEAD_TOP * 3 + 1));
}

/** Relative luminance (0..1) of an sRGB colour with 0..255 channels. */
export function relativeLuma(r: number, g: number, b: number): number {
  const lin = (c: number) => {
    const v = c / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

export interface QualityInput {
  landmarks?: ArrayLike<number>;
  faceCount: number;
  imageWidth: number;
  imageHeight: number;
  /** Mean relative luminance (0..1, linear) of the face region, if measured. */
  faceLuma?: number;
}

export function assessQuality(input: QualityInput): QualityHint[] {
  const hints: QualityHint[] = [];
  if (input.faceCount === 0 || !input.landmarks) return ['noFace'];
  if (input.faceCount > 1) hints.push('multipleFaces');
  const heightFraction = faceHeightFraction(input.landmarks);
  if (heightFraction < MIN_FACE_HEIGHT_FRACTION || heightFraction * input.imageHeight < MIN_FACE_HEIGHT_PX) {
    hints.push('tooSmall');
  }
  if (Math.abs(estimateYawDeg(input.landmarks)) > MAX_YAW_DEG) hints.push('notFrontal');
  if (input.faceLuma !== undefined && input.faceLuma < MIN_LUMA) hints.push('tooDark');
  return hints;
}

/** Mean relative luminance of the pixels inside a normalized box of an RGBA image. */
export function meanLumaInBox(
  image: { data: ArrayLike<number>; width: number; height: number },
  box: FaceBox,
  step = 4,
): number {
  const x0 = Math.max(0, Math.floor(box.minX * image.width));
  const x1 = Math.min(image.width, Math.ceil(box.maxX * image.width));
  const y0 = Math.max(0, Math.floor(box.minY * image.height));
  const y1 = Math.min(image.height, Math.ceil(box.maxY * image.height));
  let sum = 0;
  let count = 0;
  for (let y = y0; y < y1; y += step) {
    for (let x = x0; x < x1; x += step) {
      const o = (y * image.width + x) * 4;
      sum += relativeLuma(at(image.data, o), at(image.data, o + 1), at(image.data, o + 2));
      count += 1;
    }
  }
  return count ? sum / count : 0;
}
