import { describe, expect, it } from 'vitest';
import { assessQuality, estimateYawDeg, faceHeightFraction, meanLumaInBox, relativeLuma } from './quality';

function makeLandmarks(noseX: number, faceHeight = 0.5): Float32Array {
  const lm = new Float32Array(478 * 3);
  lm[234 * 3] = 0.3;
  lm[454 * 3] = 0.7;
  lm[1 * 3] = noseX;
  lm[10 * 3 + 1] = 0.2;
  lm[152 * 3 + 1] = 0.2 + faceHeight;
  return lm;
}

describe('yaw estimate', () => {
  it('is 0 for a centred nose', () => {
    expect(estimateYawDeg(makeLandmarks(0.5))).toBeCloseTo(0, 5);
  });
  it('is signed and grows with nose offset', () => {
    expect(estimateYawDeg(makeLandmarks(0.55))).toBeGreaterThan(0);
    expect(estimateYawDeg(makeLandmarks(0.4))).toBeLessThan(0);
    expect(Math.abs(estimateYawDeg(makeLandmarks(0.65)))).toBeGreaterThan(
      Math.abs(estimateYawDeg(makeLandmarks(0.55))),
    );
  });
  it('handles degenerate width', () => {
    expect(estimateYawDeg(new Float32Array(478 * 3))).toBe(0);
  });
});

describe('assessQuality', () => {
  const base = { imageWidth: 1000, imageHeight: 1000, faceCount: 1 };
  it('reports no face', () => {
    expect(assessQuality({ ...base, faceCount: 0 })).toEqual(['noFace']);
  });
  it('passes a good frontal face', () => {
    expect(assessQuality({ ...base, landmarks: makeLandmarks(0.5), faceLuma: 0.4 })).toEqual([]);
  });
  it('flags small, non frontal, dark and multiple faces', () => {
    const hints = assessQuality({ ...base, faceCount: 2, landmarks: makeLandmarks(0.62, 0.1), faceLuma: 0.01 });
    expect(hints).toEqual(expect.arrayContaining(['multipleFaces', 'tooSmall', 'notFrontal', 'tooDark']));
  });
  it('measures face height fraction', () => {
    expect(faceHeightFraction(makeLandmarks(0.5, 0.4))).toBeCloseTo(0.4, 5);
  });
});

describe('luma', () => {
  it('orders black < grey < white', () => {
    expect(relativeLuma(0, 0, 0)).toBe(0);
    expect(relativeLuma(255, 255, 255)).toBeCloseTo(1, 5);
    expect(relativeLuma(128, 128, 128)).toBeGreaterThan(0.2);
  });
  it('averages a box', () => {
    const data = new Uint8ClampedArray(4 * 4 * 4).fill(255);
    const box = { minX: 0, minY: 0, maxX: 1, maxY: 1 };
    expect(meanLumaInBox({ data, width: 4, height: 4 }, box, 1)).toBeCloseTo(1, 5);
  });
});
