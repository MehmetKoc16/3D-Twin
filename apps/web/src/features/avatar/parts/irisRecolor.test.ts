import { describe, expect, it } from 'vitest';
import {
  FADE_END,
  irisReferenceLuma,
  luma,
  meanCoveredLinear,
  parseHex,
  recolorIris,
  type IrisRegion,
} from './irisRecolor';

const SIZE = 64;
const IRIS: IrisRegion = { center: [0.5, 0.5], radius: 0.25 }; // 16 px radius

/** Synthetic eye texture: white sclera, grey iris ring with a radial pattern, a dark pupil at the centre. */
function eyeTexture(): Uint8ClampedArray {
  const data = new Uint8ClampedArray(SIZE * SIZE * 4);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const d = Math.hypot(x + 0.5 - 32, y + 0.5 - 32) / 16;
      const angle = Math.atan2(y - 32, x - 32);
      let v = 235; // sclera
      if (d <= 1.02) v = 120 + 40 * Math.sin(angle * 8); // iris fibres
      if (d <= 0.3) v = 15; // pupil
      const o = (y * SIZE + x) * 4;
      data[o] = v;
      data[o + 1] = v;
      data[o + 2] = v;
      data[o + 3] = 255;
    }
  }
  return data;
}

function pixel(data: Uint8ClampedArray, x: number, y: number): [number, number, number] {
  const o = (y * SIZE + x) * 4;
  return [data[o]!, data[o + 1]!, data[o + 2]!];
}

describe('parseHex', () => {
  it('reads sRGB bytes', () => {
    expect(parseHex('#3d6fa8')).toEqual([0x3d, 0x6f, 0xa8]);
  });
});

describe('recolorIris', () => {
  const source = eyeTexture();
  const out = new Uint8ClampedArray(source.length);
  recolorIris(source, out, SIZE, SIZE, IRIS, '#3d6fa8');

  it('tints the iris towards the chosen colour (blue channel dominates)', () => {
    for (let a = 0; a < 16; a++) {
      const angle = (a / 16) * Math.PI * 2;
      const [r, , b] = pixel(out, Math.round(32 + Math.cos(angle) * 11), Math.round(32 + Math.sin(angle) * 11));
      expect(b).toBeGreaterThan(r + 30);
    }
  });

  it('keeps the average iris colour close to the chosen colour', () => {
    const sum = [0, 0, 0];
    let n = 0;
    for (let y = 0; y < SIZE; y++)
      for (let x = 0; x < SIZE; x++) {
        const d = Math.hypot(x + 0.5 - 32, y + 0.5 - 32) / 16;
        if (d < 0.4 || d > 0.9) continue;
        pixel(out, x, y).forEach((c, k) => (sum[k] = sum[k]! + c));
        n++;
      }
    expect(Math.abs(sum[0]! / n - 0x3d)).toBeLessThan(14);
    expect(Math.abs(sum[1]! / n - 0x6f)).toBeLessThan(14);
    expect(Math.abs(sum[2]! / n - 0xa8)).toBeLessThan(14);
  });

  it('keeps the luminance detail of the grey iris', () => {
    let lo = Infinity;
    let hi = -Infinity;
    for (let a = 0; a < 64; a++) {
      const angle = (a / 64) * Math.PI * 2;
      const l = luma(...pixel(out, Math.round(32 + Math.cos(angle) * 11), Math.round(32 + Math.sin(angle) * 11)));
      lo = Math.min(lo, l);
      hi = Math.max(hi, l);
    }
    expect(hi - lo).toBeGreaterThan(0.08);
  });

  it('keeps the pupil dark', () => {
    expect(luma(...pixel(out, 32, 32))).toBeLessThan(0.1);
  });

  it('leaves the sclera and everything beyond the fade radius untouched', () => {
    const beyond = 32 + Math.ceil(16 * FADE_END) + 1;
    for (const [x, y] of [
      [2, 2],
      [60, 4],
      [32, beyond],
      [5, 32],
      [58, 32],
    ] as const) {
      expect(pixel(out, x, y)).toEqual(pixel(source, x, y));
    }
  });

  it('fades smoothly at the rim instead of cutting', () => {
    const deltas: number[] = [];
    for (let x = 32 + 12; x <= 32 + Math.ceil(16 * FADE_END) + 1; x++) {
      const a = pixel(out, x, 32);
      const b = pixel(source, x, 32);
      deltas.push(Math.abs(a[2] - b[2]) + Math.abs(a[0] - b[0]));
    }
    expect(deltas[deltas.length - 1]).toBe(0);
    // the colour change shrinks towards the rim (never grows by more than rounding noise)
    for (let i = 1; i < deltas.length; i++) expect(deltas[i]!).toBeLessThanOrEqual(deltas[i - 1]! + 2);
  });

  it('does not touch the alpha channel', () => {
    for (let i = 3; i < out.length; i += 4) expect(out[i]).toBe(255);
  });
});

describe('irisReferenceLuma', () => {
  it('measures the iris ring without the pupil', () => {
    const value = irisReferenceLuma(eyeTexture(), SIZE, SIZE, IRIS);
    expect(value).toBeGreaterThan(0.4);
    expect(value).toBeLessThan(0.55);
  });
});

describe('meanCoveredLinear', () => {
  it('averages the linear value of texels above the cutoff only', () => {
    // one covered mid-grey texel (128/255 sRGB = 0.216 linear), one transparent white texel
    const pixels = Uint8ClampedArray.from([128, 128, 128, 255, 255, 255, 255, 10]);
    expect(meanCoveredLinear(pixels, 0.4)).toBeCloseTo(0.216, 2);
  });
});
