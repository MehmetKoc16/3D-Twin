import { describe, expect, it } from 'vitest';
import { sampleBrowColor, sampleIrisColor } from './photoColors';

const W = 400;
const H = 400;

function blank(r: number, g: number, b: number): Uint8ClampedArray {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) data.set([r, g, b, 255], i * 4);
  return data;
}

function disc(data: Uint8ClampedArray, cx: number, cy: number, radius: number, rgb: [number, number, number]): void {
  for (let y = Math.floor(cy - radius); y <= Math.ceil(cy + radius); y++)
    for (let x = Math.floor(cx - radius); x <= Math.ceil(cx + radius); x++)
      if (Math.hypot(x - cx, y - cy) <= radius) data.set([...rgb, 255], (y * W + x) * 4);
}

function landmarks(): Float32Array {
  const lm = new Float32Array(478 * 3);
  const set = (i: number, x: number, y: number): void => {
    lm[i * 3] = x / W;
    lm[i * 3 + 1] = y / H;
  };
  const iris = (first: number, cx: number, cy: number, r: number): void => {
    set(first, cx, cy);
    set(first + 1, cx + r, cy);
    set(first + 2, cx, cy - r);
    set(first + 3, cx - r, cy);
    set(first + 4, cx, cy + r);
  };
  iris(468, 140, 200, 20);
  iris(473, 260, 200, 20);
  [70, 63, 105, 66, 107, 46, 53, 52, 65, 55].forEach((i, k) => set(i, 110 + (k % 5) * 15, 148 + Math.floor(k / 5) * 8));
  [300, 293, 334, 296, 336, 276, 283, 282, 295, 285].forEach((i, k) => set(i, 230 + (k % 5) * 15, 148 + Math.floor(k / 5) * 8));
  set(10, 200, 60); // face extent for the size-relative brow window
  set(152, 200, 340);
  return lm;
}

function close(hex: string | undefined, expected: [number, number, number], tolerance: number): void {
  expect(hex).toMatch(/^#[0-9a-f]{6}$/);
  const n = Number.parseInt(hex!.slice(1), 16);
  [(n >> 16) & 255, (n >> 8) & 255, n & 255].forEach((c, k) => expect(Math.abs(c - expected[k]!)).toBeLessThanOrEqual(tolerance));
}

describe('sampleIrisColor', () => {
  it('returns the iris colour, ignoring pupil and catch light', () => {
    const data = blank(230, 200, 185); // skin
    for (const cx of [140, 260]) {
      disc(data, cx, 200, 20, [40, 110, 160]); // blue iris
      disc(data, cx, 200, 7, [5, 5, 5]); // pupil
      disc(data, cx - 8, 192, 3, [255, 255, 255]); // highlight
    }
    close(sampleIrisColor({ data, width: W, height: H }, landmarks()), [40, 110, 160], 12);
  });

  it('is undefined for missing landmarks and for a degenerate iris', () => {
    const data = blank(200, 200, 200);
    expect(sampleIrisColor({ data, width: W, height: H }, new Float32Array(10))).toBeUndefined();
    expect(sampleIrisColor({ data, width: W, height: H }, new Float32Array(478 * 3))).toBeUndefined();
  });
});

describe('sampleBrowColor', () => {
  it('returns the dark hair colour rather than the skin between the hairs', () => {
    const data = blank(225, 185, 160);
    for (const x0 of [110, 230]) {
      for (const y of [148, 156]) for (let x = x0 - 6; x < x0 + 66; x += 3) disc(data, x, y, 3, [60, 40, 25]); // hair strokes
    }
    close(sampleBrowColor({ data, width: W, height: H }, landmarks()), [60, 40, 25], 20);
  });

  it('is undefined without a full landmark set', () => {
    expect(sampleBrowColor({ data: blank(0, 0, 0), width: W, height: H }, new Float32Array(30))).toBeUndefined();
  });
});
