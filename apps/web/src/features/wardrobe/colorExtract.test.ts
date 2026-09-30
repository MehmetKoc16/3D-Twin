import { describe, expect, it } from 'vitest';
import { dominantColor, dominantColors } from './colorExtract';

function image(
  width: number,
  height: number,
  pixel: (x: number, y: number) => [number, number, number, number],
): Uint8ClampedArray {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) data.set(pixel(x, y), (y * width + x) * 4);
  }
  return data;
}

const hexToRgb = (hex: string): number[] => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

function expectNear(hex: string | undefined, rgb: number[], tolerance = 12): void {
  expect(hex).toBeDefined();
  const got = hexToRgb(hex!);
  rgb.forEach((c, i) => expect(Math.abs(got[i]! - c)).toBeLessThanOrEqual(tolerance));
}

describe('dominantColor', () => {
  it('returns the colour of a flat image', () => {
    const data = image(20, 20, () => [200, 30, 40, 255]);
    expectNear(dominantColor(data, 20, 20), [200, 30, 40], 2);
  });

  it('ignores a white studio background around the garment', () => {
    // 40x40 white photo with a 20x20 navy square in the middle
    const data = image(40, 40, (x, y) =>
      x >= 10 && x < 30 && y >= 10 && y < 30 ? [20, 40, 120, 255] : [255, 255, 255, 255],
    );
    expectNear(dominantColor(data, 40, 40), [20, 40, 120]);
  });

  it('ignores a grey background just as well', () => {
    const data = image(40, 40, (x, y) =>
      x >= 12 && x < 28 && y >= 8 && y < 32 ? [180, 40, 30, 255] : [190, 190, 190, 255],
    );
    expectNear(dominantColor(data, 40, 40), [180, 40, 30]);
  });

  it('separates two garment colours into swatches, the larger one first', () => {
    const data = image(60, 60, (x, y) => {
      if (x < 4 || x >= 56 || y < 4 || y >= 56) return [255, 255, 255, 255];
      return x < 40 ? [30, 140, 60, 255] : [220, 200, 40, 255];
    });
    const swatches = dominantColors(data, 60, 60, { k: 3 });
    expect(swatches.length).toBeGreaterThanOrEqual(2);
    expectNear(swatches[0], [30, 140, 60]);
    expect(swatches.some((hex) => Math.abs(hexToRgb(hex)[0]! - 220) < 15)).toBe(true);
  });

  it('is deterministic', () => {
    const data = image(30, 30, (x, y) => [(x * 8) % 256, (y * 8) % 256, (x * y) % 256, 255]);
    expect(dominantColors(data, 30, 30)).toEqual(dominantColors(data, 30, 30));
  });

  it('handles a photo that is only background and fully transparent images', () => {
    const white = image(10, 10, () => [255, 255, 255, 255]);
    expectNear(dominantColor(white, 10, 10), [255, 255, 255], 2);
    const clear = image(10, 10, () => [10, 20, 30, 0]);
    expect(dominantColor(clear, 10, 10)).toBeUndefined();
    expect(dominantColors(new Uint8ClampedArray(0), 0, 0)).toEqual([]);
  });
});
