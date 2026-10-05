import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  CanvasTexture,
  DataTexture,
  MeshStandardMaterial,
  NoColorSpace,
  SRGBColorSpace,
  Vector2,
} from 'three';
import { twinMaterial } from './twinModel';
import {
  openingBand,
  openingEdges,
  openingTexels,
  repaintOpeningPixels,
  TwinOpeningRepair,
  type GarmentSurface,
} from './twinOpeningRepair';
import type { TexturePixels } from './twinSkinTone';

// Two triangles with a duplicated UV seam on their shared diagonal.
const surface: GarmentSurface = {
  positions: Float32Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]),
  index: [0, 1, 2, 3, 4, 5],
};
const uv = Float32Array.from([0, 0, 1, 0, 0, 1]);
const index = [0, 1, 2];
function albedo(): TexturePixels {
  return {
    width: 4,
    height: 4,
    data: Uint8ClampedArray.from({ length: 64 }, (_, i) => (i % 4 === 3 ? 255 : 70)),
  };
}

afterEach(() => vi.unstubAllGlobals());

describe('opening texture repair', () => {
  it('welds seams, excludes covered vertices and distant/interior scan vertices, and softly fades the outside band', () => {
    expect(openingEdges(surface)).toHaveLength(4);
    const scan = Float32Array.from([
      0.5, -0.01, 0, 0.5, -0.025, 0, 0.5, -0.05, 0, 0.5, 0.5, 0, 0.5, -0.01, 0, 1.01, 0.5, 0,
    ]);
    const band = openingBand(scan, Uint8Array.from([0, 0, 0, 0, 1, 0]), [surface]);
    expect(band[0]).toBe(1);
    expect(band[1]).toBeGreaterThan(0);
    expect(band[1]).toBeLessThan(1);
    expect([...band.slice(2, 5)]).toEqual([0, 0, 0]);
    expect(band[5]).toBe(1); // same algorithm for a sleeve end or waistband
    expect(openingBand(scan, new Uint8Array(6), [])).toEqual(new Float32Array(6));
  });

  it('handles closed garments with no openings and long opening segments', () => {
    const closed = {
      positions: Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
      index: [0, 1, 2, 0, 3, 1, 1, 3, 2, 2, 3, 0],
    };
    expect(openingEdges(closed)).toEqual([]);
    expect(openingBand(Float32Array.from([0.5, -0.01, 0]), new Uint8Array(1), [surface])[0]).toBe(
      1,
    );
  });

  it('selects only the UV triangle, interpolates a soft mask, and ignores degenerate/unselected triangles', () => {
    const selected = openingTexels(uv, index, Float32Array.from([1, 0, 0]), 4, 4);
    expect(selected.get(0)).toBeCloseTo(0.75);
    expect(selected.has(15)).toBe(false);
    expect(selected.get(1)).toBeLessThan(selected.get(0)!);
    expect(openingTexels(uv, index, new Float32Array(3), 4, 4).size).toBe(0);
    expect(openingTexels(new Float32Array(6), index, Float32Array.from([1, 1, 1]), 4, 4).size).toBe(
      0,
    );
  });

  it('repaints grey clothing, preserves skin and alpha, and never modifies the source pixels', () => {
    const original = albedo();
    original.data.set([201, 154, 126, 255], 4);
    original.data.set([70, 70, 70, 0], 8);
    const saved = original.data.slice();
    const { data, painted } = repaintOpeningPixels(
      original,
      new Map([
        [0, 1],
        [1, 1],
        [2, 1],
        [3, 0.5],
      ]),
      '#c99a7e',
    );
    expect([...data.slice(0, 4)]).toEqual([201, 154, 126, 255]);
    expect([...data.slice(4, 12)]).toEqual([...saved.slice(4, 12)]);
    expect(data[12]).toBeGreaterThan(70);
    expect(data[12]).toBeLessThan(201);
    expect(painted).toBe(2);
    expect(original.data).toEqual(saved);
    expect(repaintOpeningPixels(original, new Map(), '#c99a7e').data).toEqual(saved);
  });

  it('detects a grey collar of similar lightness while retaining small variations in skin colour', () => {
    const original = albedo();
    original.data.set([153, 153, 153, 255], 0);
    original.data.set([196, 150, 122, 255], 4);
    const result = repaintOpeningPixels(
      original,
      new Map([
        [0, 1],
        [1, 1],
      ]),
      '#c99a7e',
    );
    expect(result.data[0]).toBeGreaterThan(195);
    expect(result.data[2]).toBeLessThan(130);
    expect([...result.data.slice(4, 8)]).toEqual([196, 150, 122, 255]);
    expect(result.painted).toBe(1);
  });

  it('uses a disposable sRGB CanvasTexture copy, updates from the original, and restores the identical map on take-off', () => {
    const written: Uint8ClampedArray[] = [];
    vi.stubGlobal('document', {
      createElement: () => ({
        width: 0,
        height: 0,
        getContext: () => ({
          createImageData: (width: number, height: number) => ({
            data: new Uint8ClampedArray(width * height * 4),
          }),
          putImageData: (image: { data: Uint8ClampedArray }) => written.push(image.data.slice()),
        }),
      }),
    });
    const pixels = albedo();
    const original = new DataTexture(new Uint8Array(pixels.data), pixels.width, pixels.height);
    original.flipY = false;
    const normalMap = new DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1);
    const material = twinMaterial(
      new MeshStandardMaterial({
        map: original,
        normalMap,
        normalScale: new Vector2(0.35, -0.6),
      }),
    );
    const normalDisposed = vi.fn();
    normalMap.addEventListener('dispose', normalDisposed);
    const expectNormals = (): void => {
      expect(material.normalMap).toBe(normalMap);
      expect(material.normalScale.toArray()).toEqual([0.35, -0.6]);
      expect(normalMap.colorSpace).toBe(NoColorSpace);
      expect(normalDisposed).not.toHaveBeenCalled();
    };
    const repair = new TwinOpeningRepair(material, uv, '#c99a7e');
    const scan = Float32Array.from([0.2, -0.005, 0, 0.4, -0.005, 0, 0.2, -0.01, 0]);
    repair.update(scan, index, new Uint8Array(3), [surface]);
    expect(repair.painted).toBeGreaterThan(0);
    expect(repair.originalTexture).toBe(false);
    expect(material.map).toBeInstanceOf(CanvasTexture);
    expect(material.map!.flipY).toBe(false);
    expect(material.map!.colorSpace).toBe(SRGBColorSpace);
    expectNormals();
    const disposed = vi.fn();
    material.map!.addEventListener('dispose', disposed);
    const sourceDisposed = vi.fn();
    original.addEventListener('dispose', sourceDisposed);
    repair.update(scan, index, new Uint8Array(3), [surface]);
    expect(disposed).toHaveBeenCalledOnce();
    expect(written[1]).toEqual(written[0]);
    expectNormals();
    repair.update(scan, index, new Uint8Array(3), []);
    expect(material.map).toBe(original);
    expect(repair.originalTexture).toBe(true);
    expect(repair.painted).toBe(0);
    expect(original.image.data).toEqual(new Uint8Array(pixels.data));
    expect(sourceDisposed).not.toHaveBeenCalled();
    expectNormals();
  });
});
