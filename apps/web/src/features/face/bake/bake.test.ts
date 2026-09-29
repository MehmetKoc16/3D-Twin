import { describe, expect, it } from 'vitest';
import { assembleBakeGeometry } from './assemble';
import { delightGain, maskedBlur } from './delight';
import { buildMask, featherAlpha, foreheadFade, insideDistance, smoothstep, sortByAngle } from './mask';
import { makeMockFaceMap } from './mockFaceMap';
import { processFaceCrop } from './process';
import { hexToLinearRgb, linearRgbToHex, median, medianLinearRgb, sampleLandmarkColors } from './skinTone';

describe('feather mask math', () => {
  it('smoothstep is monotonic 0..1', () => {
    expect(smoothstep(0, 10, -1)).toBe(0);
    expect(smoothstep(0, 10, 11)).toBe(1);
    expect(smoothstep(0, 10, 5)).toBeCloseTo(0.5, 5);
  });
  it('feather is 0 at the edge and 1 past the feather width', () => {
    expect(featherAlpha(0, 8)).toBe(0);
    expect(featherAlpha(-3, 8)).toBe(0);
    expect(featherAlpha(8, 8)).toBe(1);
    expect(featherAlpha(4, 8)).toBeGreaterThan(0);
  });
  it('forehead fade is 0 at the top and 1 at the brow', () => {
    expect(foreheadFade(10, 10, 50)).toBe(0);
    expect(foreheadFade(50, 10, 50)).toBe(1);
  });
  it('signed distance is positive inside a square', () => {
    const square: [number, number][] = [[0, 0], [10, 0], [10, 10], [0, 10]];
    expect(insideDistance(5, 5, square)).toBeCloseTo(5, 5);
    expect(insideDistance(-2, 5, square)).toBeCloseTo(-2, 5);
  });
  it('buildMask is 0 outside, 1 deep inside', () => {
    const polygon: [number, number][] = [[4, 4], [36, 4], [36, 36], [4, 36]];
    const { alpha } = buildMask({ width: 40, height: 40, polygon, feather: 4 });
    expect(alpha[1 * 40 + 1]).toBe(0);
    expect(alpha[20 * 40 + 20]).toBe(1);
    expect(alpha[20 * 40 + 5]).toBeGreaterThan(0);
    expect(alpha[20 * 40 + 5]).toBeLessThan(1);
  });
});

describe('skin tone', () => {
  it('median ignores outliers', () => {
    expect(median([1, 2, 3, 1000])).toBe(2.5);
    expect(median([5, 1, 100])).toBe(5);
    expect(medianLinearRgb([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1, 1, 1]])).toEqual([0.2, 0.2, 0.2]);
  });
  it('hex round trips', () => {
    expect(linearRgbToHex(hexToLinearRgb('#c68642'))).toBe('#c68642');
  });
  it('samples landmark windows robustly against a specular pixel', () => {
    const width = 20;
    const height = 20;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; i += 1) data.set([200, 150, 120, 255], i * 4);
    data.set([255, 255, 255, 255], (10 * width + 10) * 4);
    const landmarks = new Float32Array(3);
    landmarks[0] = 0.5;
    landmarks[1] = 0.5;
    const rgb = sampleLandmarkColors({ data, width, height }, landmarks, [0], 2);
    expect(linearRgbToHex(rgb ?? [0, 0, 0])).toBe('#c89678');
  });
});

describe('landmark to UV triangle assembly', () => {
  const faceMap = makeMockFaceMap();
  const landmarks = new Float32Array(478 * 3);
  for (let i = 0; i < 478; i += 1) {
    landmarks[i * 3] = (i % 17) / 17;
    landmarks[i * 3 + 1] = Math.floor(i / 17) / 30;
  }
  it('places vertices at landmark UV * size and photo coords at landmark positions', () => {
    const g = assembleBakeGeometry(faceMap, landmarks, 2048, 2048);
    const lm = faceMap.landmarks[100] ?? { uv: [0, 0] as [number, number] };
    expect(g.positions[100 * 3]).toBeCloseTo(lm.uv[0] * 2048, 3);
    expect(g.positions[100 * 3 + 1]).toBeCloseTo(lm.uv[1] * 2048, 3);
    expect(g.photoUvs[100 * 2]).toBeCloseTo(landmarks[100 * 3] ?? -1, 6);
    expect(g.photoUvs[100 * 2 + 1]).toBeCloseTo(landmarks[100 * 3 + 1] ?? -1, 6);
    expect(g.indices.length).toBe(faceMap.triangles.length * 3);
  });
  it('skips triangles with unbound landmarks', () => {
    const partial = { ...faceMap, landmarks: faceMap.landmarks.filter((l) => l.index !== 0) };
    const g = assembleBakeGeometry(partial, landmarks, 100, 100);
    expect(g.indices.length).toBeLessThan(faceMap.triangles.length * 3);
    expect(Array.from(g.indices)).not.toContain(0);
  });
});

describe('delighting', () => {
  it('gain pulls dark and bright areas toward the mean', () => {
    expect(delightGain(0.2, 0.4, 1)).toBeCloseTo(2, 5);
    expect(delightGain(0.8, 0.4, 1)).toBeCloseTo(0.5, 5);
    expect(delightGain(0.3, 0.3, 0.6)).toBeCloseTo(1, 5);
    expect(delightGain(0.2, 0.4, 0)).toBe(1);
  });
  it('masked blur ignores zero-weight pixels', () => {
    const values = new Float32Array([1, 1, 1, 9, 1, 1, 1, 1, 1]);
    const weights = new Float32Array([1, 1, 1, 0, 1, 1, 1, 1, 1]);
    const out = maskedBlur(values, weights, 3, 3, 1);
    expect(out[4]).toBeCloseTo(1, 4);
  });
  it('processFaceCrop flattens a lighting gradient and writes alpha', () => {
    const width = 32;
    const height = 8;
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) rgba.set([60 + x * 5, 60 + x * 5, 60 + x * 5, 255], (y * width + x) * 4);
    }
    const spread = (data: Uint8ClampedArray) => (data[(4 * width + 30) * 4] ?? 0) - (data[(4 * width + 2) * 4] ?? 0);
    const before = spread(rgba);
    const maskAlpha = new Float32Array(width * height).fill(1);
    const inside = new Float32Array(width * height).fill(1000);
    processFaceCrop({
      rgba, width, height, maskAlpha, inside, feather: 4, skinLinear: [0.5, 0.4, 0.3], strength: 1, blurRadius: 12,
    });
    expect(spread(rgba)).toBeLessThan(before);
    expect(rgba[3]).toBe(255);
  });
});

describe('mask: rotated forehead axis and holes', () => {
  const polygon: [number, number][] = [[0, 0], [100, 0], [100, 100], [0, 100]];
  it('fades along an arbitrary axis (top of the head toward -x)', () => {
    const forehead = { origin: [60, 50] as [number, number], axis: [-1, 0] as [number, number], topT: 50, browT: 20 };
    const { alpha } = buildMask({ width: 100, height: 100, polygon, feather: 1, forehead });
    expect(alpha[50 * 100 + 5]).toBeLessThan(0.05); // top of the head side
    expect(alpha[50 * 100 + 80]).toBe(1);
  });
  it('leaves holes unpainted', () => {
    const hole: [number, number][] = [[40, 40], [60, 40], [60, 60], [40, 60]];
    const { alpha } = buildMask({ width: 100, height: 100, polygon, feather: 1, holes: [{ polygon: hole, grow: 2, feather: 2 }] });
    expect(alpha[50 * 100 + 50]).toBe(0);
    expect(alpha[20 * 100 + 20]).toBe(1);
  });
  it('sorts contour points by angle', () => {
    const sorted = sortByAngle([[1, 1], [-1, -1], [1, -1], [-1, 1]]);
    expect(sorted).toHaveLength(4);
    const angles = sorted.map(([x, y]) => Math.atan2(y, x));
    expect([...angles].sort((a, b) => a - b)).toEqual(angles);
  });
});

describe('assembleBakeGeometry slivers', () => {
  it('skips collapsed and flipped triangles', () => {
    const map = makeMockFaceMap();
    // a flipped triangle (minority orientation) and a collapsed one (two identical landmark UVs)
    const flipped: [number, number, number] = [0, 19, 1];
    const collapsedLm = map.landmarks[100]!;
    map.landmarks[101] = { ...map.landmarks[101]!, uv: [...collapsedLm.uv] };
    map.triangles.push(flipped, [100, 101, 102]);
    const landmarks = new Float32Array(478 * 3);
    const geometry = assembleBakeGeometry(map, landmarks, 100, 100);
    const triples = new Set<string>();
    for (let i = 0; i < geometry.indices.length; i += 3) triples.add(`${geometry.indices[i]},${geometry.indices[i + 1]},${geometry.indices[i + 2]}`);
    expect(triples.has('0,19,1')).toBe(false);
    expect(triples.has('100,101,102')).toBe(false);
    expect(triples.has('0,1,18')).toBe(true);
  });
});
