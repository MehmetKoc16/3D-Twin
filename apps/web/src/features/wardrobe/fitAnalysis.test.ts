import { describe, expect, it } from 'vitest';
import type { GarmentTemplateDef, StoreItemDef } from '@dt/avatar-core';
import { analyzeItemFit, formatEase, isInfoOnly, isLengthMeasure } from './fitAnalysis';

const tshirt: GarmentTemplateDef = {
  id: 'tshirt',
  kind: 'tshirt',
  category: 'top',
  label: { tr: 'Tişört', en: 'T-shirt' },
  license: 'CC0-1.0',
  mesh: 'tshirt.glb',
  nativeMeasures: { chest: 90.8, waist: 75.9, length: 45.1, sleeve: 27.4 },
  defaultEase: { chest: 3, waist: 6.5, sleeve: -26 },
  layer: 1,
  baseColor: '#838383',
};

const item: StoreItemDef = {
  id: 'i',
  name: 'Tee',
  templateId: 'tshirt',
  color: '#ff0000',
  sizes: ['S', 'M', 'L'],
  chart: { chest: [92, 100, 108], waist: [80, 88, 96], sleeve: [19, 20, 21] },
  selectedSize: 'M',
};

const body = { chest: 96, waist: 80, hip: 98, armLength: 54, shoulder: 44 };

describe('analyzeItemFit', () => {
  it('reports the ease per region for the selected size and recommends a size', () => {
    const report = analyzeItemFit(item, tshirt, body);
    expect(report).toBeDefined();
    const chest = report!.regions.find((r) => r.id === 'chest')!;
    expect(chest.easeCm).toBeCloseTo(4, 6);
    expect(chest.verdict).toBe('regular');
    expect(report!.recommendedSize).toBeDefined();
    expect(report!.size).toBe('M');
  });

  it('changes with the size and with the body', () => {
    const small = analyzeItemFit({ ...item, selectedSize: 'S' }, tshirt, body)!;
    expect(small.regions.find((r) => r.id === 'chest')!.verdict).toBe('tight');
    expect(small.overall).toBe('tight');
    const large = analyzeItemFit({ ...item, selectedSize: 'L' }, tshirt, body)!;
    expect(large.regions.find((r) => r.id === 'chest')!.easeCm).toBeCloseTo(12, 6);
    const slimmer = analyzeItemFit(item, tshirt, { ...body, chest: 80 })!;
    expect(slimmer.regions.find((r) => r.id === 'chest')!.verdict).toBe('loose');
  });

  it('judges the sleeve against the template length on this body, not against the raw arm', () => {
    const report = analyzeItemFit(item, tshirt, body)!;
    const sleeve = report.regions.find((r) => r.id === 'sleeve')!;
    // reference = arm 54 + template offset -26 = 28 cm; chart 20 -> -8
    expect(sleeve.easeCm).toBeCloseTo(-8, 6);
    expect(isLengthMeasure('sleeve')).toBe(true);
    expect(isInfoOnly(tshirt, 'sleeve')).toBe(true); // short sleeve: information only, no verdict chip
    expect(
      isInfoOnly({ ...tshirt, defaultEase: { ...tshirt.defaultEase, sleeve: -1 } }, 'sleeve'),
    ).toBe(false);
    expect(report.overall).toBe('regular'); // the sleeve never drives the overall verdict
  });

  it('returns undefined for an unknown selected size', () => {
    expect(analyzeItemFit({ ...item, selectedSize: 'XXL' }, tshirt, body)).toBeUndefined();
  });
});

describe('formatEase', () => {
  it('signs and rounds', () => {
    expect(formatEase(4)).toBe('+4.0 cm');
    expect(formatEase(-1.26)).toBe('-1.3 cm');
    expect(formatEase(0)).toBe('0.0 cm');
  });
});
