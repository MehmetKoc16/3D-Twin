import { describe, expect, it } from 'vitest';
import type { ShoeSystem } from './contracts';
import { convertShoeSize, footLengthCmFromShoe, roundShoeSize, shoeFromFootLengthCm } from './shoe';

const SYSTEMS: ShoeSystem[] = ['EU', 'US_M', 'US_W', 'UK'];

describe('shoe sizes', () => {
  it('matches the EU ~ 1.5 * foot_cm + 2 sanity relation over the table range', () => {
    for (let eu = 35; eu <= 48; eu++) {
      const cm = footLengthCmFromShoe({ system: 'EU', size: eu });
      expect(1.5 * cm + 2).toBeCloseTo(eu, 0);
      expect(Math.abs(1.5 * cm + 2 - eu)).toBeLessThan(0.1);
    }
  });

  it('gives plausible anchors', () => {
    expect(footLengthCmFromShoe({ system: 'EU', size: 42 })).toBeCloseTo(26.7, 1);
    expect(footLengthCmFromShoe({ system: 'UK', size: 8 })).toBeCloseTo(26.7, 1);
    expect(footLengthCmFromShoe({ system: 'US_M', size: 9 })).toBeCloseTo(26.7, 1);
    expect(footLengthCmFromShoe({ system: 'US_W', size: 10.5 })).toBeCloseTo(26.7, 1);
  });

  it('round-trips size -> foot length -> size in every system (table rows and interpolated sizes)', () => {
    for (const system of SYSTEMS) {
      const sizes = {
        EU: [35, 38.5, 42, 44.3, 48],
        UK: [2.5, 5.5, 8, 10.25, 12.5],
        US_M: [3.5, 6.5, 9, 12, 13.5],
        US_W: [5, 8, 10.5, 12.75, 15],
      }[system];
      for (const size of sizes) {
        const cm = footLengthCmFromShoe({ system, size });
        expect(shoeFromFootLengthCm(cm, system)).toBeCloseTo(size, 9);
      }
    }
  });

  it('round-trips foot length -> size -> foot length', () => {
    for (const system of SYSTEMS)
      for (let cm = 22; cm <= 30.5; cm += 0.37) {
        const size = shoeFromFootLengthCm(cm, system);
        expect(footLengthCmFromShoe({ system, size })).toBeCloseTo(cm, 9);
      }
  });

  it('is monotone and extrapolates smoothly outside the table', () => {
    let prev = -Infinity;
    for (let eu = 30; eu <= 52; eu += 0.5) {
      const cm = footLengthCmFromShoe({ system: 'EU', size: eu });
      expect(cm).toBeGreaterThan(prev);
      prev = cm;
    }
    expect(footLengthCmFromShoe({ system: 'EU', size: 30 })).toBeCloseTo((30 - 2) / 1.5, 0);
  });

  it('converts between systems and rounds to half sizes', () => {
    expect(convertShoeSize({ system: 'EU', size: 42 }, 'UK')).toBeCloseTo(8, 6);
    expect(convertShoeSize({ system: 'UK', size: 8 }, 'US_M')).toBeCloseTo(9, 6);
    expect(roundShoeSize(41.74)).toBe(41.5);
    expect(roundShoeSize(41.76)).toBe(42);
  });

  it('rejects non-finite input', () => {
    expect(() => footLengthCmFromShoe({ system: 'EU', size: NaN })).toThrow(RangeError);
    expect(() => shoeFromFootLengthCm(Infinity, 'EU')).toThrow(RangeError);
  });
});
