import { describe, expect, it } from 'vitest';
import { convertShoeSize, footLengthCmFromShoe, shoeSizes } from './shoeSize';

describe('shoe sizing lookup', () => {
  it('covers EU 35 through 48 and grows by foot length', () => {
    expect(shoeSizes.map((row) => row.EU)).toEqual(Array.from({ length: 14 }, (_, i) => i + 35));
    expect(footLengthCmFromShoe('EU', 42)).toBe(27);
    expect(footLengthCmFromShoe('EU', 48)).toBeGreaterThan(footLengthCmFromShoe('EU', 35));
  });
  it('preserves the same lookup row across systems', () => {
    for (const row of shoeSizes) {
      expect(convertShoeSize(row.EU, 'EU', 'UK')).toBe(row.UK);
      expect(convertShoeSize(row.US_M, 'US_M', 'US_W')).toBe(row.US_W);
    }
  });
});
