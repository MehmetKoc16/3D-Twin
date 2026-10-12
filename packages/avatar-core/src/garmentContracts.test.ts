import { describe, expect, it } from 'vitest';
import { isGarmentLicense, validateGarmentLicense } from './garmentContracts';

describe('garment licence validation', () => {
  it.each(['CC0-1.0', 'CC-BY-3.0', 'CC-BY-4.0', 'user'])(
    'accepts the exact supported identifier %s',
    (license) => {
      expect(isGarmentLicense(license)).toBe(true);
      expect(() =>
        validateGarmentLicense(license, 'Title by Author, source, exact licence, changes'),
      ).not.toThrow();
    },
  );
  it.each(['CC-BY-3.0', 'CC-BY-4.0'])('requires attribution for %s', (license) => {
    for (const attribution of [undefined, null, '', '   '])
      expect(() => validateGarmentLicense(license, attribution)).toThrow(/requires attribution/);
  });
  it.each(['CC-BY-SA-3.0', 'CC-BY-NC-3.0', 'CC-BY', 'CC-BY-2.0', undefined])(
    'rejects unsupported identifier %s',
    (license) => {
      expect(isGarmentLicense(license)).toBe(false);
      expect(() => validateGarmentLicense(license, 'Author')).toThrow(/unsupported/);
    },
  );
});
