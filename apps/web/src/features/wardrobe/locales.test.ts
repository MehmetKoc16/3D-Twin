import { describe, expect, it } from 'vitest';
import tr from './locales/tr.json';
import en from './locales/en.json';
import type { ValidationIssue } from './chartModel';

function keys(value: unknown, prefix = ''): string[] {
  if (typeof value !== 'object' || value === null) return [prefix];
  return Object.entries(value).flatMap(([key, child]) =>
    keys(child, prefix ? `${prefix}.${key}` : key),
  );
}

describe('wardrobe locales', () => {
  it('tr and en define exactly the same keys', () => {
    expect(keys(tr).sort()).toEqual(keys(en).sort());
  });

  it('has a friendly message for every validation code', () => {
    const codes: ValidationIssue['code'][] = [
      'nameRequired',
      'nameTooLong',
      'urlInvalid',
      'templateUnknown',
      'colorInvalid',
      'noSizes',
      'tooManySizes',
      'emptySize',
      'duplicateSize',
      'euSizeInvalid',
      'missingRequired',
      'incompleteRow',
      'invalidNumber',
      'outOfRange',
      'notIncreasing',
      'selectedSizeMissing',
    ];
    const errors = tr.errors as Record<string, string>;
    const errorsEn = en.errors as Record<string, string>;
    for (const code of codes) {
      expect(errors[code], `tr ${code}`).toBeTruthy();
      expect(errorsEn[code], `en ${code}`).toBeTruthy();
    }
    expect(Object.keys(errors).sort()).toEqual([...codes].sort());
  });

  it('names every garment measure and verdict in both languages', () => {
    for (const locale of [tr, en]) {
      for (const id of [
        'chest',
        'waist',
        'hip',
        'length',
        'sleeve',
        'shoulder',
        'inseam',
        'thigh',
        'footLength',
      ]) {
        expect((locale.measure as Record<string, string>)[id], id).toBeTruthy();
      }
      for (const group of [locale.verdict.fit, locale.verdict.length] as Record<string, string>[]) {
        for (const verdict of ['tight', 'snug', 'regular', 'loose', 'oversized'])
          expect(group[verdict], verdict).toBeTruthy();
      }
    }
  });
});
