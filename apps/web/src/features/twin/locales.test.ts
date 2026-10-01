import { describe, expect, it } from 'vitest';
import tr from './locales/tr.json';
import en from './locales/en.json';
import type { TwinErrorCode } from './twinDef';

function keys(value: unknown, prefix = ''): string[] {
  if (typeof value !== 'object' || value === null) return [prefix];
  return Object.entries(value).flatMap(([key, child]) =>
    keys(child, prefix ? `${prefix}.${key}` : key),
  );
}

describe('twin locales', () => {
  it('tr and en define exactly the same keys', () => {
    expect(keys(tr).sort()).toEqual(keys(en).sort());
  });

  it('has a message for every error code the app can report', () => {
    const codes: (TwinErrorCode | 'unknown' | 'files')[] = [
      'json',
      'version',
      'bones',
      'macros',
      'modifiers',
      'measurements',
      'mapping',
      'glb',
      'noSkin',
      'rest',
      'mismatch',
      'shape',
      'unknown',
      'files',
    ];
    for (const locale of [tr, en]) {
      expect(Object.keys(locale.errors).sort()).toEqual([...codes].sort());
      for (const text of Object.values(locale.errors)) expect(text.length).toBeGreaterThan(5);
    }
  });

  it('names every panel status', () => {
    for (const locale of [tr, en])
      for (const status of ['empty', 'partial', 'ready', 'error'])
        expect((locale.panel.status as Record<string, string>)[status], status).toBeTruthy();
  });
});
