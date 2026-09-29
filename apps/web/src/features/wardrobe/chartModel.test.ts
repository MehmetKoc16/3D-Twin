import { describe, expect, it } from 'vitest';
import type { GarmentTemplateDef } from '@dt/avatar-core';
import {
  addSize,
  createDraft,
  draftToItem,
  euRow,
  itemToDraft,
  measuresForKind,
  parseCm,
  removeSize,
  renameSize,
  setCell,
  shoeInnerLengthCm,
  type ChartDraft,
} from './chartModel';

function template(partial: Partial<GarmentTemplateDef> & Pick<GarmentTemplateDef, 'id' | 'kind' | 'category'>): GarmentTemplateDef {
  return {
    label: { tr: partial.id, en: partial.id },
    license: 'CC0-1.0',
    mesh: `${partial.id}.glb`,
    nativeMeasures: {},
    defaultEase: {},
    layer: 1,
    baseColor: '#808080',
    ...partial,
  };
}

const tshirt = template({ id: 'tshirt', kind: 'tshirt', category: 'top', baseColor: '#838383' });
const pants = template({ id: 'pants', kind: 'pants', category: 'bottom' });
const sneakers = template({ id: 'sneakers', kind: 'sneakers', category: 'shoes' });

function filled(base: ChartDraft, rows: Record<string, string[]>): ChartDraft {
  return { ...base, name: 'Basic tee', chart: rows };
}

describe('parseCm', () => {
  it('accepts dot and comma decimals and rejects garbage', () => {
    expect(parseCm('96')).toBe(96);
    expect(parseCm(' 96,5 ')).toBe(96.5);
    expect(parseCm('96.5')).toBe(96.5);
    expect(parseCm('')).toBeUndefined();
    expect(parseCm('abc')).toBeUndefined();
    expect(parseCm('-3')).toBeUndefined();
    expect(parseCm('1e3')).toBeUndefined();
    expect(parseCm(undefined)).toBeUndefined();
  });
});

describe('measuresForKind', () => {
  it('lists the rows of each garment family', () => {
    expect(measuresForKind('tshirt')).toEqual(['chest', 'waist', 'length', 'sleeve', 'shoulder']);
    expect(measuresForKind('jeans')).toEqual(['waist', 'hip', 'inseam', 'thigh']);
    expect(measuresForKind('boots')).toEqual(['footLength']);
  });
});

describe('draftToItem', () => {
  const base = createDraft(tshirt);

  it('builds a StoreItemDef from a valid chart', () => {
    const draft = filled(base, { chest: ['92', '98', '104'], waist: ['84', '90', '96'], length: ['66', '68', '70'] });
    const result = draftToItem(draft, tshirt, 'id-1');
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.item).toEqual({
      id: 'id-1',
      name: 'Basic tee',
      templateId: 'tshirt',
      color: '#838383',
      sizes: ['S', 'M', 'L'],
      chart: { chest: [92, 98, 104], waist: [84, 90, 96], length: [66, 68, 70] },
      selectedSize: 'M',
    });
  });

  it('doubles girths (only) when measured flat', () => {
    const draft = { ...filled(base, { chest: ['46', '49', '52'], length: ['66', '68', '70'] }), flat: true };
    const result = draftToItem(draft, tshirt);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.item.chart.chest).toEqual([92, 98, 104]);
    expect(result.item.chart.length).toEqual([66, 68, 70]); // lengths are never doubled
  });

  it('range-checks after the flat doubling', () => {
    // 46 flat -> 92 cm is fine, but 46 typed as a full chest (60..200) is not
    const flat = draftToItem({ ...filled(base, { chest: ['46', '49', '52'] }), flat: true }, tshirt);
    const full = draftToItem(filled(base, { chest: ['46', '49', '52'] }), tshirt);
    expect(flat.ok).toBe(true);
    expect(full.ok).toBe(false);
    if (!full.ok) expect(full.issues.map((i) => i.code)).toContain('outOfRange');
  });

  it('accepts decimal commas and rounds to 0.1 cm', () => {
    const result = draftToItem(filled(base, { chest: ['92,54', '98,0', '104'] }), tshirt);
    expect(result.ok && result.item.chart.chest).toEqual([92.5, 98, 104]);
  });

  it('reports every problem with a friendly code', () => {
    const draft: ChartDraft = {
      ...base,
      name: '   ',
      storeUrl: 'not a url',
      color: 'red',
      chart: { chest: ['92', 'x', '104'], waist: ['80', '', ''] },
    };
    const result = draftToItem(draft, tshirt);
    expect(result.ok).toBe(false);
    if (result.ok) return;
    const codes = result.issues.map((i) => i.code);
    expect(codes).toEqual(expect.arrayContaining(['nameRequired', 'urlInvalid', 'colorInvalid', 'invalidNumber', 'incompleteRow']));
  });

  it('requires the main girth row', () => {
    const result = draftToItem(filled(base, { length: ['66', '68', '70'] }), tshirt);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.issues).toContainEqual({ code: 'missingRequired', measure: 'chest' });
    const bottoms = draftToItem({ ...createDraft(pants), name: 'x', chart: { hip: ['96', '100', '104'] } }, pants);
    expect(bottoms.ok).toBe(false);
  });

  it('rejects unsorted rows, duplicate / empty sizes and a missing selection', () => {
    const decreasing = draftToItem(filled(base, { chest: ['104', '98', '92'] }), tshirt);
    expect(decreasing.ok === false && decreasing.issues.map((i) => i.code)).toContain('notIncreasing');

    const duplicate = draftToItem({ ...filled(base, { chest: ['92', '98', '104'] }), sizes: ['S', 's', ''] }, tshirt);
    expect(duplicate.ok === false && duplicate.issues.map((i) => i.code)).toEqual(expect.arrayContaining(['duplicateSize', 'emptySize']));

    const selection = draftToItem({ ...filled(base, { chest: ['92', '98', '104'] }), selectedSize: 'XL' }, tshirt);
    expect(selection.ok === false && selection.issues.map((i) => i.code)).toContain('selectedSizeMissing');
  });

  it('accepts http(s) store links only', () => {
    const ok = draftToItem({ ...filled(base, { chest: ['92', '98', '104'] }), storeUrl: 'https://shop.example/tee' }, tshirt);
    expect(ok.ok && ok.item.storeUrl).toBe('https://shop.example/tee');
    const bad = draftToItem({ ...filled(base, { chest: ['92', '98', '104'] }), storeUrl: 'javascript:alert(1)' }, tshirt);
    expect(bad.ok).toBe(false);
  });

  it('fails without a template', () => {
    const result = draftToItem(base, undefined);
    expect(result.ok === false && result.issues.map((i) => i.code)).toContain('templateUnknown');
  });
});

describe('shoes', () => {
  const base = { ...createDraft(sneakers), name: 'Runner' };

  it('derives the inner length from EU size labels', () => {
    const cm42 = shoeInnerLengthCm('42');
    expect(cm42).toBeGreaterThan(26);
    expect(cm42).toBeLessThan(29);
    expect(shoeInnerLengthCm('S')).toBeUndefined();
    expect(shoeInnerLengthCm('60')).toBeUndefined();
    expect(euRow(['41', '42'])).toHaveLength(2);
    const result = draftToItem({ ...base, sizes: ['41', '42', '43'], selectedSize: '42' }, sneakers);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const row = result.item.chart.footLength ?? [];
    expect(row).toHaveLength(3);
    expect(row[1]!).toBeGreaterThan(row[0]!);
    expect(row[1]!).toBe(cm42);
  });

  it('rejects size names that are not EU numbers in EU mode, but accepts a typed cm row', () => {
    const bad = draftToItem({ ...base, sizes: ['S', 'M'], selectedSize: 'S' }, sneakers);
    expect(bad.ok === false && bad.issues.map((i) => i.code)).toContain('euSizeInvalid');
    const typed = draftToItem(
      { ...base, shoeMode: 'cm', sizes: ['S', 'M'], selectedSize: 'S', chart: { footLength: ['26,5', '27.5'] } },
      sneakers,
    );
    expect(typed.ok && typed.item.chart.footLength).toEqual([26.5, 27.5]);
  });

  it('restores the EU mode of a saved item', () => {
    const result = draftToItem({ ...base, sizes: ['41', '42'], selectedSize: '41' }, sneakers);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(itemToDraft(result.item, sneakers).shoeMode).toBe('eu');
    const custom = { ...result.item, chart: { footLength: [26, 27] } };
    expect(itemToDraft(custom, sneakers).shoeMode).toBe('cm');
  });
});

describe('editing helpers', () => {
  const base = createDraft(tshirt);

  it('adds, renames and removes sizes with their cells', () => {
    let draft = setCell(base, 'chest', 1, '98');
    draft = addSize(draft, 'XL');
    expect(draft.sizes).toEqual(['S', 'M', 'L', 'XL']);
    expect(draft.chart.chest).toEqual(['', '98', '', '']);
    draft = renameSize(draft, 1, 'Medium');
    expect(draft.selectedSize).toBe('Medium'); // the selection follows its label
    draft = removeSize(draft, 1);
    expect(draft.sizes).toEqual(['S', 'L', 'XL']);
    expect(draft.chart.chest).toEqual(['', '', '']);
    expect(draft.selectedSize).toBe('S');
  });

  it('round-trips an item through a draft', () => {
    const result = draftToItem(filled(base, { chest: ['92', '98', '104'], sleeve: ['20', '21', '22'] }), tshirt, 'x');
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const back = draftToItem(itemToDraft(result.item, tshirt), tshirt, 'x');
    expect(back.ok && back.item).toEqual(result.item);
  });
});
