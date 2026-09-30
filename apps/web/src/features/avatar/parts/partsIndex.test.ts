// @ts-expect-error Node types are not in the browser app's tsconfig; Vitest supplies this module.
import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { parseGarmentBinding } from '@dt/avatar-core';
import { findPart, partsOf, validatePartsIndex } from './partsIndex';

const dir = new URL('../../../../public/assets/parts/', import.meta.url);
const file = (name: string): URL => new URL(name, dir);
const raw: unknown = JSON.parse(readFileSync(file('index.json'), 'utf8') as string);

function glbVertexCount(name: string): number {
  const buffer = readFileSync(file(name)) as Uint8Array;
  const jsonLength = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength).getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(buffer.subarray(20, 20 + jsonLength))) as {
    accessors: { count: number }[];
    meshes: { primitives: { attributes: { POSITION: number } }[] }[];
  };
  return json.accessors[json.meshes[0]!.primitives[0]!.attributes.POSITION]!.count;
}

describe('the shipped parts catalogue', () => {
  const index = validatePartsIndex(raw);

  it('has the expected catalogue and defaults', () => {
    expect(partsOf(index, 'eyes').map((p) => p.id)).toEqual(['eyes-default']);
    expect(partsOf(index, 'eyebrows')).toHaveLength(3);
    expect(partsOf(index, 'eyelashes')).toHaveLength(1);
    expect(partsOf(index, 'hair')).toHaveLength(6);
    expect(index.defaults).toEqual({
      eyes: 'eyes-default',
      eyebrows: 'eyebrows-default',
      eyelashes: 'eyelashes-default',
    });
    expect(findPart(index, 'hair-long')?.label.tr).toBe('Uzun saç');
  });

  it('references files that exist and bindings that match the meshes', () => {
    for (const part of index.parts) {
      for (const name of [part.mesh, part.binding, part.deleteVerts]) {
        if (name) expect(existsSync(file(name)), `${part.id}: ${name}`).toBe(true);
      }
      const bytes = readFileSync(file(part.binding)) as Uint8Array;
      const binding = parseGarmentBinding(Uint8Array.from(bytes).buffer);
      expect(binding.count, part.id).toBe(glbVertexCount(part.mesh));
    }
  });

  it('only the eyes carry an iris region; hair and brows are tintable, lashes are not', () => {
    for (const part of index.parts) expect(part.irisUv !== undefined).toBe(part.category === 'eyes');
    expect(index.parts.filter((p) => p.category === 'eyelashes').every((p) => !p.material.tintable)).toBe(true);
    expect(
      index.parts.filter((p) => p.category === 'hair' || p.category === 'eyebrows').every((p) => p.material.tintable),
    ).toBe(true);
    expect(findPart(index, 'eyes-default')?.material.doubleSided).toBe(false);
  });
});

describe('validatePartsIndex', () => {
  const part = (over: Record<string, unknown> = {}): Record<string, unknown> => ({
    id: 'hair-a',
    category: 'hair',
    label: { tr: 'A', en: 'A' },
    license: 'CC0-1.0',
    mesh: 'a.glb',
    binding: 'a.bind.bin',
    material: { alphaMode: 'MASK', alphaCutoff: 0.5, doubleSided: true, tintable: true },
    ...over,
  });
  const wrap = (parts: unknown[], defaults: unknown = {}): unknown => ({ version: 1, parts, defaults });

  it('accepts a minimal index', () => {
    expect(validatePartsIndex(wrap([part()])).parts).toHaveLength(1);
  });

  it.each([
    ['a wrong version', { version: 2, parts: [] }],
    ['duplicate ids', wrap([part(), part()])],
    ['an unknown category', wrap([part({ category: 'nails' })])],
    ['a missing label language', wrap([part({ label: { tr: 'A' } })])],
    ['a MASK part without cutoff', wrap([part({ material: { alphaMode: 'MASK', doubleSided: true, tintable: true } })])],
    ['broken scale refs', wrap([part({ scaleRefs: { x: [1, 2, 0], y: [1, 2, 1], z: [1, 2, 1] } })])],
    ['eyes without irisUv', wrap([part({ category: 'eyes', id: 'e' })])],
    ['a default of the wrong category', wrap([part()], { eyes: 'hair-a' })],
    ['a default that does not exist', wrap([part()], { hair: 'nope' })],
    ['a non-object', 42],
  ])('rejects %s', (_name, value) => {
    expect(() => validatePartsIndex(value)).toThrow(/parts index/);
  });
});
