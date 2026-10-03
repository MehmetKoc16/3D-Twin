import { describe, expect, it, vi } from 'vitest';
import { parseTwinBundle, loadTwinBundle } from './twinBundle';
import { bundleBytes, bundleExtras } from './twinBundleTestkit';

function parser(raw: unknown = bundleExtras()) {
  return {
    json: { asset: { extras: { dtTwin: raw } }, bufferViews: [{}] },
    getDependency: vi.fn(async () => Uint32Array.from([1, 3]).buffer),
  };
}

describe('single-file twin contract', () => {
  it('reads asset.extras.dtTwin and the bufferView dependency in little-endian order', async () => {
    const p = parser();
    const bundle = await parseTwinBundle(p);
    expect(p.getDependency).toHaveBeenCalledWith('bufferView', 0);
    expect(bundle?.mapping).toEqual(Uint32Array.from([1, 3]));
    expect(bundle?.skinToneHex).toBe('#c49a7a');
    expect(bundle?.def.measurementsCm.height).toBe(170);
    expect(await loadTwinBundle(bundleBytes())).toEqual(bundle);
  });

  it('returns null for a legacy glTF with no bundle extras', async () => {
    expect(await parseTwinBundle({ json: {}, getDependency: vi.fn() })).toBeNull();
  });

  it.each([
    null,
    [],
    { version: 2 },
    { ...bundleExtras(), twin: {} },
    { ...bundleExtras(), skinToneHex: '#bad' },
    { ...bundleExtras(), provenance: {} },
    { ...bundleExtras(), mh2twin: { bufferView: -1, count: 2, componentType: 'uint32' } },
    { ...bundleExtras(), mh2twin: { bufferView: 2, count: 2, componentType: 'uint32' } },
    { ...bundleExtras(), mh2twin: { bufferView: 0.5, count: 2, componentType: 'uint32' } },
    { ...bundleExtras(), mh2twin: { bufferView: 0, count: 2, componentType: 'float32' } },
    { ...bundleExtras(), mh2twin: { bufferView: 0, count: 3, componentType: 'uint32' } },
  ])('rejects malformed extras %#', async (raw) => {
    await expect(parseTwinBundle(parser(raw))).rejects.toThrow(/twin:/);
  });

  it('rejects truncated mapping, unreadable buffer views, out-of-range mapping and external assets', async () => {
    const p = parser();
    p.getDependency.mockResolvedValueOnce(new ArrayBuffer(7));
    await expect(parseTwinBundle(p)).rejects.toThrow(/byte length/);
    p.getDependency.mockRejectedValueOnce(new Error('missing'));
    await expect(parseTwinBundle(p)).rejects.toThrow(/unreadable/);
    p.getDependency.mockResolvedValueOnce(Uint32Array.from([1, 99]).buffer);
    await expect(parseTwinBundle(p)).rejects.toThrow(/outside the body/);
    await expect(loadTwinBundle(bundleBytes(true))).rejects.toThrow(/embed/);
  });
});

describe('dtHasMakeHumanHands flag', () => {
  const withExtras = (extras: Record<string, unknown>) => ({
    ...parser(),
    json: { asset: { extras: { dtTwin: bundleExtras(), ...extras } }, bufferViews: [{}] },
  });

  it('is false for bundles without the flag, true when set', async () => {
    expect((await parseTwinBundle(parser()))?.hasMakeHumanHands).toBe(false);
    expect(
      (await parseTwinBundle(withExtras({ dtHasMakeHumanHands: false })))?.hasMakeHumanHands,
    ).toBe(false);
    const bundle = await parseTwinBundle(
      withExtras({ dtHasMakeHumanHands: true, dtScanHandsRemoved: false }),
    );
    expect(bundle?.hasMakeHumanHands).toBe(true);
  });

  it.each([
    { dtHasMakeHumanHands: 'yes' },
    { dtHasMakeHumanHands: 1 },
    { dtHasMakeHumanHands: null },
    { dtScanHandsRemoved: 'no' },
  ])('rejects a non-boolean flag %#', async (extras) => {
    await expect(parseTwinBundle(withExtras(extras))).rejects.toThrow(/must be a boolean/);
  });
});
