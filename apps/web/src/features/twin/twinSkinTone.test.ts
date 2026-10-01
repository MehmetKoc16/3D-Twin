import { describe, expect, it } from 'vitest';
import { DataTexture, Vector2 } from 'three';
import { fakeModel } from './twinTestkit';
import {
  NEUTRAL_SKIN_TONE,
  resolveSkinTone,
  sampleSkinTone,
  textureUVs,
  type TexturePixels,
} from './twinSkinTone';

const names = ['Root', 'lowerarm_l', 'lowerarm_r', 'hand_l'];
function samples(colors: number[][]): {
  pixels: TexturePixels;
  uv: Float32Array;
  joints: Uint16Array;
  weights: Float32Array;
} {
  return {
    pixels: {
      width: colors.length,
      height: 1,
      data: Uint8ClampedArray.from(colors.flatMap((rgb) => [...rgb, 255])),
    },
    uv: Float32Array.from(colors.flatMap((_, i) => [(i + 0.5) / colors.length, 0.5])),
    joints: Uint16Array.from(colors.flatMap((_, i) => [(i % 2) + 1, 0, 0, 0])),
    weights: Float32Array.from(colors.flatMap(() => [0.75, 0.25, 0, 0])),
  };
}

describe('forearm skin tone', () => {
  it('samples both lowerarms, rejects clothing/highlight/shadow outliers and takes channel medians', () => {
    const s = samples([
      [200, 150, 120],
      [202, 152, 122],
      [204, 154, 124],
      [0, 0, 255],
      [255, 255, 255],
      [0, 0, 0],
    ]);
    expect(sampleSkinTone(s.pixels, s.uv, s.joints, s.weights, names)).toBe('#ca987a');
  });

  it('requires more than 0.5 lowerarm weight, ignoring hands and other regions', () => {
    const s = samples([
      [201, 154, 126],
      [0, 0, 255],
      [255, 255, 255],
    ]);
    s.weights[4] = 0.5;
    s.joints[8] = 3;
    expect(sampleSkinTone(s.pixels, s.uv, s.joints, s.weights, names)).toBe('#c99a7e');
  });

  it('counts texels once even at duplicated UV vertices and respects glTF image rows', () => {
    const s = samples([
      [190, 140, 110],
      [210, 160, 130],
    ]);
    s.joints = Uint16Array.from([1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0]);
    s.weights = Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
    s.pixels.width = 1;
    s.pixels.height = 2;
    s.uv = Float32Array.from([0.5, 0.25, 0.5, 0.75, 0.5, 0.25]);
    expect(sampleSkinTone(s.pixels, s.uv, s.joints, s.weights, names)).toBe('#c89678');
  });

  it('never falls back to white for missing, white, transparent or unusable samples', () => {
    const s = samples([
      [255, 255, 255],
      [201, 154, 126],
    ]);
    s.pixels.data[7] = 0;
    expect(sampleSkinTone(s.pixels, s.uv, s.joints, s.weights, names)).toBe(NEUTRAL_SKIN_TONE);
    expect(resolveSkinTone(fakeModel(), names)).toBe(NEUTRAL_SKIN_TONE);
    s.weights.fill(0);
    expect(sampleSkinTone(s.pixels, s.uv, s.joints, s.weights, names)).toBe(NEUTRAL_SKIN_TONE);
  });

  it('applies the original UV transform and caches the result across loads of one twin', () => {
    const s = samples([
      [201, 154, 126],
      [150, 100, 80],
    ]);
    const model = fakeModel();
    const texture = new DataTexture(new Uint8Array(s.pixels.data), 2, 1);
    texture.flipY = false;
    texture.offset = new Vector2(0.5, 0);
    model.material.map = texture;
    model.uv = Float32Array.from([0.25, 0.5]);
    model.skinIndex = Uint16Array.from([1, 0, 0, 0]);
    model.skinWeight = Float32Array.from([1, 0, 0, 0]);
    expect(textureUVs(model.uv, texture)?.[0]).toBeCloseTo(0.75);
    const key = {};
    expect(resolveSkinTone(model, names, key)).toBe('#966450');
    model.material.map = null;
    expect(resolveSkinTone(model, names, key)).toBe('#966450');
    expect(resolveSkinTone(model, names, {})).toBe(NEUTRAL_SKIN_TONE);
  });
});
