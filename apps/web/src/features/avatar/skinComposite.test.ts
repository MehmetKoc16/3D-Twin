import { describe, expect, it } from 'vitest';
import { skinTones } from '../../store/appearanceStore';
import { compositeSkinMap, resolveSkinHex, type CompositeContext } from './skinComposite';

function recorder() {
  const calls: string[] = [];
  const context: CompositeContext = {
    fillStyle: '',
    fillRect(x, y, w, h) {
      calls.push(`fill ${String(this.fillStyle)} ${x},${y},${w},${h}`);
    },
    drawImage(image, x, y) {
      calls.push(`draw ${String(image)} ${x},${y}`);
    },
  };
  return { calls, context };
}

describe('compositeSkinMap', () => {
  it('fills the whole map with the tone, then draws the overlay on top', () => {
    const { calls, context } = recorder();
    compositeSkinMap(context, 2048, '#c68a63', 'overlay' as unknown as CanvasImageSource);
    expect(calls).toEqual(['fill #c68a63 0,0,2048,2048', 'draw overlay 0,0']);
  });

  it('draws no overlay without a face', () => {
    const { calls, context } = recorder();
    compositeSkinMap(context, 64, '#ffffff');
    expect(calls).toEqual(['fill #ffffff 0,0,64,64']);
  });
});

describe('resolveSkinHex', () => {
  const preset = skinTones[2]!.color;
  it('prefers the photo tone when enabled and available', () => {
    expect(resolveSkinHex({ mode: 'skin', toneIndex: 2, useFaceTone: true, faceToneHex: '#aa8866' })).toBe('#aa8866');
  });
  it('falls back to the preset when off or without a photo tone', () => {
    expect(resolveSkinHex({ mode: 'skin', toneIndex: 2, useFaceTone: false, faceToneHex: '#aa8866' })).toBe(preset);
    expect(resolveSkinHex({ mode: 'skin', toneIndex: 2, useFaceTone: true })).toBe(preset);
  });
  it('clamps an invalid preset index', () => {
    expect(resolveSkinHex({ mode: 'skin', toneIndex: 99, useFaceTone: false })).toBe(skinTones[1]!.color);
  });
});
