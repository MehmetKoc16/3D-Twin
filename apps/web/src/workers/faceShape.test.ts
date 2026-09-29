import { describe, expect, it, vi } from 'vitest';
import type { FaceMapDef } from '@dt/avatar-core';
import type { AvatarFaceFit } from './avatarProtocol';
import { FaceShapeState, type FaceFitRequest } from './faceShape';

const request: FaceFitRequest = {
  faceMap: {} as FaceMapDef,
  landmarks: new Float32Array(478 * 3),
  imageWidth: 800,
  imageHeight: 1000,
};
const fit: AvatarFaceFit = { modifierValues: { 'head/head-fat': 0.4 }, rmsResidual: 1.2 };

describe('FaceShapeState', () => {
  it('does nothing without landmarks', () => {
    const fitter = vi.fn(() => fit);
    expect(new FaceShapeState(fitter).apply()).toEqual({ values: null });
    expect(fitter).not.toHaveBeenCalled();
  });

  it('fits once on the next solve, then only merges the cached values', () => {
    const fitter = vi.fn(() => fit);
    const state = new FaceShapeState(fitter);
    state.set(request);
    const first = state.apply();
    expect(first).toEqual({ values: fit.modifierValues, report: fit });
    const second = state.apply();
    expect(second).toEqual({ values: fit.modifierValues });
    expect(fitter).toHaveBeenCalledTimes(1);
    expect(fitter).toHaveBeenCalledWith(request);
  });

  it('refits after new landmarks and stops merging after clear', () => {
    const fitter = vi.fn(() => fit);
    const state = new FaceShapeState(fitter);
    state.set(request);
    state.apply();
    state.set(request);
    expect(state.apply().report).toBe(fit);
    state.clear();
    expect(state.active).toBe(false);
    expect(state.apply()).toEqual({ values: null });
    expect(fitter).toHaveBeenCalledTimes(2);
  });

  it('reports a failed fit as null and keeps the body unchanged', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const state = new FaceShapeState(() => {
      throw new Error('boom');
    });
    state.set(request);
    expect(state.apply()).toEqual({ values: null, report: null });
    expect(state.apply()).toEqual({ values: null });
    warn.mockRestore();
  });
});
