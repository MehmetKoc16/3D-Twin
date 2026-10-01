import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { TwinFormatError, type TwinDef } from './twinDef';
import { envelope, fakeAssets, fakeModel, mapping } from './twinTestkit';

const loadTwinModel = vi.fn();
vi.mock('./twinModel', () => ({ loadTwinModel: (...args: unknown[]) => loadTwinModel(...args) }));

import { isTwinActive, useTwinStore, type TwinPack } from '../../store/twinStore';
import { TwinMode, type ModeParts } from './twinMode';

const def: TwinDef = {
  version: 1,
  boneOrder: ['Root', 'pelvis', 'spine_01'],
  fittedMacros: { gender: 0.6, height: 0.55 },
  fittedModifiers: { 'torso/torso-scale-horiz': -0.2 },
  measurementsCm: { height: 170, chest: 95 },
};
const pack: TwinPack = {
  def,
  glb: new ArrayBuffer(8),
  mapping,
  names: { glb: 'rigged.glb', json: 'twin.json', mapping: 'mh2twin.bin' },
};

const pristine = useTwinStore.getState();
const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

function setup() {
  const calls: string[] = [];
  const client = {
    setFixedShape: vi.fn(async (shape: unknown) => {
      calls.push(shape ? 'fixed' : 'free');
    }),
  };
  const assets = fakeAssets(client);
  const parts: { onSolve: ReturnType<typeof vi.fn>; dispose: ReturnType<typeof vi.fn> }[] = [];
  const createParts = vi.fn((): ModeParts => {
    const p = { onSolve: vi.fn(), dispose: vi.fn() };
    parts.push(p);
    return p;
  });
  const requestSolve = vi.fn(() => void calls.push('solve'));
  const mode = new TwinMode(assets, createParts, requestSolve);
  const twinMesh = () => assets.scene.children.find((c) => c.name === 'twin');
  return { assets, client, calls, parts, createParts, requestSolve, mode, twinMesh };
}

function chooseTwin(): void {
  useTwinStore.setState({
    pack,
    status: 'ready',
    mode: 'twin',
    revision: useTwinStore.getState().revision + 1,
  });
}

beforeEach(() => {
  useTwinStore.setState({ ...pristine, hydrated: true }, true);
  loadTwinModel.mockReset();
  loadTwinModel.mockImplementation(async () => fakeModel());
  vi.spyOn(console, 'error').mockImplementation(() => undefined);
});
afterEach(() => vi.restoreAllMocks());

describe('TwinMode', () => {
  it('standard mode is the unchanged behaviour: parts are mounted at once, the worker is left alone', () => {
    const { createParts, client, mode, parts } = setup();
    expect(createParts).toHaveBeenCalledTimes(1);
    expect(client.setFixedShape).not.toHaveBeenCalled();
    expect(mode.active).toBe(false);
    const solve = envelope(false);
    mode.onSolve(solve);
    expect(parts[0]!.onSolve).toHaveBeenCalledWith(solve);
    expect(mode.displaySolve(solve).achievedCm).toBe(solve.result.achievedCm);
    mode.dispose();
    expect(parts[0]!.dispose).toHaveBeenCalled();
    expect(client.setFixedShape).not.toHaveBeenCalled();
  });

  it('twin mode: parts go, the worker gets the fitted shape, the scan is mounted and shown after the first solve', async () => {
    const { assets, client, calls, parts, requestSolve, mode, twinMesh } = setup();
    chooseTwin();
    await flush();
    expect(parts[0]!.dispose).toHaveBeenCalled();
    expect(client.setFixedShape).toHaveBeenCalledWith({
      macros: def.fittedMacros,
      modifiers: def.fittedModifiers,
    });
    expect(requestSolve).toHaveBeenCalled();
    expect(calls).toEqual(['fixed', 'solve']);
    expect(mode.active).toBe(true);
    expect(twinMesh()).toBeDefined();
    expect(assets.mesh.visible).toBe(true); // swapped in with the first fixed-shape solve, not before
    const fixed = envelope(true);
    mode.onSolve(fixed);
    expect(assets.mesh.visible).toBe(false);
    expect(twinMesh()?.visible).toBe(true);
    expect(useTwinStore.getState().runtime?.hiddenTriangles).toBe(0);
    // the panel shows the twin's measurements, the garments keep the hidden body's own geometry
    const shown = mode.displaySolve(fixed);
    expect(shown.achievedCm).toEqual({ ...fixed.result.achievedCm, height: 170, chest: 95 });
    expect(shown.unreachable).toEqual([]);
    mode.dispose();
  });

  it('switching back restores the standard model: worker reset, re-solve, parts again, body visible', async () => {
    const { assets, calls, createParts, mode, twinMesh } = setup();
    chooseTwin();
    await flush();
    mode.onSolve(envelope(true));
    calls.length = 0;
    useTwinStore.getState().setMode('standard');
    await flush();
    expect(calls).toEqual(['free', 'solve']);
    expect(createParts).toHaveBeenCalledTimes(2);
    expect(twinMesh()).toBeUndefined();
    expect(assets.mesh.visible).toBe(true);
    expect(mode.active).toBe(false);
    expect(useTwinStore.getState().runtime).toBeNull();
    mode.dispose();
  });

  it('a twin.json that does not match the glb is rejected on the first solve and the standard model returns', async () => {
    const { calls, createParts, mode, twinMesh, assets } = setup();
    chooseTwin();
    await flush();
    calls.length = 0;
    const foreign = envelope(true);
    foreign.result.joints[6 + 1] = 5; // the pelvis is 4 m away from where the twin expects it
    mode.onSolve(foreign);
    await flush();
    const state = useTwinStore.getState();
    expect(state.error?.code).toBe('mismatch');
    expect(state.mode).toBe('standard');
    expect(twinMesh()).toBeUndefined();
    expect(assets.mesh.visible).toBe(true);
    expect(calls).toEqual(['free', 'solve']);
    expect(createParts).toHaveBeenCalledTimes(2);
    mode.dispose();
  });

  it('a glb that cannot be loaded reports its error code and stays on the standard model', async () => {
    loadTwinModel.mockRejectedValue(new TwinFormatError('noSkin', 'no skin'));
    const { calls, client, createParts, mode } = setup();
    chooseTwin();
    await flush();
    const state = useTwinStore.getState();
    expect(state.error?.code).toBe('noSkin');
    expect(state.mode).toBe('standard');
    expect(isTwinActive(state)).toBe(false);
    expect(client.setFixedShape).not.toHaveBeenCalled(); // nothing was switched, nothing to undo
    expect(calls).toEqual([]);
    expect(createParts).toHaveBeenCalledTimes(2);
    mode.dispose();
  });

  it('a shape the worker rejects falls back and undoes the switch', async () => {
    const { calls, client, mode } = setup();
    client.setFixedShape.mockImplementationOnce(async () => {
      throw new Error('unknown modifier');
    });
    chooseTwin();
    await flush();
    expect(useTwinStore.getState().error?.code).toBe('shape');
    expect(useTwinStore.getState().mode).toBe('standard');
    expect(calls).toContain('free');
    mode.dispose();
  });

  it('quick toggling ends in the last requested state with exactly one scan', async () => {
    const { calls, mode, twinMesh, assets } = setup();
    chooseTwin();
    useTwinStore.getState().setMode('standard');
    useTwinStore.getState().setMode('twin');
    await flush();
    await flush();
    expect(assets.scene.children.filter((c) => c.name === 'twin')).toHaveLength(1);
    expect(twinMesh()).toBeDefined();
    expect(mode.active).toBe(true);
    // the worker's last word is the fixed shape
    expect(calls.filter((c) => c !== 'solve').at(-1)).toBe('fixed');
    useTwinStore.getState().setMode('standard');
    await flush();
    expect(calls.filter((c) => c !== 'solve').at(-1)).toBe('free');
    expect(twinMesh()).toBeUndefined();
    mode.dispose();
  });

  it('removing the twin while it is shown returns to the standard model', async () => {
    const { mode, twinMesh, calls } = setup();
    chooseTwin();
    await flush();
    calls.length = 0;
    await useTwinStore.getState().remove();
    await flush();
    expect(twinMesh()).toBeUndefined();
    expect(calls).toEqual(['free', 'solve']);
    mode.dispose();
  });

  it('dispose while the twin is shown resets the worker', async () => {
    const { calls, mode, twinMesh } = setup();
    chooseTwin();
    await flush();
    calls.length = 0;
    mode.dispose();
    expect(calls).toEqual(['free']);
    expect(twinMesh()).toBeUndefined();
  });
});
