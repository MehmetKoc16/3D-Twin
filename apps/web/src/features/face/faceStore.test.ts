import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('idb-keyval', () => ({
  get: vi.fn(async () => undefined),
  set: vi.fn(async () => undefined),
  del: vi.fn(async () => undefined),
}));
vi.mock('./landmarker', () => ({ detectFace: vi.fn() }));
vi.mock('./bake/bake', () => ({
  bakeFace: vi.fn(),
  readPhotoPixels: vi.fn(() => ({ data: new Uint8ClampedArray(16).fill(255), width: 2, height: 2 })),
}));
vi.mock('./bake/faceMap', () => ({
  loadFaceMap: vi.fn(),
  FaceMapUnavailableError: class FaceMapUnavailableError extends Error {},
}));

import { del, set } from 'idb-keyval';
import { bakeFace } from './bake/bake';
import { FaceMapUnavailableError, loadFaceMap } from './bake/faceMap';
import { detectFace } from './landmarker';
import { useFaceStore } from '../../store/faceStore';

const bitmap = { width: 100, height: 100, close: vi.fn() } as unknown as ImageBitmap;

function goodLandmarks(): Float32Array {
  const lm = new Float32Array(478 * 3);
  lm[234 * 3] = 0.3;
  lm[454 * 3] = 0.7;
  lm[1 * 3] = 0.5;
  lm[10 * 3 + 1] = 0.2;
  lm[152 * 3 + 1] = 0.8;
  return lm;
}

beforeEach(async () => {
  vi.stubGlobal('createImageBitmap', vi.fn(async () => bitmap));
  vi.mocked(detectFace).mockReset();
  vi.mocked(bakeFace).mockReset();
  vi.mocked(loadFaceMap).mockReset();
  vi.mocked(set).mockClear();
  vi.mocked(del).mockClear();
  await useFaceStore.getState().clear();
  useFaceStore.setState({ revision: 0 });
});

describe('faceStore', () => {
  it('rejects unsupported types and oversize files', async () => {
    await useFaceStore.getState().loadPhoto(new Blob(['x'], { type: 'image/gif' }));
    expect(useFaceStore.getState().error).toBe('face.errors.invalidType');
    await useFaceStore.getState().loadPhoto(new Blob([new Uint8Array(16 * 1024 * 1024)], { type: 'image/png' }));
    expect(useFaceStore.getState().error).toBe('face.errors.tooLarge');
  });

  it('reports no face', async () => {
    vi.mocked(detectFace).mockResolvedValue({ faceCount: 0 });
    await useFaceStore.getState().loadPhoto(new Blob(['x'], { type: 'image/png' }));
    expect(useFaceStore.getState().status).toBe('error');
    expect(useFaceStore.getState().error).toBe('face.errors.noFace');
  });

  it('detects, persists, bakes and bumps the revision', async () => {
    vi.mocked(detectFace).mockResolvedValue({ faceCount: 1, landmarks: goodLandmarks() });
    await useFaceStore.getState().loadPhoto(new Blob(['x'], { type: 'image/jpeg' }));
    expect(useFaceStore.getState().status).toBe('ready-to-bake');
    expect(vi.mocked(set).mock.calls.map((c) => c[0])).toEqual(
      expect.arrayContaining(['face:photo', 'face:landmarks']),
    );

    vi.mocked(loadFaceMap).mockResolvedValue({} as never);
    const overlayCanvas = {} as HTMLCanvasElement;
    vi.mocked(bakeFace).mockResolvedValue({
      overlayCanvas,
      skinToneHex: '#aa8866',
      uvBounds: { min: [0, 0], max: [1, 1] },
    });
    await useFaceStore.getState().bake();
    const state = useFaceStore.getState();
    expect(state.status).toBe('baked');
    expect(state.skinToneHex).toBe('#aa8866');
    expect(state.revision).toBe(1);
  });

  it('maps a missing face map to a friendly error', async () => {
    vi.mocked(detectFace).mockResolvedValue({ faceCount: 1, landmarks: goodLandmarks() });
    await useFaceStore.getState().loadPhoto(new Blob(['x'], { type: 'image/webp' }));
    vi.mocked(loadFaceMap).mockRejectedValue(new FaceMapUnavailableError());
    await useFaceStore.getState().bake();
    expect(useFaceStore.getState().error).toBe('face.errors.faceMapMissing');
  });

  it('clear removes stored data', async () => {
    await useFaceStore.getState().clear();
    expect(vi.mocked(del).mock.calls.map((c) => c[0])).toEqual(
      expect.arrayContaining(['face:photo', 'face:landmarks', 'face:skin', 'face:size']),
    );
    expect(useFaceStore.getState().status).toBe('idle');
  });
});
