import { del, get, set } from 'idb-keyval';
import { create } from 'zustand';
import { FaceMapUnavailableError, loadFaceMap } from '../features/face/bake/faceMap';
import { assessQuality, landmarkBox, LANDMARK_COUNT, meanLumaInBox, type QualityHint } from '../features/face/quality';

export type FaceStatus =
  | 'idle'
  | 'loading-model'
  | 'detecting'
  | 'ready-to-bake'
  | 'baking'
  | 'baked'
  | 'error';

/** Result of fitting the avatar's face modifiers to the photo landmarks (computed in the avatar worker). */
export interface FaceFit {
  modifierValues: Record<string, number>;
  /** RMS landmark residual after the fit, in mm of the avatar head. */
  rmsResidual: number;
}

export const MAX_PHOTO_BYTES = 15 * 1024 * 1024;
export const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const KEYS = { photo: 'face:photo', landmarks: 'face:landmarks', skin: 'face:skin', size: 'face:size' } as const;

interface FaceState {
  status: FaceStatus;
  /** i18n key of the last error. */
  error?: string;
  photoBlob?: Blob;
  bitmap?: ImageBitmap;
  landmarks?: Float32Array;
  imageSize?: { width: number; height: number };
  hints: QualityHint[];
  overlayCanvas?: HTMLCanvasElement;
  skinToneHex?: string;
  /** Bumped on every successful bake and on clear. */
  revision: number;
  /** Face-shape fit reported by the worker; undefined until fitted, null if the fit failed. */
  fit?: FaceFit | null;
  setFit: (fit: FaceFit | null) => void;
  loadPhoto: (blob: Blob) => Promise<void>;
  bake: () => Promise<void>;
  hydrate: () => Promise<void>;
  /** Restores a stored photo and re-bakes it if needed (used by the integration on startup). */
  ensureBaked: () => Promise<void>;
  clear: () => Promise<void>;
}

const initial = {
  status: 'idle' as FaceStatus,
  error: undefined,
  photoBlob: undefined,
  bitmap: undefined,
  landmarks: undefined,
  imageSize: undefined,
  hints: [] as QualityHint[],
  overlayCanvas: undefined,
  skinToneHex: undefined,
  fit: undefined,
};

let token = 0;

async function persist(entries: [string, unknown][]): Promise<void> {
  try {
    await Promise.all(entries.map(([key, value]) => set(key, value)));
  } catch {
    /* storage unavailable (private mode / quota): the session still works */
  }
}

function decode(blob: Blob): Promise<ImageBitmap> {
  return createImageBitmap(blob, { imageOrientation: 'from-image' });
}

export const useFaceStore = create<FaceState>((update, read) => ({
  ...initial,
  revision: 0,
  setFit: (fit) => update({ fit }),

  loadPhoto: async (blob) => {
    const mine = ++token;
    if (!ACCEPTED_TYPES.includes(blob.type)) return update({ status: 'error', error: 'face.errors.invalidType' });
    if (blob.size > MAX_PHOTO_BYTES) return update({ status: 'error', error: 'face.errors.tooLarge' });
    read().bitmap?.close();
    update({ ...initial, status: 'loading-model' });
    try {
      const bitmap = await decode(blob);
      if (mine !== token) return bitmap.close();
      update({ status: 'detecting' });
      // heavy modules (MediaPipe, WebGL baker) load on demand so the app shell stays small
      const { detectFace } = await import('../features/face/landmarker');
      const detected = await detectFace(bitmap);
      if (mine !== token) return bitmap.close();
      if (!detected.landmarks || detected.landmarks.length < LANDMARK_COUNT * 3) {
        bitmap.close();
        return update({ status: 'error', error: 'face.errors.noFace', hints: ['noFace'] });
      }
      const imageSize = { width: bitmap.width, height: bitmap.height };
      let faceLuma: number | undefined;
      try {
        const { readPhotoPixels } = await import('../features/face/bake/bake');
        faceLuma = meanLumaInBox(readPhotoPixels(bitmap), landmarkBox(detected.landmarks));
      } catch {
        faceLuma = undefined;
      }
      const hints = assessQuality({
        landmarks: detected.landmarks,
        faceCount: detected.faceCount,
        imageWidth: imageSize.width,
        imageHeight: imageSize.height,
        faceLuma,
      });
      update({ status: 'ready-to-bake', photoBlob: blob, bitmap, landmarks: detected.landmarks, imageSize, hints });
      await persist([[KEYS.photo, blob], [KEYS.landmarks, detected.landmarks], [KEYS.size, imageSize]]);
    } catch {
      if (mine === token) update({ status: 'error', error: 'face.errors.detectFailed' });
    }
  },

  bake: async () => {
    const { bitmap, landmarks } = read();
    if (!bitmap || !landmarks) return;
    const mine = token;
    update({ status: 'baking', error: undefined });
    try {
      const faceMap = await loadFaceMap();
      const { bakeFace } = await import('../features/face/bake/bake');
      const result = await bakeFace(bitmap, landmarks, faceMap);
      if (mine !== token) return;
      update((state) => ({
        status: 'baked',
        overlayCanvas: result.overlayCanvas,
        skinToneHex: result.skinToneHex,
        fit: undefined,
        revision: state.revision + 1,
      }));
      await persist([[KEYS.skin, result.skinToneHex]]);
    } catch (error) {
      if (mine !== token) return;
      console.warn('[face] bake failed', error);
      update({
        status: 'error',
        error: error instanceof FaceMapUnavailableError ? 'face.errors.faceMapMissing' : 'face.errors.bakeFailed',
      });
    }
  },

  hydrate: async () => {
    if (read().photoBlob || read().status !== 'idle') return;
    try {
      const [photo, landmarks, skin, size] = await Promise.all([
        get<Blob>(KEYS.photo), get<Float32Array>(KEYS.landmarks), get<string>(KEYS.skin),
        get<{ width: number; height: number }>(KEYS.size),
      ]);
      if (!(photo instanceof Blob) || !(landmarks instanceof Float32Array) || landmarks.length < LANDMARK_COUNT * 3) return;
      const mine = token;
      const bitmap = await decode(photo);
      if (mine !== token || read().status !== 'idle') return bitmap.close();
      update({
        status: 'ready-to-bake',
        photoBlob: photo,
        bitmap,
        landmarks,
        imageSize: size ?? { width: bitmap.width, height: bitmap.height },
        skinToneHex: typeof skin === 'string' ? skin : undefined,
        hints: [],
      });
    } catch {
      /* nothing stored or storage unavailable */
    }
  },

  ensureBaked: async () => {
    await read().hydrate();
    if (read().status === 'ready-to-bake' && !read().overlayCanvas) await read().bake();
  },

  clear: async () => {
    token += 1;
    read().bitmap?.close();
    update((state) => ({ ...initial, revision: state.revision + 1 }));
    try {
      await Promise.all(Object.values(KEYS).map((key) => del(key)));
    } catch {
      /* ignore */
    }
  },
}));
