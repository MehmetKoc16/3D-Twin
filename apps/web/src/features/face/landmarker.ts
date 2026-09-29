import { FaceLandmarker, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';
// The package `exports` map hides the wasm folder from bare specifiers; `@mediapipe-wasm` is a Vite alias
// (vite.config.ts) to the wasm folder of the installed package, wherever npm put it.
import wasmLoaderPath from '@mediapipe-wasm/vision_wasm_internal.js?url';
import wasmBinaryPath from '@mediapipe-wasm/vision_wasm_internal.wasm?url';

export const LOCAL_MODEL_URL = '/models/face_landmarker.task';
export const REMOTE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

let instance: Promise<FaceLandmarker> | undefined;

async function fetchLocalModel(): Promise<Uint8Array | undefined> {
  try {
    const response = await fetch(LOCAL_MODEL_URL);
    if (!response.ok) return undefined;
    if ((response.headers.get('content-type') ?? '').includes('text/html')) return undefined;
    const bytes = new Uint8Array(await response.arrayBuffer());
    return bytes.length > 100_000 ? bytes : undefined;
  } catch {
    return undefined;
  }
}

/** Lazily creates the singleton Face Landmarker (IMAGE mode). Only static wasm/model files are fetched. */
export function getFaceLandmarker(): Promise<FaceLandmarker> {
  instance ??= (async () => {
    const fileset = { wasmLoaderPath, wasmBinaryPath };
    const options = {
      runningMode: 'IMAGE' as const,
      numFaces: 2, // 2 so that "multiple faces" can be reported; the largest face is used
      outputFaceBlendshapes: false,
    };
    const local = await fetchLocalModel();
    if (local) {
      return FaceLandmarker.createFromOptions(fileset, {
        ...options,
        baseOptions: { modelAssetBuffer: local, delegate: 'CPU' },
      });
    }
    return FaceLandmarker.createFromOptions(fileset, {
      ...options,
      baseOptions: { modelAssetPath: REMOTE_MODEL_URL, delegate: 'CPU' },
    });
  })().catch((error: unknown) => {
    instance = undefined;
    throw error;
  });
  return instance;
}

export interface DetectedFaces {
  faceCount: number;
  /** Landmarks of the largest face: 478 * 3 floats (normalized x, y, z). */
  landmarks?: Float32Array;
}

export function flattenLandmarks(points: { x: number; y: number; z: number }[]): Float32Array {
  const out = new Float32Array(points.length * 3);
  points.forEach((p, i) => {
    out[i * 3] = p.x;
    out[i * 3 + 1] = p.y;
    out[i * 3 + 2] = p.z;
  });
  return out;
}

export function pickLargestFace(result: FaceLandmarkerResult): DetectedFaces {
  const faces = result.faceLandmarks;
  if (faces.length === 0) return { faceCount: 0 };
  let best = faces[0];
  let bestArea = -1;
  for (const face of faces) {
    const xs = face.map((p) => p.x);
    const ys = face.map((p) => p.y);
    const area = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
    if (area > bestArea) {
      bestArea = area;
      best = face;
    }
  }
  return { faceCount: faces.length, landmarks: flattenLandmarks(best ?? []) };
}

export async function detectFace(image: ImageBitmap): Promise<DetectedFaces> {
  const landmarker = await getFaceLandmarker();
  return pickLargestFace(landmarker.detect(image));
}
