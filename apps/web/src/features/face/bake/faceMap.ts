import type { FaceMapDef } from '@dt/avatar-core';

export const FACE_MAP_URL = `${import.meta.env.BASE_URL}assets/body/face-map.json`;

export class FaceMapUnavailableError extends Error {
  constructor() {
    super('face-map.json is not available');
    this.name = 'FaceMapUnavailableError';
  }
}

let cached: Promise<FaceMapDef> | undefined;

function isFaceMap(value: unknown): value is FaceMapDef {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Partial<FaceMapDef>;
  return v.version === 1 && Array.isArray(v.triangles) && Array.isArray(v.landmarks) && Array.isArray(v.faceOval);
}

/** Fetches (once) the static face map; throws FaceMapUnavailableError if missing or malformed. */
export function loadFaceMap(): Promise<FaceMapDef> {
  cached ??= (async () => {
    const response = await fetch(FACE_MAP_URL);
    if (!response.ok) throw new FaceMapUnavailableError();
    const type = response.headers.get('content-type') ?? '';
    if (type.includes('text/html')) throw new FaceMapUnavailableError();
    const json: unknown = await response.json();
    if (!isFaceMap(json)) throw new FaceMapUnavailableError();
    return json;
  })().catch((error: unknown) => {
    cached = undefined;
    throw error instanceof FaceMapUnavailableError ? error : new FaceMapUnavailableError();
  });
  return cached;
}
