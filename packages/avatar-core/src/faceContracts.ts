import type { Vec3 } from './contracts';

/**
 * Binding of one MediaPipe Face Landmarker canonical landmark (0..467, irises excluded)
 * to a point on the MakeHuman body surface. Produced by the asset pipeline (face-map.json).
 */
export interface FaceMapLandmark {
  /** MediaPipe landmark index (0..467). */
  index: number;
  /** Render-vertex triangle (combined vertex space) containing the bound surface point. */
  tri: [number, number, number];
  /** Barycentric weights of the bound point inside `tri` (sum = 1). */
  bary: Vec3;
  /** Body texture coordinate of the bound point, same convention as base.glb TEXCOORD_0. */
  uv: [number, number];
}

/** Contents of `apps/web/public/assets/body/face-map.json`. */
export interface FaceMapDef {
  version: 1;
  source: { mediapipeCommit: string; license: 'Apache-2.0' };
  /** Canonical face mesh triangles as landmark-index triples (from canonical_face_model.obj). */
  triangles: [number, number, number][];
  landmarks: FaceMapLandmark[];
  /** Ordered landmark indices of the face oval, used as the photo mask boundary. */
  faceOval: number[];
  /** Landmark index groups; left/right are the SUBJECT's own left/right. */
  regions: {
    leftEye: number[];
    rightEye: number[];
    lips: number[];
    leftCheek: number[];
    rightCheek: number[];
    forehead: number[];
  };
  /** UV-space bounds of the face region on the body texture (TEXCOORD_0 convention). */
  uvBounds: { min: [number, number]; max: [number, number] };
  /** Face-shape modifier ids (present in manifest.modifiers) the runtime may fit to photo landmarks. */
  fitModifiers: string[];
}
