import type { FaceMapDef } from '@dt/avatar-core';

/**
 * Small hand-made FaceMapDef for tests: 468 landmarks laid out on an 18 x 26 grid inside
 * u in [0.4, 0.6], v in [0.05, 0.25]. It is topologically a flat sheet, not a real face.
 */
export function makeMockFaceMap(): FaceMapDef {
  const cols = 18;
  const rows = 26;
  const u0 = 0.4;
  const u1 = 0.6;
  const v0 = 0.05;
  const v1 = 0.25;
  const landmarks: FaceMapDef['landmarks'] = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      landmarks.push({
        index: r * cols + c,
        tri: [0, 1, 2],
        bary: [1, 0, 0],
        uv: [u0 + ((u1 - u0) * c) / (cols - 1), v0 + ((v1 - v0) * r) / (rows - 1)],
      });
    }
  }
  const triangles: [number, number, number][] = [];
  for (let r = 0; r < rows - 1; r += 1) {
    for (let c = 0; c < cols - 1; c += 1) {
      const a = r * cols + c;
      triangles.push([a, a + 1, a + cols], [a + 1, a + cols + 1, a + cols]);
    }
  }
  const oval: number[] = [];
  for (let c = 0; c < cols; c += 1) oval.push(c);
  for (let r = 1; r < rows; r += 1) oval.push(r * cols + cols - 1);
  for (let c = cols - 2; c >= 0; c -= 1) oval.push((rows - 1) * cols + c);
  for (let r = rows - 2; r >= 1; r -= 1) oval.push(r * cols);
  const at = (r: number, c: number) => r * cols + c;
  return {
    version: 1,
    source: { mediapipeCommit: 'mock', license: 'Apache-2.0' },
    triangles,
    landmarks,
    faceOval: oval,
    regions: {
      leftEye: [at(8, 4), at(8, 5), at(9, 4), at(9, 5)],
      rightEye: [at(8, 12), at(8, 13), at(9, 12), at(9, 13)],
      lips: [at(19, 8), at(19, 9), at(20, 8), at(20, 9)],
      leftCheek: [at(13, 3), at(13, 4), at(14, 3), at(14, 4)],
      rightCheek: [at(13, 13), at(13, 14), at(14, 13), at(14, 14)],
      forehead: [at(2, 8), at(2, 9), at(3, 8), at(3, 9)],
    },
    uvBounds: { min: [u0, v0], max: [u1, v1] },
    fitModifiers: [],
  };
}
