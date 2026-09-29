import { at } from './util';
/** MediaPipe face oval landmark ring (preview drawing only; baking uses faceMap.faceOval). */
export const PREVIEW_FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150,
  136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

/** Draws the photo with landmark dots and the face oval onto a canvas. */
export function drawPreview(
  canvas: HTMLCanvasElement,
  bitmap: ImageBitmap,
  landmarks: ArrayLike<number> | undefined,
  maxSide = 640,
): void {
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const context = canvas.getContext('2d');
  if (!context) return;
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  if (!landmarks || landmarks.length < 468 * 3) return;
  context.fillStyle = 'rgba(56, 189, 248, 0.85)';
  for (let i = 0; i < 468; i += 1) {
    context.fillRect(at(landmarks, i * 3) * canvas.width - 0.75, at(landmarks, i * 3 + 1) * canvas.height - 0.75, 1.5, 1.5);
  }
  context.strokeStyle = 'rgba(250, 204, 21, 0.95)';
  context.lineWidth = 2;
  context.beginPath();
  PREVIEW_FACE_OVAL.forEach((index, n) => {
    const x = at(landmarks, index * 3) * canvas.width;
    const y = at(landmarks, index * 3 + 1) * canvas.height;
    if (n === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.closePath();
  context.stroke();
}
