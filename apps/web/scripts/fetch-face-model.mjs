// Downloads the MediaPipe Face Landmarker model into apps/web/public/models/ (gitignored via *.task).
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const url =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const target = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'models', 'face_landmarker.task');

if (existsSync(target) && !process.argv.includes('--force')) {
  console.log(`Model already present: ${target}`);
  process.exit(0);
}
mkdirSync(dirname(target), { recursive: true });
const response = await fetch(url);
if (!response.ok) {
  console.error(`Download failed: ${response.status} ${response.statusText}`);
  process.exit(1);
}
writeFileSync(target, Buffer.from(await response.arrayBuffer()));
console.log(`Saved ${target}`);
