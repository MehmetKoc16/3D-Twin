import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  LinearFilter,
  LinearSRGBColorSpace,
  Mesh,
  MeshBasicMaterial,
  NoColorSpace,
  OrthographicCamera,
  Scene,
  Texture,
  WebGLRenderer,
} from 'three';
import type { BakeGeometryData } from './assemble';

/**
 * Renders the photo, warped piecewise-affinely into body-UV space, using an offscreen WebGL context.
 * Pixel (x, y) of the result corresponds to (u * size, v * size) with v measured from the top.
 * Colours are passed through untouched (sRGB bytes in, sRGB bytes out); uncovered pixels are transparent.
 */
export function warpPhotoToUv(
  image: ImageBitmap | HTMLCanvasElement,
  geometry: BakeGeometryData,
  size: number,
): HTMLCanvasElement {
  const renderer = new WebGLRenderer({
    antialias: false,
    alpha: true,
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
  });
  const geometryObject = new BufferGeometry();
  const material = new MeshBasicMaterial({ side: DoubleSide, toneMapped: false });
  try {
    renderer.setPixelRatio(1);
    renderer.setSize(size, size, false);
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = LinearSRGBColorSpace;

    let source: ImageBitmap | HTMLCanvasElement = image;
    const maxTexture = renderer.capabilities.maxTextureSize;
    if (Math.max(image.width, image.height) > maxTexture) {
      const scale = maxTexture / Math.max(image.width, image.height);
      const canvas = document.createElement('canvas');
      canvas.width = Math.floor(image.width * scale);
      canvas.height = Math.floor(image.height * scale);
      canvas.getContext('2d')?.drawImage(image, 0, 0, canvas.width, canvas.height);
      source = canvas;
    }
    const texture = new Texture(source);
    texture.flipY = false;
    texture.colorSpace = NoColorSpace;
    texture.generateMipmaps = false;
    texture.minFilter = LinearFilter;
    texture.magFilter = LinearFilter;
    texture.needsUpdate = true;
    material.map = texture;

    geometryObject.setAttribute('position', new BufferAttribute(geometry.positions, 3));
    geometryObject.setAttribute('uv', new BufferAttribute(geometry.photoUvs, 2));
    geometryObject.setIndex(new BufferAttribute(geometry.indices, 1));

    const scene = new Scene();
    scene.add(new Mesh(geometryObject, material));
    // left, right, top, bottom: y = 0 is the top row of the framebuffer (v measured from the top).
    const camera = new OrthographicCamera(0, size, 0, size, -1, 1);
    renderer.render(scene, camera);

    const output = document.createElement('canvas');
    output.width = size;
    output.height = size;
    const context = output.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('2D canvas is not available');
    context.drawImage(renderer.domElement, 0, 0);
    texture.dispose();
    return output;
  } finally {
    geometryObject.dispose();
    material.dispose();
    renderer.dispose();
    renderer.forceContextLoss();
  }
}
