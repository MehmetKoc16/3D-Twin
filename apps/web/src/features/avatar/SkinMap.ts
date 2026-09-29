import { CanvasTexture, SRGBColorSpace } from 'three';
import { compositeSkinMap, SKIN_MAP_SIZE } from './skinComposite';

/**
 * Owns the composite skin canvas (2048^2: skin tone + face overlay) and its CanvasTexture. The canvas is reused
 * across updates; `dispose()` releases the GPU texture.
 */
export class SkinMap {
  readonly canvas: HTMLCanvasElement;
  readonly texture: CanvasTexture;

  constructor(size = SKIN_MAP_SIZE) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = size;
    this.canvas.height = size;
    this.texture = new CanvasTexture(this.canvas);
    this.texture.flipY = false; // glTF UV convention (v down)
    this.texture.colorSpace = SRGBColorSpace;
    this.texture.anisotropy = 4;
  }

  update(toneHex: string, overlay?: CanvasImageSource): void {
    const context = this.canvas.getContext('2d');
    if (!context) throw new Error('2D canvas is not available');
    compositeSkinMap(context, this.canvas.width, toneHex, overlay);
    this.texture.needsUpdate = true;
  }

  dispose(): void {
    this.texture.dispose();
  }
}
