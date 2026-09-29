import { skinTones, type AvatarMaterialMode } from '../../store/appearanceStore';

export const SKIN_MAP_SIZE = 2048;

/** The part of CanvasRenderingContext2D the compositor uses (lets tests run without a DOM). */
export interface CompositeContext {
  fillStyle: string | CanvasGradient | CanvasPattern;
  fillRect(x: number, y: number, width: number, height: number): void;
  drawImage(image: never, dx: number, dy: number): void;
}

export interface SkinColorInput {
  mode: AvatarMaterialMode;
  toneIndex: number;
  /** "Use the skin tone estimated from the photo" toggle. */
  useFaceTone: boolean;
  faceToneHex?: string;
}

/** Base skin colour (sRGB hex): the photo tone when a face is baked and the toggle is on, else the preset tone. */
export function resolveSkinHex(input: SkinColorInput): string {
  if (input.useFaceTone && input.faceToneHex) return input.faceToneHex;
  return (skinTones[input.toneIndex] ?? skinTones[1]!).color;
}

/**
 * Fills the whole skin map with the skin tone, then alpha-composites the face overlay (2048 x 2048, straight alpha,
 * body UV layout) on top of it.
 */
export function compositeSkinMap(
  context: CompositeContext,
  size: number,
  toneHex: string,
  overlay?: CanvasImageSource,
): void {
  context.fillStyle = toneHex;
  context.fillRect(0, 0, size, size);
  if (overlay) context.drawImage(overlay as never, 0, 0);
}
