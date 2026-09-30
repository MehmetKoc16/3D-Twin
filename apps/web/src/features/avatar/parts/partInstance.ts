import {
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Color,
  DynamicDrawUsage,
  Float32BufferAttribute,
  MeshStandardMaterial,
  SkinnedMesh,
  SRGBColorSpace,
  type Matrix4,
  type Skeleton,
} from 'three';
import { bindGarment } from '@dt/avatar-core';
import { computeWeldedNormals } from '../meshMath';
import { recolorIris } from './irisRecolor';
import type { PartRuntime } from './partsRuntime';

/** Largest channel of a tint colour (linear); tone mapping absorbs the overshoot of light colours on dark textures. */
const MAX_TINT = 2.5;

/** Colour multiplier that makes a neutral hair / brow texture show `hex` on average. */
export function tintColor(hex: string, textureMean: number, target: Color = new Color()): Color {
  target.set(hex);
  target.r = Math.min(MAX_TINT, target.r / textureMean);
  target.g = Math.min(MAX_TINT, target.g / textureMean);
  target.b = Math.min(MAX_TINT, target.b / textureMean);
  return target;
}

/** Owns the recoloured eye texture (a copy of the original on a canvas) and its CanvasTexture. */
class IrisTexture {
  readonly texture: CanvasTexture;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly out: ImageData;
  /** The recoloured texture canvas (exposed for the dev / e2e probe). */
  get image(): HTMLCanvasElement {
    return this.canvas;
  }
  private colorHex = '';

  constructor(
    private readonly source: NonNullable<PartRuntime['texturePixels']>,
    private readonly iris: { center: readonly [number, number]; radius: number },
  ) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = source.width;
    this.canvas.height = source.height;
    const context = this.canvas.getContext('2d');
    if (!context) throw new Error('2D canvas is not available');
    this.context = context;
    this.out = context.createImageData(source.width, source.height);
    this.texture = new CanvasTexture(this.canvas);
    this.texture.flipY = false; // glTF convention (v down), same as the source texture
    this.texture.colorSpace = SRGBColorSpace;
    this.texture.anisotropy = 8;
  }

  setColor(hex: string): void {
    if (hex === this.colorHex) return;
    this.colorHex = hex;
    recolorIris(this.source.data, this.out.data, this.source.width, this.source.height, this.iris, hex);
    this.context.putImageData(this.out, 0, 0);
    this.texture.needsUpdate = true;
  }

  dispose(): void {
    this.texture.dispose();
  }
}

/** One mounted part: a SkinnedMesh on the avatar skeleton, re-bound to the solved body after every solve. */
export class PartInstance {
  readonly mesh: SkinnedMesh;
  private readonly geometry = new BufferGeometry();
  private readonly material: MeshStandardMaterial;
  private readonly rest: Float32Array;
  private readonly normalScratch: Float32Array;
  private readonly iris: IrisTexture | null = null;
  private fitted = false;

  constructor(
    readonly runtime: PartRuntime,
    skeleton: Skeleton,
    bindMatrix: Matrix4,
  ) {
    const def = runtime.def;
    const n = runtime.vertexCount;
    this.rest = new Float32Array(n * 3);
    this.normalScratch = new Float32Array(runtime.weld.groupCount * 3);
    this.geometry.setAttribute('position', new Float32BufferAttribute(new Float32Array(n * 3), 3).setUsage(DynamicDrawUsage));
    this.geometry.setAttribute('normal', new Float32BufferAttribute(new Float32Array(n * 3), 3).setUsage(DynamicDrawUsage));
    this.geometry.setAttribute('uv', new Float32BufferAttribute(runtime.uv, 2));
    this.geometry.setAttribute('skinIndex', new BufferAttribute(runtime.skinIndices, 4));
    this.geometry.setAttribute('skinWeight', new BufferAttribute(runtime.skinWeights, 4));
    this.geometry.setIndex(new BufferAttribute(Uint32Array.from(runtime.index), 1));

    this.material = runtime.material.clone();
    if (def.material.alphaMode === 'MASK') {
      // alpha-tested cards stay in the opaque pass (depth write, no sorting); MSAA coverage softens the strand edges
      this.material.transparent = false;
      this.material.alphaToCoverage = true;
      this.material.alphaTest = def.material.alphaCutoff ?? 0.5;
    }
    if (def.category === 'eyebrows' || def.category === 'eyelashes') {
      // brows and lashes lie a fraction of a millimetre above the skin
      this.material.polygonOffset = true;
      this.material.polygonOffsetFactor = -2;
      this.material.polygonOffsetUnits = -2;
    }
    if (def.category === 'eyes' && runtime.texturePixels && def.irisUv) {
      this.iris = new IrisTexture(runtime.texturePixels, def.irisUv);
      this.material.map = this.iris.texture;
      this.material.color.set('#ffffff');
      this.material.needsUpdate = true;
    }

    this.mesh = new SkinnedMesh(this.geometry, this.material);
    this.mesh.name = `part:${def.id}`;
    this.mesh.frustumCulled = false; // bounds of a posed skinned mesh are not tracked
    this.mesh.castShadow = def.category === 'hair';
    this.mesh.receiveShadow = def.category === 'hair';
    this.mesh.visible = false; // until the first solve was applied
    this.mesh.bind(skeleton, bindMatrix);
  }

  get id(): string {
    return this.runtime.def.id;
  }

  /** Follows the body rest bind matrix after the skeleton was rebuilt. */
  syncBind(bindMatrix: Matrix4): void {
    this.mesh.bindMatrix.copy(bindMatrix);
    this.mesh.bindMatrixInverse.copy(bindMatrix).invert();
  }

  /** Hair / brow tint (`tintable` parts) or iris colour (eyes). Lashes are not tintable. */
  setColor(hex: string): void {
    const def = this.runtime.def;
    if (!def.material.tintable) return;
    if (this.iris) this.iris.setColor(hex);
    else tintColor(hex, this.runtime.textureMean, this.material.color);
  }

  /** The recoloured iris texture canvas of an eye part (dev / e2e probe), null for other parts. */
  get irisCanvas(): HTMLCanvasElement | null {
    return this.iris?.image ?? null;
  }

  /** Whether at least one solve was applied (the mesh is invisible before, its geometry is all zeros). */
  get isFitted(): boolean {
    return this.fitted;
  }

  /** Re-binds the part to the solved, grounded body render positions and refreshes the normals. */
  update(body: Float32Array): void {
    const { runtime, rest } = this;
    bindGarment(runtime.binding, runtime.def.scaleRefs, body, rest);
    const position = this.geometry.getAttribute('position');
    (position.array as Float32Array).set(rest);
    position.needsUpdate = true;
    const normal = this.geometry.getAttribute('normal');
    computeWeldedNormals(rest, runtime.index, runtime.weld, normal.array as Float32Array, this.normalScratch);
    normal.needsUpdate = true;
    this.mesh.visible = true;
    this.fitted = true;
  }

  dispose(): void {
    this.mesh.removeFromParent();
    this.geometry.dispose();
    this.iris?.dispose();
    this.material.dispose(); // the glb texture belongs to the runtime and stays
  }
}
