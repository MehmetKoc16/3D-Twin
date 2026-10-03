import * as THREE from 'three';
import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  DynamicDrawUsage,
  Float32BufferAttribute,
  Group,
  Mesh,
  MeshDepthMaterial,
  MeshStandardMaterial,
  NoColorSpace,
  PropertyBinding,
  RGBADepthPacking,
  SkinnedMesh,
  Sphere,
  Vector3,
  type InterleavedBufferAttribute,
  type Matrix4,
  type Object3D,
  type Skeleton,
  type Texture,
} from 'three';
import type { Vec3 } from '@dt/avatar-core';
import {
  applyHairShader,
  type HairRendererLike,
  type HairShaderHandle,
} from '../../vendor/threejs-hair-shader/hair-shader.js';
import { normalizeSkinWeights, remapSkinIndices, translatePositions } from './twinBinding';
import { TwinFormatError } from './twinDef';

/**
 * Strand hair of the hybrid twin (contract `rcov-groot-bvar/1`, see docs/ARCHITECTURE.md): a SEPARATE skinned node
 * named by `asset.extras.dtHairNode`, rendered with the vendored MIT hair-card shader. The body mesh never carries
 * hair; garments, hands and the opening repair never see this mesh.
 */

export const HAIR_FORMAT = 'rcov-groot-bvar/1';

/** `asset.extras.dtHairNode` plus the hair material's `extras.dtHair`, validated. */
export interface TwinHairInfo {
  nodeName: string;
  format: typeof HAIR_FORMAT;
  /** sRGB hex of the base hair colour (`color` of the shader). */
  colorHex: string;
  rootHex: string | null;
  tipHex: string | null;
  cardCount: number | null;
}

interface HairJson {
  asset?: { extras?: { dtHairNode?: unknown } };
  nodes?: unknown[];
  meshes?: unknown[];
  materials?: unknown[];
}

const record = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

const HEX = /^#[\da-f]{6}$/i;

function optionalHex(source: Record<string, unknown>, key: 'rootHex' | 'tipHex'): string | null {
  const value = source[key];
  if (value === undefined) return null;
  if (typeof value !== 'string' || !HEX.test(value))
    throw new TwinFormatError('json', `dtHair.${key} must be #rrggbb`);
  return value;
}

/** Materials used by the primitives of the glTF node called `nodeName` (null when there is no such node). */
function nodeMaterials(json: HairJson, nodeName: string): unknown[] | null {
  const node = (json.nodes ?? []).find((n) => record(n) && n.name === nodeName);
  if (!record(node)) return null;
  const mesh = Number.isSafeInteger(node.mesh) ? (json.meshes ?? [])[node.mesh as number] : null;
  const primitives = record(mesh) && Array.isArray(mesh.primitives) ? mesh.primitives : [];
  return primitives.flatMap((p) =>
    record(p) && Number.isSafeInteger(p.material)
      ? [(json.materials ?? [])[p.material as number]]
      : [],
  );
}

/**
 * Reads `asset.extras.dtHairNode` and the hair material's `extras.dtHair`. Null for bundles without hair (they behave
 * exactly as before); `TwinFormatError('json')` for anything malformed.
 */
export function parseTwinHair(json: HairJson): TwinHairInfo | null {
  const nodeName = json.asset?.extras?.dtHairNode;
  if (nodeName === undefined) return null;
  if (typeof nodeName !== 'string' || nodeName.length === 0)
    throw new TwinFormatError('json', 'asset.extras.dtHairNode must be a non-empty string');
  const materials = nodeMaterials(json, nodeName);
  if (materials === null)
    throw new TwinFormatError('json', `dtHairNode "${nodeName}" is not a node of the glTF`);
  const raw = materials
    .map((m) => (record(m) && record(m.extras) ? m.extras.dtHair : undefined))
    .find((extras) => extras !== undefined);
  if (!record(raw)) throw new TwinFormatError('json', 'the hair material needs extras.dtHair');
  if (raw.format !== HAIR_FORMAT)
    throw new TwinFormatError('json', `dtHair.format must be "${HAIR_FORMAT}"`);
  if (typeof raw.colorHex !== 'string' || !HEX.test(raw.colorHex))
    throw new TwinFormatError('json', 'dtHair.colorHex must be #rrggbb');
  const cardCount = raw.cardCount;
  if (
    cardCount !== undefined &&
    (typeof cardCount !== 'number' || !Number.isSafeInteger(cardCount) || cardCount < 0)
  )
    throw new TwinFormatError('json', 'dtHair.cardCount must be a non-negative integer');
  return {
    nodeName,
    format: HAIR_FORMAT,
    colorHex: raw.colorHex,
    rootHex: optionalHex(raw, 'rootHex'),
    tipHex: optionalHex(raw, 'tipHex'),
    cardCount: cardCount === undefined ? null : cardCount,
  };
}

/** The hair mesh of a parsed twin, copied out of the glTF and re-indexed to the avatar's bone order. */
export interface TwinHairModel {
  info: TwinHairInfo;
  vertexCount: number;
  /** Rest positions in the twin's own frame (like the body), 3 floats per vertex. */
  position: Float32Array;
  normal: Float32Array;
  uv: Float32Array;
  index: Uint32Array;
  skinIndex: Uint16Array;
  skinWeight: Float32Array;
  /** Strand data atlas, `NoColorSpace` (R coverage, G root-to-tip, B variation). */
  atlas: Texture;
  /** Frees the atlas and the loader's geometry and material. */
  dispose(): void;
}

function copyAttribute(attribute: BufferAttribute | InterleavedBufferAttribute): Float32Array {
  const size = attribute.itemSize;
  const out = new Float32Array(attribute.count * size);
  for (let i = 0; i < attribute.count; i++) {
    out[i * size] = attribute.getX(i);
    if (size > 1) out[i * size + 1] = attribute.getY(i);
    if (size > 2) out[i * size + 2] = attribute.getZ(i);
    if (size > 3) out[i * size + 3] = attribute.getW(i);
  }
  return out;
}

/** SkinnedMeshes at or below the glTF node called `nodeName` (GLTFLoader sanitises node names). */
export function hairSkinnedMeshes(root: Object3D, nodeName: string): SkinnedMesh[] {
  const node = root.getObjectByName(PropertyBinding.sanitizeNodeName(nodeName));
  const found: SkinnedMesh[] = [];
  node?.traverse((o) => {
    if (o instanceof SkinnedMesh) found.push(o);
  });
  return found;
}

/**
 * Remap hair joint -> avatar bone. The hair shares the body's skin in the contract, but a hair skin that lists only some
 * bones (or in another order) is accepted as long as every joint exists in the avatar and its inverse bind matrix
 * equals the body's for the same bone name: hair and body then share one rest pose.
 */
export function hairBoneRemap(
  hair: Skeleton,
  body: Skeleton,
  appNames: readonly string[],
  tolerance = 1e-4,
): Int32Array {
  const appIndex = new Map(appNames.map((n, i) => [n, i]));
  const bodyIndex = new Map(body.bones.map((b, i) => [b.name, i]));
  const remap = new Int32Array(hair.bones.length);
  hair.bones.forEach((bone, i) => {
    const app = appIndex.get(bone.name);
    const bodyBone = bodyIndex.get(bone.name);
    if (app === undefined || bodyBone === undefined)
      throw new TwinFormatError(
        'bones',
        `dtHair bone "${bone.name}" does not exist in the twin skeleton`,
      );
    const a = hair.boneInverses[i]?.elements;
    const b = body.boneInverses[bodyBone]?.elements;
    if (!a || !b || a.some((v, k) => Math.abs(v - b[k]!) > tolerance))
      throw new TwinFormatError(
        'mismatch',
        `dtHair bone "${bone.name}" has another rest pose than the twin body`,
      );
    remap[i] = app;
  });
  if (new Set(remap).size !== remap.length)
    throw new TwinFormatError('bones', 'duplicate bone names in the dtHair skin');
  return remap;
}

/** Builds the hair model from the loader's hair mesh. The hair atlas is the material's base-colour texture. */
export function extractTwinHair(
  info: TwinHairInfo,
  mesh: SkinnedMesh,
  body: SkinnedMesh,
  appNames: readonly string[],
): TwinHairModel {
  const geometry = mesh.geometry;
  const index = geometry.getIndex();
  const position = geometry.getAttribute('position');
  const skinIndex = geometry.getAttribute('skinIndex');
  const skinWeight = geometry.getAttribute('skinWeight');
  const uv = geometry.getAttribute('uv');
  if (!index || !position || !skinIndex || !skinWeight || !uv)
    throw new TwinFormatError(
      'noSkin',
      'the dtHair mesh needs POSITION, TEXCOORD_0, JOINTS_0, WEIGHTS_0 and indices',
    );
  const source = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  const atlas = source instanceof MeshStandardMaterial ? source.map : null;
  if (!atlas)
    throw new TwinFormatError('json', 'the dtHair material needs a baseColorTexture atlas');
  const remap = hairBoneRemap(mesh.skeleton, body.skeleton, appNames);
  if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
  const weights = copyAttribute(skinWeight);
  normalizeSkinWeights(weights);
  // GLTFLoader tags base-colour textures sRGB; the atlas is data. Set before the first GPU upload.
  atlas.colorSpace = NoColorSpace;
  return {
    info,
    vertexCount: position.count,
    position: copyAttribute(position),
    normal: copyAttribute(geometry.getAttribute('normal')),
    uv: copyAttribute(uv),
    index: Uint32Array.from(index.array as ArrayLike<number>),
    skinIndex: remapSkinIndices(copyAttribute(skinIndex), remap),
    skinWeight: weights,
    atlas,
    dispose: () => {
      atlas.dispose();
      source?.dispose();
      geometry.dispose();
    },
  };
}

/** Coverage (atlas R) below this does not cast a shadow. */
export const HAIR_SHADOW_COVERAGE = 0.2;

interface GlSamples {
  SAMPLES: number;
  getParameter(parameter: number): unknown;
}

/** True when the renderer's drawing buffer is multisampled (the shader's own test: `gl.SAMPLES > 0`). */
export function rendererHasMsaa(renderer: HairRendererLike | null | undefined): boolean {
  try {
    const gl = renderer?.getContext?.() as GlSamples | undefined;
    if (!gl || typeof gl.getParameter !== 'function') return false;
    const samples = gl.getParameter(gl.SAMPLES);
    return typeof samples === 'number' && samples > 0;
  } catch {
    return false;
  }
}

/** Shadow caster for hair cards: cuts out on atlas R (three's own alphaMap reads G, the root-to-tip channel). */
export function hairDepthMaterial(atlas: Texture): MeshDepthMaterial {
  const material = new MeshDepthMaterial({
    depthPacking: RGBADepthPacking,
    alphaMap: atlas,
    side: DoubleSide,
  });
  material.name = 'dtHair:shadow';
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <alphamap_fragment>',
      `#ifdef USE_ALPHAMAP
  if ( texture2D( alphaMap, vAlphaMapUv ).r < ${HAIR_SHADOW_COVERAGE.toFixed(3)} ) discard;
#endif`,
    );
  };
  material.customProgramCacheKey = () => 'dt-hair-shadow';
  return material;
}

export interface TwinHairOptions {
  /** The renderer, used to detect MSAA (alpha-to-coverage path) when `msaa` is not given. */
  renderer?: HairRendererLike | null;
  /** Force the render path: true = alpha-to-coverage + blended fringe, false = alpha-tested core + blended outer. */
  msaa?: boolean;
}

/**
 * The hair in the avatar scene: a SkinnedMesh bound to the avatar's skeleton exactly like the twin body (and translated
 * by the same rest alignment), shaded by the vendored hair shader. The shader adds a core or fringe mesh next to it that
 * shares the geometry and skeleton; everything lives in one group, removed and disposed with `dispose`.
 */
export class TwinHair {
  readonly group = new Group();
  readonly mesh: SkinnedMesh;
  readonly shader: HairShaderHandle;
  /** True when the alpha-to-coverage (MSAA) path is used. */
  readonly msaa: boolean;
  private readonly geometry = new BufferGeometry();
  private readonly position: BufferAttribute;
  private readonly depth: MeshDepthMaterial;
  private disposed = false;

  constructor(
    parent: Object3D,
    skeleton: Skeleton,
    bindMatrix: Matrix4,
    private readonly model: TwinHairModel,
    options: TwinHairOptions = {},
  ) {
    this.msaa = options.msaa ?? rendererHasMsaa(options.renderer);
    this.position = new Float32BufferAttribute(new Float32Array(model.position), 3).setUsage(
      DynamicDrawUsage,
    );
    this.geometry.setAttribute('position', this.position);
    this.geometry.setAttribute('normal', new Float32BufferAttribute(model.normal, 3));
    this.geometry.setAttribute('uv', new Float32BufferAttribute(model.uv, 2));
    this.geometry.setAttribute('skinIndex', new BufferAttribute(model.skinIndex, 4));
    this.geometry.setAttribute('skinWeight', new BufferAttribute(model.skinWeight, 4));
    this.geometry.setIndex(new BufferAttribute(Uint32Array.from(model.index), 1));
    const placeholder = new MeshStandardMaterial({ name: 'dtHair', side: DoubleSide });
    this.mesh = new SkinnedMesh(this.geometry, placeholder);
    this.mesh.name = 'twin:hair';
    this.mesh.frustumCulled = false; // bounds of a posed skinned mesh are not tracked
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.mesh.boundingSphere = new Sphere(new Vector3(0, 0.95, 0), 2);
    this.mesh.bind(skeleton, bindMatrix);
    this.group.name = 'twin:hair:group';
    this.group.visible = false; // shown with the body, after the first aligned solve
    this.group.add(this.mesh);
    parent.add(this.group);

    const { info } = model;
    const multi = info.rootHex !== null;
    // The shader runs root -> tip between `rootColor` and `color`; with only colorHex it is one colour.
    this.shader = applyHairShader(this.group, {
      THREE,
      ...(options.renderer ? { renderer: options.renderer } : {}),
      atlas: model.atlas,
      color: multi ? (info.tipHex ?? info.colorHex) : info.colorHex,
      ...(multi ? { rootMode: 'multi' as const, rootColor: info.rootHex! } : {}),
      alphaToCoverage: this.msaa,
    });
    this.depth = hairDepthMaterial(model.atlas);
    this.group.traverse((o) => {
      if (!(o instanceof Mesh)) return;
      o.frustumCulled = false;
      // Only the main pass casts (cut-out); the core / fringe passes would double or block it.
      const casts = o === this.mesh;
      o.castShadow = casts;
      if (casts) o.customDepthMaterial = this.depth;
    });
  }

  /** Every mesh of the hair (main pass first, then the core or fringe pass the shader added). */
  get meshes(): SkinnedMesh[] {
    const out: SkinnedMesh[] = [];
    this.group.traverse((o) => {
      if (o instanceof SkinnedMesh) out.push(o);
    });
    return out;
  }

  get visible(): boolean {
    return this.group.visible;
  }

  /** Follows the twin body: translates the rest positions onto the avatar's rest heads and takes its bind matrix. */
  onSolve(offset: Vec3, bindMatrix: Matrix4): void {
    if (this.disposed) return;
    translatePositions(this.model.position, offset, this.position.array as Float32Array);
    this.position.needsUpdate = true;
    for (const mesh of this.meshes) {
      mesh.bindMatrix.copy(bindMatrix);
      mesh.bindMatrixInverse.copy(bindMatrix).invert();
    }
    this.group.visible = true;
  }

  /** Removes the group and frees the shader passes, materials, geometry and the atlas. Safe to call twice. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.shader.dispose();
    this.depth.dispose();
    this.geometry.dispose();
    this.group.removeFromParent();
    this.model.dispose();
  }
}
