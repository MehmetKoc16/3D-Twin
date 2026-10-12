import { Mesh, MeshStandardMaterial, type BufferAttribute, type Object3D } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import {
  garmentSkinWeights,
  parseGarmentBinding,
  parseGarmentRigidAttachments,
  validateGarmentLicense,
  type GarmentBinding,
  type GarmentRigidAttachments,
  type GarmentTemplateDef,
} from '@dt/avatar-core';
import { buildWeldMap, type WeldMap } from '../avatar/meshMath';
import { parseDeleteVerts } from './bodyHide';

/** Everything that is loaded once per template (geometry index, UVs, binding, skin weights, texture material). */
export interface TemplateRuntime {
  def: GarmentTemplateDef;
  vertexCount: number;
  index: Uint32Array;
  uv: Float32Array;
  binding: GarmentBinding;
  neutralPositions: Float32Array;
  rigidAttachments: GarmentRigidAttachments | null;
  skinIndices: Uint16Array;
  skinWeights: Float32Array;
  weld: WeldMap;
  deleteVerts: Uint32Array;
  /** Material of the glb (texture, alpha test, sidedness); instances clone it. */
  material: MeshStandardMaterial;
}

/** Skin data of the avatar body the garments are skinned like. */
export interface BodySkinData {
  renderVertexCount: number;
  skinIndex: ArrayLike<number>;
  skinWeight: ArrayLike<number>;
}

async function fetchBuffer(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  return response.arrayBuffer();
}

/** Importer candidates predate the catalogue field: probe their established shoe filename as well. */
export async function loadRigidAttachments(
  def: GarmentTemplateDef,
  baseUrl: string,
): Promise<{ data: unknown } | null> {
  if (
    def.rigidAttachments !== undefined &&
    (typeof def.rigidAttachments !== 'string' || def.rigidAttachments.trim().length === 0)
  )
    throw new Error(`garment ${def.id}: invalid rigid attachment path`);
  if (!def.rigidAttachments && def.category !== 'shoes') return null;
  if (def.category !== 'shoes')
    throw new Error(`garment ${def.id}: rigid attachments require shoes`);
  const path = def.rigidAttachments ?? `${def.id}.rigid.json`;
  const response = await fetch(`${baseUrl}${path}`);
  if (!response.ok) {
    if (!def.rigidAttachments && response.status === 404) return null;
    throw new Error(`Failed to fetch ${baseUrl}${path}: ${response.status}`);
  }
  const data: unknown = await response.json();
  return { data };
}

function firstMesh(root: Object3D): Mesh {
  let found: Mesh | null = null;
  root.traverse((o) => {
    if (!found && o instanceof Mesh) found = o;
  });
  if (!found) throw new Error('garment glb contains no mesh');
  return found;
}

export async function loadTemplateRuntime(
  def: GarmentTemplateDef,
  body: BodySkinData,
  baseUrl: string,
): Promise<TemplateRuntime> {
  validateGarmentLicense(def.license, def.attribution);
  if (!def.binding)
    throw new Error(`garment ${def.id}: rigid (unbound) templates are not supported`);
  const [gltf, bindingBuffer, deleteBuffer, rigidJson] = await Promise.all([
    new GLTFLoader().loadAsync(`${baseUrl}${def.mesh}`),
    fetchBuffer(`${baseUrl}${def.binding}`),
    def.deleteVerts ? fetchBuffer(`${baseUrl}${def.deleteVerts}`) : Promise.resolve(null),
    loadRigidAttachments(def, baseUrl),
  ]);
  const mesh = firstMesh(gltf.scene);
  const position = mesh.geometry.getAttribute('position') as BufferAttribute;
  const uv = mesh.geometry.getAttribute('uv') as BufferAttribute | undefined;
  const vertexCount = position.count;
  const binding = parseGarmentBinding(bindingBuffer, body.renderVertexCount);
  if (binding.count !== vertexCount) {
    throw new Error(
      `garment ${def.id}: binding has ${binding.count} vertices, the mesh ${vertexCount}`,
    );
  }
  const indexAttribute = mesh.geometry.getIndex();
  const index = indexAttribute
    ? Uint32Array.from(indexAttribute.array as ArrayLike<number>)
    : Uint32Array.from({ length: vertexCount }, (_, i) => i);
  const { skinIndices, skinWeights } = garmentSkinWeights(binding, body.skinIndex, body.skinWeight);
  const source = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  const material =
    source instanceof MeshStandardMaterial
      ? source
      : new MeshStandardMaterial({ color: def.baseColor, roughness: 0.85 });
  return {
    def,
    vertexCount,
    index,
    uv: uv ? Float32Array.from(uv.array as ArrayLike<number>) : new Float32Array(vertexCount * 2),
    binding,
    neutralPositions: Float32Array.from(position.array as ArrayLike<number>),
    rigidAttachments:
      rigidJson === null ? null : parseGarmentRigidAttachments(rigidJson.data, vertexCount),
    skinIndices,
    skinWeights,
    weld: buildWeldMap(position.array as ArrayLike<number>),
    deleteVerts: deleteBuffer ? parseDeleteVerts(deleteBuffer) : new Uint32Array(0),
    material,
  };
}
