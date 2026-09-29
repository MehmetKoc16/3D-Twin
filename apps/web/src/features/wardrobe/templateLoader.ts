import { Mesh, MeshStandardMaterial, type BufferAttribute, type Object3D } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { garmentSkinWeights, parseGarmentBinding, type GarmentBinding, type GarmentTemplateDef } from '@dt/avatar-core';
import { buildWeldMap, type WeldMap } from '../avatar/meshMath';
import { parseDeleteVerts } from './bodyHide';

/** Everything that is loaded once per template (geometry index, UVs, binding, skin weights, texture material). */
export interface TemplateRuntime {
  def: GarmentTemplateDef;
  vertexCount: number;
  index: Uint32Array;
  uv: Float32Array;
  binding: GarmentBinding;
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
  if (!def.binding) throw new Error(`garment ${def.id}: rigid (unbound) templates are not supported`);
  const [gltf, bindingBuffer, deleteBuffer] = await Promise.all([
    new GLTFLoader().loadAsync(`${baseUrl}${def.mesh}`),
    fetchBuffer(`${baseUrl}${def.binding}`),
    def.deleteVerts ? fetchBuffer(`${baseUrl}${def.deleteVerts}`) : Promise.resolve(null),
  ]);
  const mesh = firstMesh(gltf.scene);
  const position = mesh.geometry.getAttribute('position') as BufferAttribute;
  const uv = mesh.geometry.getAttribute('uv') as BufferAttribute | undefined;
  const vertexCount = position.count;
  const binding = parseGarmentBinding(bindingBuffer, body.renderVertexCount);
  if (binding.count !== vertexCount) {
    throw new Error(`garment ${def.id}: binding has ${binding.count} vertices, the mesh ${vertexCount}`);
  }
  const indexAttribute = mesh.geometry.getIndex();
  const index = indexAttribute
    ? Uint32Array.from(indexAttribute.array as ArrayLike<number>)
    : Uint32Array.from({ length: vertexCount }, (_, i) => i);
  const { skinIndices, skinWeights } = garmentSkinWeights(binding, body.skinIndex, body.skinWeight);
  const source = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  const material =
    source instanceof MeshStandardMaterial ? source : new MeshStandardMaterial({ color: def.baseColor, roughness: 0.85 });
  return {
    def,
    vertexCount,
    index,
    uv: uv ? Float32Array.from(uv.array as ArrayLike<number>) : new Float32Array(vertexCount * 2),
    binding,
    skinIndices,
    skinWeights,
    weld: buildWeldMap(position.array as ArrayLike<number>),
    deleteVerts: deleteBuffer ? parseDeleteVerts(deleteBuffer) : new Uint32Array(0),
    material,
  };
}
