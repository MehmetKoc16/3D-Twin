import { MeshPhysicalMaterial, ShaderChunk, type MeshPhysicalMaterialParameters } from 'three';

const directDiffuse =
  'reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution ) * ( 1.0 - F );';

/** A diffuse-only wrap lobe: respects light colour/shadowing and leaves the physical specular lobe intact. */
export function skinLightingChunk(chunk: string): string {
  if (!chunk.includes(directDiffuse))
    throw new Error('Skin lighting requires the three.js physical diffuse shader');
  return chunk.replace(
    directDiffuse,
    `${directDiffuse}
    float dtSkinNL = dot( geometryNormal, directLight.direction );
    float dtSkinWrap = saturate( ( dtSkinNL + 0.28 ) / 1.28 );
    float dtSkinTerminator = dtSkinWrap * ( 1.0 - smoothstep( 0.0, 0.45, dtSkinNL ) );
    reflectedLight.directDiffuse += BRDF_Lambert( material.diffuseContribution )
      * directLight.color * vec3( 1.0, 0.32, 0.22 ) * dtSkinTerminator * dtSkinScatter;
  `,
  );
}

/** No extra texture, render pass or tangent attribute. Each material owns its scatter uniform. */
export function createSkinMaterial(
  parameters: MeshPhysicalMaterialParameters = {},
): MeshPhysicalMaterial {
  const material = new MeshPhysicalMaterial({
    roughness: 0.55,
    metalness: 0,
    sheen: 0.12,
    sheenRoughness: 0.65,
    sheenColor: '#ffe1d2',
    specularIntensity: 0.65,
    envMapIntensity: 0.85,
    ...parameters,
  });
  const scatter = { value: 0.12 };
  material.userData.skinScatter = scatter;
  material.onBeforeCompile = (shader) => {
    shader.uniforms.dtSkinScatter = scatter;
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <lights_physical_pars_fragment>',
      `uniform float dtSkinScatter;\n${skinLightingChunk(ShaderChunk.lights_physical_pars_fragment)}`,
    );
  };
  material.customProgramCacheKey = () => 'dt-skin-wrap-v1';
  return material;
}

export function setSkinAppearance(material: MeshPhysicalMaterial, mannequin: boolean): void {
  material.roughness = mannequin ? 0.58 : 0.55;
  material.sheen = mannequin ? 0.06 : 0.12;
  const scatter = material.userData.skinScatter as { value: number } | undefined;
  if (scatter) scatter.value = mannequin ? 0.025 : 0.12;
}
