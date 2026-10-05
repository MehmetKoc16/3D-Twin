import { describe, expect, it } from 'vitest';
import { ShaderChunk, ShaderLib, type WebGLRenderer } from 'three';
import { createSkinMaterial, setSkinAppearance, skinLightingChunk } from './skinMaterial';

describe('physical skin shader compatibility', () => {
  it('extends the installed three.js diffuse shader, retaining physical specular and shadow inputs', () => {
    const chunk = skinLightingChunk(ShaderChunk.lights_physical_pars_fragment);
    expect(chunk).toContain('reflectedLight.directSpecular += irradiance * specularBRDF');
    expect(chunk).toContain('BRDF_Lambert( material.diffuseContribution ) * ( 1.0 - F )');
    expect(chunk).toContain('* directLight.color * vec3( 1.0, 0.32, 0.22 )');
    expect(chunk.match(/float dtSkinTerminator/g)).toHaveLength(1);
    expect(() => skinLightingChunk('incompatible upstream shader')).toThrow(
      'three.js physical diffuse shader',
    );
  });

  it('installs the wrap on the real physical shader and updates mannequin scattering without recompilation', () => {
    const material = createSkinMaterial();
    const shader = { ...ShaderLib.physical, uniforms: { ...ShaderLib.physical.uniforms } };
    material.onBeforeCompile(
      shader as Parameters<typeof material.onBeforeCompile>[0],
      {} as WebGLRenderer,
    );
    expect(shader.fragmentShader).not.toContain('#include <lights_physical_pars_fragment>');
    expect(shader.fragmentShader).toContain('uniform float dtSkinScatter;');
    expect(shader.uniforms.dtSkinScatter?.value).toBe(0.12);
    const version = material.version;
    setSkinAppearance(material, true);
    expect(shader.uniforms.dtSkinScatter?.value).toBe(0.025);
    expect(material.roughness).toBe(0.58);
    setSkinAppearance(material, false);
    expect(shader.uniforms.dtSkinScatter?.value).toBe(0.12);
    expect(material.version).toBe(version);
    material.dispose();
  });

  it('does not share scatter state between the twin and the mannequin', () => {
    const twin = createSkinMaterial();
    const mannequin = createSkinMaterial();
    setSkinAppearance(mannequin, true);
    expect(twin.userData.skinScatter).not.toBe(mannequin.userData.skinScatter);
    expect(twin.userData.skinScatter.value).toBe(0.12);
    expect(twin.customProgramCacheKey()).toBe(mannequin.customProgramCacheKey());
    twin.dispose();
    mannequin.dispose();
  });
});
