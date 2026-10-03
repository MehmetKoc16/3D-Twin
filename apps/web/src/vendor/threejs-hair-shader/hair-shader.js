/* eslint-disable -- vendored third-party file, see README.md (the only local modification is this line) */
// Created by Sander Mørch-Jensen — creategamecharacters.com
// SPDX-License-Identifier: MIT
// Open-source Three.js hair-card materials for WebGLRenderer and WebGPURenderer.
// Use applyHairShader(root, { THREE, renderer, atlas, color }) with your own
// card meshes and linear-data strand atlas. See README.md for setup.
// To see rendered hair, use the web-based character creator at
// https://creategamecharacters.com.

export const DEFAULTS = {
  // Coverage gain. The atlas is authored for alpha-BLEND, so its values are
  // low; the cut threshold is measured against coverage × this, never against
  // the raw texel. Per style in mh_materials.json (`hair_density`).
  density: 2.5,
  // The INNER core's clip, measured against RAW coverage (not the gained
  // coverage the outer pass uses). It is a solid centre for the strand
  // bodies to sit on, never the visible surface — style key
  // `inner_alpha_threshold`. Toward 1 → the core shrinks to nothing.
  innerThreshold: 0.5,
  // The BLEND fringe (third pass, 'both' mode only) — not the outer pass, which is never
  // dimmed. viewer.js's own default for the clone.
  blendOpacity: 0.4,
  // The fringe is matte on purpose: it inherits the base's shiny roughness otherwise and
  // reads as uncontrolled glints. The outer pass owns the sharp specular.
  blendRoughness: 0.6,
  rootDarkening: 0.0,       // root→tip tone (atlas G / legacy R) — MONO-COLOR only
  // TWO ROOT MODES (owner, 2026-08-10).
  //   'mono'  — one hair colour; the root end is merely DARKENED by `rootDarkening`.
  //   'multi' — the root end takes its OWN colour, so the strand runs between two colours.
  //             It does NOT have to be darker: blonde roots under dark tips is a real look,
  //             and the scalar could never express it. `rootDarkening` is bypassed here.
  // `natural`/`colored` were the day-one names and are still accepted, because colour styles
  // were saved under them.
  rootMode: 'mono',
  rootColor: '#1a1008',
  rootStrength: 1.0,        // how far up the strand the root colour reaches
  seedVariation: 0.36,      // per-strand brightness variance (atlas B)
  roughnessFloor: 0.55,
  roughnessSeedAmp: 0.08,
  anisotropy: 0.44,         // GGX highlight along the strand tangent
  anisotropyRotation: -1.09,
  color: '#3a2a1e',
  alphaChannel: 'r',        // compact atlas: R=coverage, G=root→tip, B=seed
};

const SWZ = { r: 'r', g: 'g', b: 'b', a: 'a' };

/** Is this root mode the two-colour one? Accepts the day-one names too. */
export const isMultiRoot = (m) =>
  ['multi', 'multi-color', 'multi-colour', 'colored', 'coloured'].includes(String(m || '').toLowerCase());

/**
 * The colour swatches the product ships. Authored on the real hair in the character editor —
 * these are the settings, not approximations of them — and shipped so every integrator gets
 * the same starting set instead of dialling raw numbers. A project's own saved styles are
 * merged over these by NAME, so a customer can override "Black" without losing the rest.
 *
 * `root_mode: 'multi'` is the two-colour look: `root_color` at the root running into `color`
 * by the tip. The rest are one colour with the root end darkened by `root_darkening`.
 */
export const HAIR_COLOR_STYLES = {
  'Blonde':                     { color: '#927554', root_mode: 'mono',  root_color: '#1a1008', root_strength: 1, root_darkening: 0.26, hair_density: 1.2, hair_roughness_floor: 0.62, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Dark Blonde':                { color: '#523d28', root_mode: 'mono',  root_color: '#1a1008', root_strength: 1, root_darkening: 0,    hair_density: 1.3, hair_roughness_floor: 0.72, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Platina Blonde':             { color: '#d2bca2', root_mode: 'multi', root_color: '#a18d68', root_strength: 1, root_darkening: 0.26, hair_density: 1.3, hair_roughness_floor: 0.62, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Strawberry Blonde':          { color: '#997156', root_mode: 'multi', root_color: '#562915', root_strength: 1, root_darkening: 0,    hair_density: 1.0, hair_roughness_floor: 0.66, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Dirty Blonde':               { color: '#140b06', root_mode: 'multi', root_color: '#6d4f36', root_strength: 1, root_darkening: 0,    hair_density: 1.3, hair_roughness_floor: 0.62, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Brown':                      { color: '#493222', root_mode: 'mono',  root_color: '#1a1008', root_strength: 1, root_darkening: 0,    hair_density: 1.3, hair_roughness_floor: 0.72, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Beaver Brown':               { color: '#372920', root_mode: 'mono',  root_color: '#1a1008', root_strength: 1, root_darkening: 0,    hair_density: 1.3, hair_roughness_floor: 0.78, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Dark Brown':                 { color: '#2e1c10', root_mode: 'mono',  root_color: '#1a1008', root_strength: 1, root_darkening: 0,    hair_density: 1.3, hair_roughness_floor: 0.72, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Deep Brown':                 { color: '#130701', root_mode: 'mono',  root_color: '#1a1008', root_strength: 1, root_darkening: 0.08, hair_density: 1.5, hair_roughness_floor: 0.84, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Black':                      { color: '#050200', root_mode: 'mono',  root_color: '#1a1008', root_strength: 1, root_darkening: 0,    hair_density: 1.2, hair_roughness_floor: 0.62, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Fiery red':                  { color: '#290a05', root_mode: 'multi', root_color: '#9c4021', root_strength: 1, root_darkening: 0,    hair_density: 1.3, hair_roughness_floor: 0.62, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Pastel pink striped blonde': { color: '#8b4b4b', root_mode: 'multi', root_color: '#9f8650', root_strength: 1, root_darkening: 0,    hair_density: 1.3, hair_roughness_floor: 0.62, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Blue Black Striped':         { color: '#000000', root_mode: 'multi', root_color: '#2b6a6e', root_strength: 1, root_darkening: 0,    hair_density: 1.3, hair_roughness_floor: 0.62, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
};

/**
 * FACIAL-HAIR swatches — a SEPARATE set, authored on a beard, not derived from the hair ones.
 *
 * The same shader drives both, but the head-hair numbers read wrong on a chin: beard cards are
 * shorter and denser and lie against skin rather than over a scalp, so a swatch tuned on a head
 * lands somewhere else on a jaw. Every one of these was dialled in on the real beards in the
 * character editor and read back out — they are the settings, not approximations of them.
 *
 * All of them are `root_mode: 'multi'`: a beard reads as two-tone far more than head hair does,
 * because the root is right against lit skin. "Salt and pepper" is the extreme of that — white
 * tips over black roots — and it is the case that will not survive being flattened to mono.
 *
 * COLOUR ONLY, like the hair set. These carried a material half too — an anisotropic highlight
 * with no tangent map behind it, and a blend opacity/roughness dialled against the beard-only
 * inner core — which is the separate beard shader that is gone (owner, 2026-09-19: hair is hair,
 * on the head or on the face). The passes take their material numbers from the style.
 *
 * Merged UNDER a project's own saved swatches by name, same as the hair set.
 */
export const BEARD_COLOR_STYLES = {
  'Blonde':          { color: '#e4b89b', root_mode: 'multi', root_color: '#956d50', root_strength: 1, root_darkening: 0, hair_density: 1.2, hair_roughness_floor: 0.72, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'DirtyBlonde':     { color: '#dfaa86', root_mode: 'multi', root_color: '#59351c', root_strength: 1, root_darkening: 0, hair_density: 1.2, hair_roughness_floor: 0.8, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Light brown':     { color: '#ba805a', root_mode: 'multi', root_color: '#401b0d', root_strength: 1, root_darkening: 0, hair_density: 1.2, hair_roughness_floor: 0.8, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Redbrown':        { color: '#7a3a18', root_mode: 'multi', root_color: '#3e1c0f', root_strength: 1, root_darkening: 0, hair_density: 1.2, hair_roughness_floor: 0.8, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Orange':          { color: '#dd7936', root_mode: 'multi', root_color: '#522614', root_strength: 1, root_darkening: 0, hair_density: 1.2, hair_roughness_floor: 0.8, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Grey':            { color: '#828282', root_mode: 'multi', root_color: '#555453', root_strength: 1, root_darkening: 0, hair_density: 1.2, hair_roughness_floor: 0.72, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Salt and pepper': { color: '#d4d4d4', root_mode: 'multi', root_color: '#000000', root_strength: 1, root_darkening: 0, hair_density: 1.2, hair_roughness_floor: 0.72, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
  'Black':           { color: '#2e2e2e', root_mode: 'multi', root_color: '#000000', root_strength: 1, root_darkening: 0, hair_density: 1.2, hair_roughness_floor: 0.78, seed_variation: 0.36, hair_roughness_seed_amp: 0.08 },
};

/** One beard swatch as `createHairMaterials` opts. Unknown names return null, never a guess. */
export function beardColorStyle(name, extra = {}) {
  const s = { ...BEARD_COLOR_STYLES, ...(extra || {}) }[name];
  if (!s) return null;
  return {
    color: s.color,
    rootMode: s.root_mode,
    rootColor: s.root_color,
    rootStrength: typeof s.root_strength === 'number' ? s.root_strength : 1,
    rootDarkening: s.root_darkening,
    density: s.hair_density,
    roughnessFloor: s.hair_roughness_floor,
    seedVariation: s.seed_variation,
    roughnessSeedAmp: s.hair_roughness_seed_amp,
    // The MATERIAL half of a swatch — the beard set carries an anisotropy, a rotation and the
    // inner core's opacity and roughness, which the editor applies and a resolver that dropped
    // them left the runtime with matte beards (owner, 2026-09-16). Absent fields stay absent.
    ...(typeof s.anisotropy === 'number' ? { anisotropy: s.anisotropy } : {}),
    ...(typeof s.anisotropy_rotation === 'number' ? { anisotropyRotation: s.anisotropy_rotation } : {}),
    ...(typeof s.blend_opacity === 'number' ? { blendOpacity: s.blend_opacity } : {}),
    ...(typeof s.blend_roughness === 'number' ? { blendRoughness: s.blend_roughness } : {}),
    ...(s.mode ? { mode: s.mode } : {}),
  };
}

/** One swatch as `createHairMaterials` opts. Unknown names return null rather than a guess. */
export function hairColorStyle(name, extra = {}) {
  const s = { ...HAIR_COLOR_STYLES, ...(extra || {}) }[name];
  if (!s) return null;
  return {
    color: s.color,
    rootMode: s.root_mode,
    rootColor: s.root_color,
    rootStrength: typeof s.root_strength === 'number' ? s.root_strength : 1,
    rootDarkening: s.root_darkening,
    density: s.hair_density,
    roughnessFloor: s.hair_roughness_floor,
    seedVariation: s.seed_variation,
    roughnessSeedAmp: s.hair_roughness_seed_amp,
    // The MATERIAL half of a swatch — the beard set carries an anisotropy, a rotation and the
    // inner core's opacity and roughness, which the editor applies and a resolver that dropped
    // them left the runtime with matte beards (owner, 2026-09-16). Absent fields stay absent.
    ...(typeof s.anisotropy === 'number' ? { anisotropy: s.anisotropy } : {}),
    ...(typeof s.anisotropy_rotation === 'number' ? { anisotropyRotation: s.anisotropy_rotation } : {}),
    ...(typeof s.blend_opacity === 'number' ? { blendOpacity: s.blend_opacity } : {}),
    ...(typeof s.blend_roughness === 'number' ? { blendRoughness: s.blend_roughness } : {}),
    ...(s.mode ? { mode: s.mode } : {}),
  };
}
let _cacheKey = 0;

// Unique program cache key: these materials carry hand-written shader chunks,
// so two of them must never share a compiled program.
function uniqueCacheKey(mat) {
  const k = 'gcc-hair-' + (++_cacheKey);
  mat.customProgramCacheKey = () => k;
}

// ── WHICH three IS DRAWING — detected, never chosen by the caller ────────────
//
// A classic WebGLRenderer compiles GLSL, and every patch in this file rides `onBeforeCompile`.
// The WebGPURenderer (and its WebGL fallback backend) builds NODE materials from TSL and never
// calls onBeforeCompile: on it a patched MeshPhysicalMaterial silently draws as a stock one,
// whose stock alphaMap reads the GREEN channel while this atlas keeps coverage in RED — the
// hair simply vanished on the first WebGPU game (riften i tid, 2026-09-09). So there are two
// twins of each pass below, GLSL and TSL, and the RENDERER says which one is built:
// `renderer.isWebGPURenderer` is the whole test (true for both of that renderer's backends —
// the node path is per renderer, not per GPU API). Without a renderer, a `three/webgpu`
// namespace gives it away: it has TSL and no WebGLRenderer export at all.
//
// Select the renderer's material system.
export function usesNodeMaterials(THREE, renderer) {
  if (renderer) return renderer.isWebGPURenderer === true;
  return !!(THREE && THREE.TSL && THREE.MeshPhysicalNodeMaterial && !THREE.WebGLRenderer);
}

// The node path needs the node material classes and TSL, and `three/webgpu` is the namespace
// that carries them (`import * as THREE from 'three/webgpu'`); `three` proper does not.
export function requireTSL(THREE, who) {
  if (!THREE || !THREE.TSL || !THREE.MeshPhysicalNodeMaterial || !THREE.MeshStandardNodeMaterial) {
    throw new Error(who + ': a WebGPURenderer needs the node material classes — pass THREE from '
      + '"three/webgpu" (import * as THREE from "three/webgpu"), not from "three"');
  }
  return THREE.TSL;
}

// A node material carrying a classic material's values — what the WebGPURenderer does itself
// for an untouched material (NodeLibrary.fromMaterial copies every enumerable key), done here so
// custom nodes can be attached before the first draw. TEXTURES stay shared by reference (one
// upload for the whole crowd); VALUE objects — Color, Vector2, … — are cloned, because three's
// key copy hands the SAME Color to every material made from one source, and then one hair's
// `setColor` recolours every character wearing that style. `userData` is copied, not shared:
// the passes write their uniforms into it. Nodes pass by reference, as NodeMaterial.copy does.
export function toNodeMaterial(THREE, source, Cls) {
  const m = new Cls();
  if (source) {
    for (const k in source) {
      if (k === 'uuid' || k === 'type' || k === 'userData' || k === 'onBeforeCompile' || k === 'customProgramCacheKey') continue;
      const v = source[k];
      m[k] = (v && typeof v === 'object' && typeof v.clone === 'function' && !v.isTexture && !v.isNode) ? v.clone() : v;
    }
    m.userData = { ...(source.userData || {}) };
    m.name = source.name;
  }
  return m;
}

// `clone()` of a NODE material keeps its nodes and its base Material fields and DROPS the
// rest — colour, roughness, alphaMap, anisotropy all come back at defaults (three r183,
// NodeMaterial.copy → Material.copy). So a pass cloned from another pass is copied key by
// key on the node path, and cloned normally on the classic one.
export function cloneMaterial(THREE, m) {
  return m.isNodeMaterial ? toNodeMaterial(THREE, m, m.constructor) : m.clone();
}

// Read a style's authored params: GET <styleUrl>/mh_materials.json.
export async function loadHairParams(styleUrl) {
  const res = await fetch(styleUrl.replace(/\/$/, '') + '/mh_materials.json');
  if (!res.ok) throw new Error('hair params ' + res.status + ' for ' + styleUrl);
  const json = await res.json();
  const entry = (json.materials || []).find((m) => m.kind === 'hair') || (json.materials || [])[0];
  return { params: (entry && entry.params) || {}, textures: (entry && entry.textures) || {} };
}

// Where a style keeps its strand coverage atlas, relative to the style folder (null = none).
export function hairAtlasPath(params = {}, textures = {}) {
  return params.alpha_stem ? 'textures/' + params.alpha_stem + '.png'
       : (textures.alpha_r || textures.alpha || null);
}

// Load the strand coverage atlas. It is DATA, not colour: no sRGB decode.
export function loadHairAtlas(THREE, styleUrl, { params = {}, textures = {} } = {}) {
  const rel = hairAtlasPath(params, textures);
  if (!rel) return null;
  const tex = new THREE.TextureLoader().load(styleUrl.replace(/\/$/, '') + '/' + rel);
  tex.flipY = false;                       // glTF UV convention
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  if ('colorSpace' in tex) tex.colorSpace = THREE.NoColorSpace;
  else tex.encoding = 3000;                // LinearEncoding, three < r152
  return tex;
}

// The TSL twin of `hairFragment` below — the same tone, roughness and coverage → alpha route
// as NODES on a MeshPhysicalNodeMaterial, for the WebGPURenderer. Line for line the same math;
// where the GLSL patches a chunk, this replaces the material's node for that slot:
//   map_fragment          → colorNode      (materialColor × tone × strand × self-AO, root colour)
//   roughnessmap_fragment → roughnessNode  (materialRoughness + seed + tip, floored)
//   alphamap_fragment     → opacityNode    (opacity × coverage — gained for the outer pass,
//                                           RAW for the core, exactly as the GLSL)
// Returns the uniform NODES under the same names the GLSL uniforms have, so every writer
// (`setUniform`, the SDK's hair colour styles) reaches them through `.value` unchanged.
// With no atlas nothing is patched — the GLSL sits under `#ifdef USE_ALPHAMAP` for the same reason.
function hairNodes(THREE, mat, { atlas, alphaSwz, rootChan, seedChan, core, rootMode, rootColor, o, density, inst = null }) {
  const T = THREE.TSL;
  // WHERE THE SWATCH COMES FROM. A spawned strand: uniform nodes. A crowd pass (`inst`) is one
  // material for every wearer, so the root swatch reads two vec4 instance attributes and the tip
  // colour a vec3 one — the same six numbers the GLSL crowd routes through #defines. Same graph.
  const u = inst ? {
    uHairRootDark: inst.rootB.y, uHairRootMode: inst.rootA.w, uHairRootColor: inst.rootA.rgb, uHairRootStr: inst.rootB.x,
    uHairSeedAmp: T.uniform(o.seedVariation), uHairRoughFlr: inst.rootB.w, uHairRoughSeed: T.uniform(o.roughnessSeedAmp),
  } : {
    uHairRootDark: T.uniform(o.rootDarkening),
    uHairRootMode: T.uniform(rootMode),
    uHairRootColor: T.uniform(rootColor),
    uHairRootStr: T.uniform(o.rootStrength),
    uHairSeedAmp: T.uniform(o.seedVariation),
    uHairRoughFlr: T.uniform(o.roughnessFloor),
    uHairRoughSeed: T.uniform(o.roughnessSeedAmp),
  };
  if (!core) u.uHairDensity = inst ? inst.rootB.z : T.uniform(density);
  if (!atlas) return u;
  const tex = T.texture(atlas);
  const rootT = tex[rootChan], seed = tex[seedChan], cov = tex[alphaSwz];
  const multi = u.uHairRootMode.greaterThan(0.5);
  // MONO-COLOR keeps the scalar darkening; MULTI-COLOR bypasses it (same two lines as the site).
  const toneMul = T.select(multi, T.float(1), T.mix(u.uHairRootDark, T.float(1), rootT));
  const strandMul = seed.sub(0.5).mul(2).mul(u.uHairSeedAmp).add(1);
  const ndv = T.abs(T.dot(T.normalize(T.normalView), T.positionViewDirection));
  const hairAO = T.mix(T.float(0.3), T.float(1), T.smoothstep(T.float(0.05), T.float(0.55), ndv));
  // THE TIP COLOUR GOES IN BEFORE THE ROOT MIX: material.color for a spawned strand, the
  // instance's own for a crowd strand (three's post-multiply of instanceColor would land AFTER
  // the mix — the orange beard — so the crowd never sets instanceColor on this path).
  const tip = inst ? T.materialColor.rgb.mul(inst.tip) : T.materialColor.rgb;
  const base = tip.mul(toneMul.mul(strandMul).mul(hairAO));
  // The root colour is a GRADIENT along the strand: full at the root, gone by the tip.
  const rooted = T.mix(u.uHairRootColor.mul(strandMul).mul(hairAO), base, rootT);
  mat.colorNode = T.vec4(T.select(multi, T.mix(base, rooted, u.uHairRootStr), base), 1);
  const rough = T.materialRoughness
    .add(seed.sub(0.5).mul(2).mul(u.uHairRoughSeed))
    .add(rootT.oneMinus().mul(0.06));
  mat.roughnessNode = T.clamp(T.max(rough, u.uHairRoughFlr), 0, 1);
  const opacity = T.materialReference('opacity', 'float');
  mat.opacityNode = core ? opacity.mul(cov)
                         : opacity.mul(T.clamp(cov.mul(u.uHairDensity), 0, 1));
  mat.needsUpdate = true;
  return u;
}

// The fragment patch both passes share: strand tone, roughness, and the
// coverage → alpha route (with the gain applied BEFORE any cut).
function hairFragment(shader, { alphaSwz, rootChan, seedChan, core }) {
  shader.fragmentShader = `
    // Created by Sander Mørch-Jensen — creategamecharacters.com
    // SPDX-License-Identifier: MIT
    
    uniform float uHairRootDark;
    uniform float uHairRootMode;
    uniform vec3  uHairRootColor;
    uniform float uHairRootStr;
    uniform float uHairSeedAmp;
    uniform float uHairRoughFlr;
    uniform float uHairRoughSeed;
    uniform float uHairDensity;
  ` + shader.fragmentShader;

  // Root→tip darkening + per-strand tint variance + normal-based self-AO:
  // cards edge-on or facing away sit deeper in the hair volume, so they darken.
  shader.fragmentShader = shader.fragmentShader.replace(
    '#include <map_fragment>',
    `
    #include <map_fragment>
    #ifdef USE_ALPHAMAP
      vec4 hairAtlas = texture2D( alphaMap, vAlphaMapUv );
      float rootT = hairAtlas.${rootChan};
      float seed  = hairAtlas.${seedChan};
      // MONO-COLOR keeps the scalar darkening; MULTI-COLOR bypasses it so the root colour
      // below is not fighting a darkener it did not ask for. Same two lines as the site.
      float toneMul   = uHairRootMode > 0.5 ? 1.0 : mix( uHairRootDark, 1.0, rootT );
      float strandMul = 1.0 + (seed - 0.5) * 2.0 * uHairSeedAmp;
      float ndv = abs( dot( normalize(vNormal), normalize(vViewPosition) ) );
      float hairAO = mix( 0.3, 1.0, smoothstep( 0.05, 0.55, ndv ) );
      diffuseColor.rgb *= toneMul * strandMul * hairAO;
      if ( uHairRootMode > 0.5 ) {
        // The root colour is a GRADIENT along the strand: full at the root, gone by the tip.
        vec3 rooted = mix( uHairRootColor * strandMul * hairAO, diffuseColor.rgb, rootT );
        diffuseColor.rgb = mix( diffuseColor.rgb, rooted, uHairRootStr );
      }
    #endif
    `
  );

  // Roughness floor + per-strand seed: tips slightly rougher than roots, and
  // every card its own offset so the head isn't one uniform mirror.
  shader.fragmentShader = shader.fragmentShader.replace(
    '#include <roughnessmap_fragment>',
    `
    float roughnessFactor = roughness;
    #ifdef USE_ROUGHNESSMAP
      vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
      roughnessFactor *= texelRoughness.g;
    #endif
    #ifdef USE_ALPHAMAP
      float hairRootT = texture2D( alphaMap, vAlphaMapUv ).${rootChan};
      float hairSeed  = texture2D( alphaMap, vAlphaMapUv ).${seedChan};
      roughnessFactor = max(
        roughnessFactor + (hairSeed - 0.5) * 2.0 * uHairRoughSeed + (1.0 - hairRootT) * 0.06,
        uHairRoughFlr );
    #endif
    roughnessFactor = clamp( roughnessFactor, 0.0, 1.0 );
    `
  );

  // Coverage → alpha. three's stock chunk samples .g; the compact atlas keeps
  // coverage in the declared channel.
  //
  // THE GAIN IS THE OUTER PASS'S ALONE. The atlas is authored for alpha-BLEND, so its
  // values are low and the outer pass multiplies them up by `hair_density` to get the
  // coverage the strands should actually have. The INNER core deliberately clips the RAW
  // texel instead: gaining it first — which this module did until 2026-08-09 — turns a
  // small solid centre into most of the card, and since the root end of the atlas is the
  // dark end, that reads as heavy black roots. It also leaves no feathered edge for the
  // outer pass to own, which is the hard hairline.
  shader.fragmentShader = shader.fragmentShader.replace(
    '#include <alphamap_fragment>',
    core
      ? `
    #ifdef USE_ALPHAMAP
      diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).${alphaSwz};
    #endif
    `
      : `
    #ifdef USE_ALPHAMAP
      float hairCoverage = texture2D( alphaMap, vAlphaMapUv ).${alphaSwz};
      diffuseColor.a *= clamp( hairCoverage * uHairDensity, 0.0, 1.0 );
    #endif
    `
  );
}

// ── the two passes (ported from shared/viewer.js, 2026-08-09) ───────────────
//
// THIS IS A PORT OF THE SITE'S OWN RENDERER, NOT A RE-EXPRESSION OF IT. Every flag,
// threshold and default below is taken from `shared/viewer.js` (`applyHair`,
// `addHairInnerPass`, and the `__blend` clone built beside them), because the character
// editor is the thing an integrator's hair has to match and matching it by copying VALUES
// across two implementations demonstrably does not converge — it was tried, twice, and the
// owner rejected both results. The editor is deliberately NOT refactored to import this:
// it works, and moving the working side to fix the broken one is the wrong risk. That
// leaves two copies, so `dev/sdk-check.mjs` compares them and fails on drift.
//
// There are TWO passes on every hair mesh, and MSAA decides which two. Getting the passes
// wrong is the whole history of this file:
//
//   WITH MSAA (mode 'both', the normal case)
//   A2C           the hair. `alphaToCoverage`, so strand edges resolve through multisampling
//                 instead of a hard cut; OPAQUE bucket, writes depth — this is the pass that
//                 makes the hair solid. Carries the tone shader (root→tip darkening,
//                 per-strand seed, self-AO) and the roughness floor.
//   BLEND         a soft fringe on top. Transparent, no depth write, low opacity. Its shader
//                 is deliberately NOT the tone shader: coverage with its own fixed 2.5 gain
//                 and a `< 0.06` discard so nothing renders between strands, plus the
//                 strand's root→tip colour. Root darkening, the seed and the self-AO stay
//                 out — they would double-darken the fringe.
//
//   WITHOUT MSAA (mode 'blend')
//   OUTER         the hair as a transparent pass that still writes depth, texels under 0.06
//                 discarded. Same tone shader.
//   INNER         a small opaque core, alphaTest against RAW coverage — never the gained
//                 coverage the outer pass uses — at the outer pass's own depth (the same
//                 small polygon offset), so the outer pass drawn after it lands on top of
//                 it. Same tone shader. It exists only here: with MSAA the A2C pass is solid.
//
// Mode 'a2c' builds the MSAA pair with the fringe hidden. A saved per-material mode wins over
// the MSAA check. Every hair mesh — head, beard, brow — follows the same rule: hair is hair
// wherever it grows.
// A TRANSPARENT HAIR PASS MUST NOT PUNCH A HOLE IN THE CANVAS.
//
// Default blending writes `src.a` into the framebuffer's ALPHA as well as blending colour,
// so a fringe at alpha 0.4 leaves the destination alpha at 0.4. three.js creates EVERY canvas
// context with `alpha: true` (its `alpha` option only sets the clear alpha), so the compositor
// then shows the PAGE through the hair, and the hair reads as see-through in a way it does in no offline
// render. A game hits this as "the backdrop comes through the hair", and the only fix from
// outside the SDK is to give up the transparent canvas entirely: one integration moved its
// whole photographic backdrop into the GL scene and forced an opaque canvas for exactly this
// (`stage.js`, 2026-09-11 — "with `alpha: true` the hair's transparent pass blends toward an
// alpha of 0, so the page behind showed straight through the hair"). That is the SDK making
// the integrator solve the SDK's problem.
//
// COLOUR IS UNCHANGED — src·a + dst·(1−a), exactly as before, so nothing moves on an opaque
// canvas. Only the ALPHA channel is separated: it ACCUMULATES (src + dst, saturating at 1),
// so wherever the hair draws, the canvas ends up opaque there and the page stays behind it.
function keepCanvasOpaque(THREE, mat) {
  mat.blending = THREE.CustomBlending;
  mat.blendEquation = THREE.AddEquation;
  mat.blendSrc = THREE.SrcAlphaFactor;
  mat.blendDst = THREE.OneMinusSrcAlphaFactor;
  mat.blendEquationAlpha = THREE.AddEquation;
  mat.blendSrcAlpha = THREE.OneFactor;
  mat.blendDstAlpha = THREE.OneFactor;
}

// THE OPAQUE HAIR PASSES MUST NOT PUNCH A HOLE IN THE CANVAS EITHER — and they did, on EVERY
// WebGL page, because three.js creates its canvas context with `alpha: true` whatever the
// renderer was given (`alpha: false` only changes the clear alpha). An opaque pass gets
// NoBlending from three (NormalBlending + !transparent), which writes the fragment's alpha
// straight into the canvas: the A2C pass's coverage alpha on every partly covered strand
// texel, the no-MSAA core's alpha above its clip. The browser then composites the PAGE behind
// the canvas through every strand edge. On a dark page (the editor, the dev scenes) that
// only reads as denser hair, so nothing showed; on a light page it paints every strand white
// — Prøvetakingsmesteren's grey-white hair (2026-09-24): identical materials, shader, atlas
// and geometry to a page that rendered brown, and the hair turned brown the moment the page
// behind its canvas was made black.
//
// COLOUR IS WRITTEN STRAIGHT (src·1 + dst·0 — exactly what NoBlending wrote, so the picture on
// the canvas does not change). Only ALPHA is separated: it ACCUMULATES like the transparent
// passes' (src + dst, saturating at 1), so where the hair covers the head or the scene
// background the canvas stays opaque. Alpha-to-coverage is untouched — the coverage mask is
// taken from the fragment's alpha before blending. WebGL only: the WebGPU path keeps its own
// blending (see the core pass below).
function keepCanvasOpaqueStraight(THREE, mat) {
  mat.blending = THREE.CustomBlending;
  mat.blendEquation = THREE.AddEquation;
  mat.blendSrc = THREE.OneFactor;
  mat.blendDst = THREE.ZeroFactor;
  mat.blendEquationAlpha = THREE.AddEquation;
  mat.blendSrcAlpha = THREE.OneFactor;
  mat.blendDstAlpha = THREE.OneFactor;
}

export function createHairMaterials(THREE, source, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  // `cutThreshold` was this number's name while the passes were the wrong way round. It
  // means the inner core's clip either way, so an older caller keeps working.
  if (typeof opts.innerThreshold !== 'number' && typeof opts.cutThreshold === 'number') {
    o.innerThreshold = opts.cutThreshold;
  }
  const alphaSwz = SWZ[String(o.alphaChannel).toLowerCase()] || 'r';
  // Compact atlas (alpha=R) → root=G, seed=B. Legacy atlas (alpha=A) → root=R.
  const rootChan = alphaSwz === 'a' ? 'r' : 'g';
  const seedChan = 'b';
  const rootMode = isMultiRoot(o.rootMode) ? 1 : 0;
  const rootColor = new THREE.Color(o.rootColor || DEFAULTS.rootColor);

  // ALPHA-TO-COVERAGE NEEDS MSAA, and viewer.js decides it exactly this way:
  // `gl.getParameter(gl.SAMPLES) > 0`. Pass `renderer` (or `alphaToCoverage: true|false`).
  // With neither, assume none — correct, but softer-edged than the site, so pass it.
  let msaa = o.alphaToCoverage;
  if (typeof msaa !== 'boolean') {
    msaa = false;
    try {
      const r = o.renderer;
      if (r && r.isWebGPURenderer) {
        // No GL context to ask: `WebGPURenderer({ antialias })` sets its sample count (4).
        msaa = ((r.samples | 0) > 0) || ((r.currentSamples | 0) > 0);
      } else {
        const gl = r && r.getContext && r.getContext();
        if (gl) msaa = gl.getParameter(gl.SAMPLES) > 0;
      }
    } catch (_) { /* no context to ask — keep the fallback */ }
  }
  // GLSL patches or TSL nodes — the renderer decides (see usesNodeMaterials).
  const nodes = usesNodeMaterials(THREE, o.renderer);
  if (nodes) requireTSL(THREE, 'createHairMaterials');

  // HAIR IS HAIR, on the head or on the face (owner, 2026-09-19). One rule for every hair mesh:
  // a beard, a brow or a mustache gets exactly the passes, mode and density head hair gets. The
  // name-keyed "facial hair" branch this had (never alpha-to-coverage, density 1.0) gave facial
  // hair a different shader than head hair, which was never the design. A saved per-material
  // mode still wins over the auto-detect.
  const name = (source && source.name) || o.materialName || 'hair';
  const savedMode = o.mode;                       // 'a2c' | 'blend' | 'both' | undefined
  const useA2C = savedMode ? (savedMode === 'a2c' || savedMode === 'both') : msaa;
  const mode = savedMode || (useA2C ? 'both' : 'blend');
  // viewer.js: `p.hair_density ?? (A2C ? 2.5 : 1.0)`. Read the CALLER's value, not the merged
  // one: DEFAULTS carries 2.5 as documentation of the A2C figure, and merging it first would
  // make the no-MSAA branch unreachable.
  const density = typeof opts.density === 'number' ? opts.density : (msaa ? 2.5 : 1.0);

  // ── OUTER: the hair ──
  // Physical, not Standard: the strand highlight is anisotropic.
  const outer = nodes ? toNodeMaterial(THREE, source, THREE.MeshPhysicalNodeMaterial)
                      : new THREE.MeshPhysicalMaterial();
  if (source && !nodes) THREE.MeshStandardMaterial.prototype.copy.call(outer, source);
  outer.name = name + (useA2C ? '__a2c' : '__outer');
  outer.map = null;                     // the bound texture is DATA, not colour
  outer.alphaMap = o.atlas || null;
  outer.alphaHash = false;
  outer.color.set(o.color);
  outer.roughness = typeof o.roughness === 'number' ? o.roughness : 0.5;
  outer.side = THREE.DoubleSide;        // cards are single-sided quads
  outer.alphaTest = 0.0;                // coverage decides, never a hard cut
  if (useA2C) {
    outer.alphaToCoverage = true;
    outer.transparent = false;          // MSAA resolves the edge; opaque bucket
    outer.depthWrite = true;
    if (!nodes) keepCanvasOpaqueStraight(THREE, outer);   // its coverage alpha never reaches the page
  } else {
    // WITHOUT MSAA THE BLENDED OUTER PASS STILL WRITES DEPTH, as the A2C pass does. Without it
    // three draws a mesh's cards in index order, not by distance, so a card at the back drawn
    // after one in front landed ON TOP of it: the shaded back of a beard or a head of hair bled
    // through its front (owner, 2026-09-19: "backside is bleeding through, this is all a depth
    // issue"). Worst at small texture sizes, where the blurred coverage leaves the opaque core
    // almost nothing to occlude with. Near-empty texels are discarded — the fringe pass's own
    // 0.06 rule — so an empty card corner never writes depth over what is behind it.
    outer.alphaToCoverage = false;
    outer.transparent = true;
    outer.depthWrite = true;
    outer.alphaTest = 0.06;
    keepCanvasOpaque(THREE, outer);
  }
  // Hair sits flush against the face; a SMALL bias wins the depth test there without
  // pulling the far side of voluminous hair in front of the head (viewer.js learned this
  // the hard way — the old -4/-16 bled back-facing specular through the face).
  // A CONSTANT bias only — no slope factor (owner, 2026-09-19). The factor scales the pull
  // with the card's depth slope, so a card seen side-on was dragged toward the camera by up
  // to a pixel's worth of its own depth: those cards did not respect depth, came through the
  // other hair and the skin as dark polygons, and it grew with distance. The two units still
  // let a flush card win against the skin under it.
  outer.polygonOffset = true;
  outer.polygonOffsetFactor = 0;
  outer.polygonOffsetUnits = -2;
  if ('anisotropy' in outer && o.atlas && o.tangent) {
    outer.anisotropyMap = o.tangent;
    outer.anisotropy = o.anisotropy;
    outer.anisotropyRotation = o.anisotropyRotation;
  }
  outer.userData.hairPass = useA2C ? 'a2c' : 'outer';
  outer.userData.hairMode = mode;
  uniqueCacheKey(outer);
  // The shaders attach at the END, once the inner/blend clones exist: a clone of a NODE
  // material copies its nodes by reference, and two passes sharing one set of uniform nodes
  // would restyle each other. (The GLSL twin never had this problem — clone() drops
  // onBeforeCompile — and attaching it late changes nothing for it.)
  // PER INSTANCE (`instanced`, a crowd pass): the swatch and the tip ride instance attributes —
  // one set of varying nodes per pass, never shared between passes (nodes copy by reference).
  const instNodes = () => {
    const T = THREE.TSL;
    return { rootA: T.varying(T.attribute('gccRootA', 'vec4')), rootB: T.varying(T.attribute('gccRootB', 'vec4')), tip: T.varying(T.attribute('gccTip', 'vec3')) };
  };
  const attachTone = (mat, core) => {
    if (nodes) {
      mat.userData.hairUniforms = hairNodes(THREE, mat, {
        atlas: o.atlas, alphaSwz, rootChan, seedChan, core, rootMode, rootColor, o, density,
        inst: o.instanced ? instNodes() : null,
      });
      return;
    }
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uHairRootDark = { value: o.rootDarkening };
      shader.uniforms.uHairRootMode = { value: rootMode };
      shader.uniforms.uHairRootColor = { value: rootColor };
      shader.uniforms.uHairRootStr = { value: o.rootStrength };
      shader.uniforms.uHairSeedAmp = { value: o.seedVariation };
      shader.uniforms.uHairRoughFlr = { value: o.roughnessFloor };
      shader.uniforms.uHairRoughSeed = { value: o.roughnessSeedAmp };
      if (!core) shader.uniforms.uHairDensity = { value: density };
      mat.userData.hairUniforms = shader.uniforms;
      hairFragment(shader, { alphaSwz, rootChan, seedChan, core });
    };
  };

  // ── INNER: the opaque core — ONLY WITHOUT MSAA ──
  //
  // This is a FALLBACK, not a third layer. viewer.js builds it under `if (!_useAlphaToCoverage)`
  // and nowhere else: with A2C the outer pass is already opaque and depth-writing, so the hair
  // is solid without it. Adding it anyway — which this module did — runs the tone shader
  // (root->tip darkening) on TWO overlapping opaque layers instead of one, and the roots come
  // out visibly darker than the editor's. It shows first on blonde hair, where the darkening
  // has the most contrast to add.
  //
  // The "hair is never blend-only" law still holds either way: with A2C the outer pass IS the
  // opaque layer; without it, the outer goes transparent and this core is what keeps the hair
  // from reading as glass.
  const inner = useA2C ? null : cloneMaterial(THREE, outer);
  if (inner) {
  inner.name = name + '__inner';
  inner.alphaToCoverage = false;        // never together with alphaTest
  inner.transparent = false;
  // THE CORE WRITES STRAIGHT, NEVER BLENDS its colour. The clone carried the outer pass's
  // colour-blending CustomBlending (keepCanvasOpaque). The WebGPURenderer honours it on an opaque
  // material, so each core strand blended with whatever was drawn before it — the teeth — while
  // writing its pulled-forward depth, and the lips drawn after lost the depth test: teeth showing
  // through the mustache (owner, 2026-09-17, the first WebGPU test project). WebGPU: NoBlending.
  // WebGL: keepCanvasOpaqueStraight — colour exactly as NoBlending, alpha accumulated, so its
  // clipped strands do not show the page behind the canvas (see keepCanvasOpaqueStraight).
  if (nodes) inner.blending = THREE.NoBlending;
  else keepCanvasOpaqueStraight(THREE, inner);
  inner.depthWrite = true;
  inner.opacity = 1.0;
  inner.alphaTest = o.innerThreshold;
  // THE CORE SITS AT THE OUTER PASS'S DEPTH — the same small offset, never a deeper one. The outer
  // pass is drawn after it, so at the same depth it lands ON TOP of the core and hides its hard cut.
  // The deeper -4/-16 this had pulled the core TOWARD the camera on a standard depth buffer (the
  // WebGLRenderer, and a WebGPURenderer without reversedDepthBuffer): the outer pass then failed the
  // depth test wherever the core drew, and the core's hard alpha-cut edges and edge-on slivers were
  // the hair you saw — grain that reads as a texture far sharper than the one loaded (owner, riften
  // i tid on Low/Potat, 2026-09-19). Only a reversed depth buffer turned that offset into "behind",
  // which is why the same hair looked right on the WebGPU levels.
  inner.polygonOffsetFactor = outer.polygonOffsetFactor;
  inner.polygonOffsetUnits = outer.polygonOffsetUnits;
  inner.userData = { ...inner.userData, hairPass: 'inner' };
  uniqueCacheKey(inner);
  }

  // ── BLEND: the fringe, A2C only, visible only in 'both' ──
  let blend = null;
  if (useA2C) {
    blend = cloneMaterial(THREE, outer);
    blend.name = name + '__blend';
    blend.alphaToCoverage = false;
    blend.transparent = true;
    blend.depthWrite = false;
    blend.alphaTest = 0.0;
    keepCanvasOpaque(THREE, blend);
    blend.opacity = typeof o.blendOpacity === 'number' ? o.blendOpacity : 0.4;
    // The clone inherits the base's shiny roughness, which reads as uncontrolled glints on
    // a translucent fringe. Matte by default; the outer pass owns the sharp specular.
    blend.roughness = typeof o.blendRoughness === 'number' ? o.blendRoughness : 0.6;
    blend.userData = { ...blend.userData, hairPass: 'blend' };
    uniqueCacheKey(blend);
    // ALPHA, AND THE STRAND'S OWN COLOUR — no tone treatment. The root DARKENING, the per-strand
    // seed and the self-AO stay off this pass: applying them darkens an already-darkened base
    // twice. The two-colour root is not a darkener but the colour the strand IS, and leaving it
    // out painted the flat TIP colour over everything: a white-tipped beard came out white on
    // every MSAA renderer, and any two-tone hair lost its roots under the fringe (owner,
    // 2026-09-20). Mono strands are one colour, so nothing moves there.
    if (nodes) {
      const T = THREE.TSL;
      const instB = o.instanced ? instNodes() : null;
      const u = { uBlendDensity: T.uniform(2.5) };
      // A crowd's swatch rides its instance attributes; a spawned strand's rides uniform nodes,
      // under the names every swatch writer already knows.
      const bMode = instB ? instB.rootA.w : (u.uHairRootMode = T.uniform(rootMode));
      const bColor = instB ? instB.rootA.rgb : (u.uHairRootColor = T.uniform(rootColor));
      const bStr = instB ? instB.rootB.x : (u.uHairRootStr = T.uniform(o.rootStrength));
      const tipB = instB ? T.materialColor.rgb.mul(instB.tip) : T.materialColor.rgb;
      if (o.atlas) {
        const covB = T.texture(o.atlas)[alphaSwz];
        const opacity = T.materialReference('opacity', 'float');
        blend.opacityNode = T.Fn(() => {
          covB.lessThan(0.06).discard();
          return opacity.mul(T.clamp(covB.mul(u.uBlendDensity), 0, 1));
        })();
        const rootTB = T.texture(o.atlas)[rootChan];
        const rootedB = T.mix(bColor, tipB, rootTB);
        blend.colorNode = T.vec4(T.select(bMode.greaterThan(0.5), T.mix(tipB, rootedB, bStr), tipB), 1);
      } else if (instB) {
        blend.colorNode = T.vec4(tipB, 1);        // no atlas: only the crowd's tip colour to carry
      }
      blend.userData.hairUniforms = u;
    } else {
      blend.onBeforeCompile = (shader) => {
        shader.uniforms.uBlendDensity = { value: 2.5 };
        shader.uniforms.uHairRootMode = { value: rootMode };
        shader.uniforms.uHairRootColor = { value: rootColor };
        shader.uniforms.uHairRootStr = { value: o.rootStrength };
        shader.fragmentShader = `
          // Created by Sander Mørch-Jensen — creategamecharacters.com
          // SPDX-License-Identifier: MIT
          
          uniform float uBlendDensity;
          uniform float uHairRootMode;
          uniform vec3  uHairRootColor;
          uniform float uHairRootStr;
        ` + shader.fragmentShader;
        shader.fragmentShader = shader.fragmentShader.replace(
          '#include <map_fragment>',
          `
          #include <map_fragment>
          #ifdef USE_ALPHAMAP
            if ( uHairRootMode > 0.5 ) {
              float rootTB = texture2D( alphaMap, vAlphaMapUv ).${rootChan};
              vec3 rootedB = mix( uHairRootColor, diffuseColor.rgb, rootTB );
              diffuseColor.rgb = mix( diffuseColor.rgb, rootedB, uHairRootStr );
            }
          #endif
          `
        );
        shader.fragmentShader = shader.fragmentShader.replace(
          '#include <alphamap_fragment>',
          `
          #ifdef USE_ALPHAMAP
            float covB = texture2D( alphaMap, vAlphaMapUv ).${alphaSwz};
            if ( covB < 0.06 ) discard;
            diffuseColor.a *= clamp( covB * uBlendDensity, 0.0, 1.0 );
          #endif
          `
        );
        blend.userData.hairUniforms = shader.uniforms;
      };
    }
    blend.needsUpdate = true;
  }
  attachTone(outer, false);
  if (inner) attachTone(inner, true);
  outer.needsUpdate = true;
  if (inner) inner.needsUpdate = true;

  // `cut`/`soft` are the original two-pass key names, kept so existing callers keep
  // working: they put `cut` on the mesh and `soft` on a clone. `cut` is the outer pass —
  // the hair — and `soft` the inner core; either object may carry either, since three
  // draws the opaque bucket before the transparent one and the polygon offsets order the
  // two opaque passes.
  return { outer, inner, blend, mode, useA2C, density, cut: outer, soft: inner };
}

// Apply the passes to every hair mesh under `root`.
//
// The mesh keeps the OUTER material — that is the hair — and the inner core and (in 'both'
// mode) the blend fringe are added as mesh clones sharing the SAME BufferGeometry, so there
// is no extra geometry or texture memory, and a SkinnedMesh clone keeps its skeleton and
// deforms with the rig. This mirrors shared/viewer.js exactly: there, `mat` is the mesh's own
// material and `addHairInnerPass` / the `__blend` clone are siblings beside it.
export function applyHairShader(root, opts = {}) {
  const THREE = opts.THREE;
  if (!THREE) throw new Error('applyHairShader: pass { THREE }');
  const filter = opts.filter || ((mesh) => !/__(soft|inner|blend)$/.test(mesh.name));
  const outers = [];
  const inners = [];
  const blends = [];
  const meshes = [];
  root.traverse((o) => { if ((o.isMesh || o.isSkinnedMesh) && filter(o)) meshes.push(o); });

  for (const mesh of meshes) {
    if (Array.isArray(mesh.material)) continue;   // multi-slot hair: not a case we ship
    // Per-material overrides are keyed by the material's leading word, the way viewer.js
    // labels them: `eyebrows_*` counts as `brows`, anything unlabelled as `hair`.
    const label = (() => {
      const w = String((mesh.material && mesh.material.name) || mesh.name || '').split('_')[0].toLowerCase();
      return w === 'eyebrows' ? 'brows' : (w || 'hair');
    })();
    const per = (opts.materials && opts.materials[label]) || null;
    const { outer, inner, blend, mode } = createHairMaterials(THREE, mesh.material, {
      ...opts,
      materialName: (mesh.material && mesh.material.name) || mesh.name,
      ...(per || {}),
      // A per-material entry may carry its own colour, density, mode and fringe settings.
      color: (per && per.color) || opts.color,
      density: (per && typeof per.hair_density === 'number') ? per.hair_density : opts.density,
      mode: (per && per.mode) || opts.mode,
      blendOpacity: (per && typeof per.blend_opacity === 'number') ? per.blend_opacity : opts.blendOpacity,
      blendRoughness: (per && typeof per.blend_roughness === 'number') ? per.blend_roughness : opts.blendRoughness,
    });
    if (mesh.material && mesh.material.dispose) mesh.material.dispose();
    mesh.material = outer;

    // The core exists only in the no-MSAA fallback — see createHairMaterials.
    let innerMesh = null;
    if (inner) {
      innerMesh = mesh.clone();
      innerMesh.name = mesh.name + '__inner';
      innerMesh.material = inner;
      innerMesh.userData.hairPass = 'inner';
      (mesh.parent || root).add(innerMesh);
      inners.push(inner);
    }

    let blendMesh = null;
    if (blend) {
      blendMesh = mesh.clone();
      blendMesh.name = mesh.name + '__blend';
      blendMesh.material = blend;
      blendMesh.userData.hairPass = 'blend';
      // 'a2c' means base + core only; the fringe exists but stays hidden until 'both'.
      blendMesh.visible = mode === 'both';
      (mesh.parent || root).add(blendMesh);
      blends.push(blend);
    }

    if (inner) outer.userData.innerPass = { mesh: innerMesh, material: inner };
    if (blend) outer.userData.blendClone = { mesh: blendMesh, mat: blend };
    outers.push(outer);
  }

  const setUniform = (mats, name, value) => {
    for (const m of mats) {
      const u = m.userData.hairUniforms && m.userData.hairUniforms[name];
      if (u) u.value = value;
    }
  };
  const all = outers.concat(inners, blends);
  return {
    materials: all,
    outer: outers, inner: inners, blend: blends,
    cut: outers, soft: inners,                 // the older names
    setColor(c) { for (const m of all) m.color.set(c); },
    setDensity(v) { setUniform(outers, 'uHairDensity', v); },
    setBlendDensity(v) { setUniform(blends, 'uBlendDensity', v); },
    setInnerThreshold(v) { for (const m of inners) { m.alphaTest = v; m.needsUpdate = true; } },
    setCutThreshold(v) { this.setInnerThreshold(v); },
    setBlendOpacity(v) { for (const m of blends) m.opacity = v; },
    /** 'a2c' (base + core) | 'both' (adds the fringe). */
    setMode(mode) {
      for (const m of outers) {
        m.userData.hairMode = mode;
        const bc = m.userData.blendClone;
        if (bc) bc.mesh.visible = mode === 'both';
      }
    },
    setSoftPassVisible(on) { this.setMode(on ? 'both' : 'a2c'); },
    dispose() {
      for (const m of outers) {
        const ip = m.userData.innerPass;
        if (ip) { ip.mesh.removeFromParent(); ip.material.dispose(); }
        const bc = m.userData.blendClone;
        if (bc) { bc.mesh.removeFromParent(); bc.mat.dispose(); }
        m.dispose();
      }
    },
  };
}

// The one-call version: params + atlas straight from the style folder.
//   applyHairStyle(gltf.scene, { THREE, renderer, styleUrl: '…/_hair/Quiff', color })
export async function applyHairStyle(root, opts = {}) {
  const { THREE, styleUrl } = opts;
  if (!THREE || !styleUrl) throw new Error('applyHairStyle: pass { THREE, styleUrl }');
  const { params, textures } = await loadHairParams(styleUrl);
  const atlas = loadHairAtlas(THREE, styleUrl, { params, textures });
  return applyHairShader(root, {
    ...opts,
    atlas,
    alphaChannel: params.alpha_channel || DEFAULTS.alphaChannel,
    // Left UNDEFINED when the style does not declare one, so createHairMaterials can apply
    // the site's rule: 2.5 for A2C head hair, 1.0 for facial hair and for no-MSAA.
    density: typeof params.hair_density === 'number' ? params.hair_density : opts.density,
    // The inner core's clip. `cut_threshold` is the older name for the same number.
    innerThreshold: typeof params.inner_alpha_threshold === 'number' ? params.inner_alpha_threshold
                  : typeof params.cut_threshold === 'number' ? params.cut_threshold
                  : DEFAULTS.innerThreshold,
    roughness: typeof params.roughness === 'number' ? params.roughness : 0.5,
    roughnessFloor: typeof params.hair_roughness_floor === 'number' ? params.hair_roughness_floor : DEFAULTS.roughnessFloor,
    roughnessSeedAmp: typeof params.hair_roughness_seed_amp === 'number' ? params.hair_roughness_seed_amp : DEFAULTS.roughnessSeedAmp,
    rootDarkening: typeof params.root_darkening === 'number' ? params.root_darkening : DEFAULTS.rootDarkening,
    // The style's own root mode/colour, unless the caller passed one (a colour swatch does).
    rootMode: opts.rootMode || params.root_mode || DEFAULTS.rootMode,
    rootColor: opts.rootColor || params.root_color || DEFAULTS.rootColor,
    rootStrength: typeof opts.rootStrength === 'number' ? opts.rootStrength
                : typeof params.root_strength === 'number' ? params.root_strength : DEFAULTS.rootStrength,
    seedVariation: typeof params.seed_variation === 'number' ? params.seed_variation : DEFAULTS.seedVariation,
    anisotropy: typeof params.anisotropy === 'number' ? params.anisotropy : DEFAULTS.anisotropy,
    anisotropyRotation: typeof params.anisotropy_rotation === 'number' ? params.anisotropy_rotation : DEFAULTS.anisotropyRotation,
    blendOpacity: typeof params.blend_opacity === 'number' ? params.blend_opacity : DEFAULTS.blendOpacity,
    blendRoughness: typeof params.blend_roughness === 'number' ? params.blend_roughness : DEFAULTS.blendRoughness,
    color: opts.color || DEFAULTS.color,
  });
}

export default applyHairStyle;
