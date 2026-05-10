// heapEffect — effect introspection + bucket-layout generation.
//
// Two stages:
//
//   1. `compileHeapEffect(effect)` — runs `effect.compile({target:"wgsl"})`
//      and returns the raw stage WGSL (untouched) plus a `schema`
//      describing the effect's surviving inputs / uniforms / outputs
//      (post link + DCE). No name hardcoding; everything is read off
//      `CompiledEffect.interface`.
//
//   2. `buildBucketLayout(schema, hasTextures, opts)` — turns a schema
//      into a concrete per-bucket WGSL prelude + a JS-side layout
//      object that knows where each named uniform / attribute lives
//      (struct field offset for packing). The IR rewriter in
//      `heapEffectIR.ts` consumes this layout to substitute uniform /
//      attribute reads with heap loads at the IR level.

import type { Effect } from "@aardworx/wombat.shader";

// ─── Schema (extracted from CompiledEffect.interface) ───────────────

export interface HeapEffectInput {
  readonly name: string;
  readonly wgslType: string;
  readonly byteSize: number;        // tightly-packed (vec3<f32> = 12, mat4 = 64)
  readonly location?: number;       // present on vertex-stage attributes
}

/**
 * One field of the VS-output / FS-input inter-stage struct ("VsOut"
 * conventionally). Drives the per-effect VsOut struct generated into
 * the prelude — no hardcoded clipPos/worldPos/etc. The runtime stays
 * agnostic to varying names, just like uniforms.
 */
export interface HeapVarying {
  readonly name: string;
  readonly wgslType: string;
  readonly location?: number;
  readonly builtin?: string;
  readonly interpolation?: "smooth" | "flat" | "centroid" | "sample" | "no-perspective";
}

/** One texture binding the effect declares (post link + DCE). */
export interface HeapTextureBinding {
  readonly name: string;
  /** WGSL type expression, e.g. "texture_2d<f32>", "texture_cube<f32>". */
  readonly wgslType: string;
}

/** One sampler binding the effect declares (post link + DCE). */
export interface HeapSamplerBinding {
  readonly name: string;
  /** WGSL type — "sampler" or "sampler_comparison". */
  readonly wgslType: string;
}

export interface HeapEffectSchema {
  readonly attributes: readonly HeapEffectInput[];
  readonly uniforms: readonly HeapEffectInput[];
  readonly varyings: readonly HeapVarying[];
  readonly fragmentOutputs: readonly { readonly name: string; readonly location: number; readonly wgslType: string }[];
  readonly textures: readonly HeapTextureBinding[];
  readonly samplers: readonly HeapSamplerBinding[];
}

export interface CompiledHeapEffect {
  readonly rawVs: string;
  readonly rawFs: string;
  readonly schema: HeapEffectSchema;
}

/**
 * Maps fragment-output names to framebuffer locations. Outputs not
 * in the map are pruned by `linkFragmentOutputs`, which cascades
 * through `pruneCrossStage` / `reduceUniforms` — uniforms that only
 * fed the dropped outputs vanish from the schema entirely.
 */
export interface FragmentOutputLayout {
  readonly locations: ReadonlyMap<string, number>;
}

export function compileHeapEffect(effect: Effect, fragmentOutputLayout?: FragmentOutputLayout): CompiledHeapEffect {
  const compiled = effect.compile(
    fragmentOutputLayout !== undefined
      ? { target: "wgsl", fragmentOutputLayout }
      : { target: "wgsl" },
  );
  const vsStage = compiled.stages.find(s => s.stage === "vertex");
  const fsStage = compiled.stages.find(s => s.stage === "fragment");
  return {
    rawVs: vsStage?.source ?? "",
    rawFs: fsStage?.source ?? "",
    schema: buildSchema(compiled.interface),
  };
}

// ─── IR-Type → WGSL-string + packed-size helpers ────────────────────

type ProgramInterface = ReturnType<Effect["compile"]>["interface"];
type IrType = ProgramInterface["uniformBlocks"][number]["fields"][number]["type"];

function buildSchema(iface: ProgramInterface): HeapEffectSchema {
  const attributes: HeapEffectInput[] = iface.attributes.map(a => ({
    name: a.name,
    wgslType: irTypeToWgsl(a.type),
    byteSize: a.byteSize,
    location: a.location,
  }));
  const uniforms: HeapEffectInput[] = [];
  for (const block of iface.uniformBlocks) {
    for (const f of block.fields) {
      uniforms.push({
        name: f.name,
        wgslType: irTypeToWgsl(f.type),
        byteSize: irTypePackedSize(f.type),
      });
    }
  }
  // GLSL-style "loose" uniforms (no enclosing buffer name). On WGSL these
  // emit as individual `var<uniform>` decls; the heap IR rewrite substitutes
  // each `ReadInput("Uniform", name, ...)` with a per-draw heap load before
  // WGSL emit, so the loose decls never reach the final shader.
  for (const u of iface.uniforms) {
    uniforms.push({
      name: u.name,
      wgslType: irTypeToWgsl(u.type),
      byteSize: irTypePackedSize(u.type),
    });
  }
  // Inter-stage varyings: VS outputs (post link + DCE). The FS inputs
  // mirror these — `linkCrossStage` aligns them by semantic, so we
  // can use VS outputs as the single source of truth for VsOut.
  const vsStage = iface.stages.find(s => s.stage === "vertex");
  const varyings: HeapVarying[] = vsStage === undefined
    ? []
    : vsStage.outputs.map(o => {
        const v: {
          name: string; wgslType: string;
          location?: number; builtin?: string;
          interpolation?: "smooth" | "flat" | "centroid" | "sample" | "no-perspective";
        } = { name: o.name, wgslType: irTypeToWgsl(o.type) };
        if (o.location      !== undefined) v.location      = o.location;
        if (o.builtin       !== undefined) v.builtin       = o.builtin;
        if (o.interpolation !== undefined) v.interpolation = o.interpolation;
        return v as HeapVarying;
      });
  const fragmentOutputs = iface.fragmentOutputs.map(o => ({
    name: o.name, location: o.location, wgslType: irTypeToWgsl(o.type),
  }));
  // Texture/sampler bindings post-DCE. The IR Type for textures is a
  // structured kind we don't fully model here yet — for v1 every
  // texture maps to "texture_2d<f32>" and every sampler to plain
  // "sampler". Specialise as multi-kind support lands in the DSL.
  const textures: HeapTextureBinding[] = iface.textures.map(t => ({
    name: t.name, wgslType: "texture_2d<f32>",
  }));
  const samplers: HeapSamplerBinding[] = iface.samplers.map(s => ({
    name: s.name, wgslType: "sampler",
  }));
  return { attributes, uniforms, varyings, fragmentOutputs, textures, samplers };
}

function irTypeToWgsl(t: IrType): string {
  switch (t.kind) {
    case "Float":  return "f32";
    case "Int":    return t.signed ? "i32" : "u32";
    case "Bool":   return "bool";
    case "Vector": return `vec${t.dim}<${irTypeToWgsl(t.element)}>`;
    case "Matrix": return `mat${t.cols}x${t.rows}<${irTypeToWgsl(t.element)}>`;
    case "Array": {
      const inner = irTypeToWgsl(t.element);
      return t.length === "runtime" ? `array<${inner}>` : `array<${inner}, ${t.length}>`;
    }
    default:
      throw new Error(`compileHeapEffect: cannot map IR Type kind '${t.kind}' to WGSL`);
  }
}

function irTypePackedSize(t: IrType): number {
  switch (t.kind) {
    case "Float":  return 4;
    case "Int":    return 4;
    case "Bool":   return 4;
    case "Vector": return t.dim * irTypePackedSize(t.element);
    case "Matrix": return t.cols * t.rows * irTypePackedSize(t.element);
    case "Array": {
      if (t.length === "runtime") {
        throw new Error("compileHeapEffect: runtime-sized arrays have no fixed packed size");
      }
      return t.length * irTypePackedSize(t.element);
    }
    default:
      throw new Error(`compileHeapEffect: cannot size IR Type kind '${t.kind}'`);
  }
}

// ─── Bucket layout ──────────────────────────────────────────────────

export interface DrawHeaderField {
  readonly name: string;            // schema name (e.g. "ModelTrafo")
  readonly wgslName: string;        // WGSL field name (e.g. "modelTrafoRef")
  readonly wgslType: string;        // "u32" for refs; "scalar"-packed for atlas texture-ref entries
  readonly byteOffset: number;
  readonly byteSize: number;
  readonly kind: "uniform-ref" | "attribute-ref" | "texture-ref";
  /**
   * For `uniform-ref`: the uniform's underlying WGSL type. The
   * rewriter uses this to pick the right inline read expression
   * (mat4x4 → 4 vec4 reads, vec3 → 3 f32 reads, etc.).
   */
  readonly uniformWgslType?: string;
  /**
   * For `attribute-ref`: the attribute's declared WGSL type. Drives
   * the inline `attributeLoadExpr` when generating preamble lets.
   */
  readonly attributeWgslType?: string;
  /**
   * For `texture-ref`: which sub-entry this is. Atlas-variant texture
   * bindings expand into four contiguous drawHeader entries —
   * `pageRef` (u32 slot index), `formatBits` (u32 packed sampler
   * state + mip + format), `origin` (vec2<f32>, mip-0 top-left in
   * normalized atlas coords), `size` (vec2<f32>, mip-0 size). The
   * runtime packs these as inline values (not pool refs); the
   * shader pass that consumes them lands in a follow-up PR.
   */
  readonly textureSub?: "pageRef" | "formatBits" | "origin" | "size";
  /** For `texture-ref`: the schema's logical texture binding name (shared by all four sub-entries). */
  readonly textureBindingName?: string;
}

export interface BucketLayout {
  readonly drawHeaderFields: readonly DrawHeaderField[];
  readonly drawHeaderBytes: number;
  /** Generated WGSL prelude — bindings + per-effect VsOut struct. No helper fns, no DrawHeader struct. */
  readonly preludeWgsl: string;
  /** DrawHeader stride in u32 elements (drawHeaderBytes / 4). */
  readonly strideU32: number;
  /**
   * Names (schema-side) of uniforms that vary per instance. Populated
   * from `spec.instances.values` at addDraw time; the rewriter pulls
   * these via `instanceLoadExpr` instead of `uniformLoadExpr`.
   */
  readonly perInstanceUniforms: ReadonlySet<string>;
  /**
   * Names (schema-side) of attributes that vary per instance. Populated
   * from `spec.instanceAttributes` at addDraw time. The rewriter
   * indexes per-instance attribute reads by `iidx` (= `instance_index`
   * the megacall path) instead of `vid`.
   */
  readonly perInstanceAttributes: ReadonlySet<string>;
  /**
   * Texture bindings: name + WGSL type + bind-group binding number
   * (allocated starting at TEX_BINDING_START, after the four heap-
   * data bindings 0–3 and the three megacall bindings 4–6).
   */
  readonly textureBindings: readonly { readonly name: string; readonly wgslType: string; readonly binding: number }[];
  /** Sampler bindings, allocated after textures. */
  readonly samplerBindings: readonly { readonly name: string; readonly wgslType: string; readonly binding: number }[];
  /**
   * Atlas-routed texture binding names, exposed for the FS rewriter so
   * it can substitute `textureSample(name, smp, uv)` with the heap
   * `atlasSample(...)` helper for these names only.
   */
  readonly atlasTextureBindings: ReadonlySet<string>;
}

/**
 * First texture binding number; samplers come after the textures.
 * Bindings 0–3 are the four heap arena views, 4–6 are megacall
 * (drawTable, indexStorage, firstDrawInTile).
 */
export const HEAP_TEX_BINDING_START = 7;

/**
 * Number of independent pages per format. Drives both the BGL slot
 * count (N consecutive linear bindings + N consecutive srgb bindings
 * + 1 sampler) and the shader's `switch pageRef` ladder. Must match
 * `ATLAS_MAX_PAGES_PER_FORMAT` in `textureAtlas/atlasPool.ts`.
 */
export const ATLAS_ARRAY_SIZE = 8;

/** First atlas binding (linear texture for page 0). */
export const ATLAS_LINEAR_BINDING_BASE = 11;
/** First srgb atlas binding (= linear base + N). */
export const ATLAS_SRGB_BINDING_BASE = ATLAS_LINEAR_BINDING_BASE + ATLAS_ARRAY_SIZE;
/** Atlas sampler binding (= linear base + 2N). */
export const ATLAS_SAMPLER_BINDING = ATLAS_LINEAR_BINDING_BASE + 2 * ATLAS_ARRAY_SIZE;

export function buildBucketLayout(
  schema: HeapEffectSchema,
  _hasTextures: boolean,
  opts: {
    perInstanceUniforms?: ReadonlySet<string>;
    perInstanceAttributes?: ReadonlySet<string>;
    /**
     * Names of schema texture bindings routed via the atlas binding-array
     * path. Each such binding is dropped from `textureBindings` /
     * `samplerBindings` and replaced with four `texture-ref` drawHeader
     * fields (pageRef / formatBits / origin / size). Pass an empty set
     * (default) for the standalone texture path.
     */
    atlasTextureBindings?: ReadonlySet<string>;
  } = {},
): BucketLayout {
  const perInstanceUniforms = opts.perInstanceUniforms ?? new Set<string>();
  const perInstanceAttributes = opts.perInstanceAttributes ?? new Set<string>();
  const atlasTextureBindings = opts.atlasTextureBindings ?? new Set<string>();
  const drawHeaderFields: DrawHeaderField[] = [];

  // Per-draw uniforms → u32 ref slots in the DrawHeader. The runtime
  // (UniformPool) keeps one arena allocation per unique aval; sharing
  // across draws → shared ref → shared upload.
  let dhOff = 0;
  for (const u of schema.uniforms) {
    dhOff = roundUp(dhOff, 4);
    drawHeaderFields.push({
      name: u.name, wgslName: lowerFirst(u.name) + "Ref", wgslType: "u32",
      byteOffset: dhOff, byteSize: 4, kind: "uniform-ref",
      uniformWgslType: u.wgslType,
    });
    dhOff += 4;
  }

  // Attribute refs — same shape as uniform refs, also routed through the pool.
  for (const a of schema.attributes) {
    dhOff = roundUp(dhOff, 4);
    drawHeaderFields.push({
      name: a.name, wgslName: lowerFirst(a.name) + "Ref", wgslType: "u32",
      byteOffset: dhOff, byteSize: 4, kind: "attribute-ref",
      attributeWgslType: a.wgslType,
    });
    dhOff += 4;
  }

  // Atlas-routed texture bindings: 24 bytes / binding split into
  // pageRef (u32) + formatBits (u32) + origin (vec2<f32>) + size
  // (vec2<f32>). Inline values, not pool refs — the runtime packs
  // them at addDraw time from the atlas acquisition + ISampler.
  for (const t of schema.textures) {
    if (!atlasTextureBindings.has(t.name)) continue;
    const base = lowerFirst(t.name);
    dhOff = roundUp(dhOff, 4);
    drawHeaderFields.push({
      name: `${t.name}.pageRef`, wgslName: `${base}PageRef`, wgslType: "u32",
      byteOffset: dhOff, byteSize: 4, kind: "texture-ref",
      textureSub: "pageRef", textureBindingName: t.name,
    });
    dhOff += 4;
    drawHeaderFields.push({
      name: `${t.name}.formatBits`, wgslName: `${base}FormatBits`, wgslType: "u32",
      byteOffset: dhOff, byteSize: 4, kind: "texture-ref",
      textureSub: "formatBits", textureBindingName: t.name,
    });
    dhOff += 4;
    dhOff = roundUp(dhOff, 8);
    drawHeaderFields.push({
      name: `${t.name}.origin`, wgslName: `${base}Origin`, wgslType: "vec2<f32>",
      byteOffset: dhOff, byteSize: 8, kind: "texture-ref",
      textureSub: "origin", textureBindingName: t.name,
    });
    dhOff += 8;
    drawHeaderFields.push({
      name: `${t.name}.size`, wgslName: `${base}Size`, wgslType: "vec2<f32>",
      byteOffset: dhOff, byteSize: 8, kind: "texture-ref",
      textureSub: "size", textureBindingName: t.name,
    });
    dhOff += 8;
  }

  const drawHeaderBytes = roundUp(dhOff, 16);
  const strideU32 = drawHeaderBytes / 4;

  // Allocate texture + sampler bindings starting at HEAP_TEX_BINDING_START.
  // Order: all textures, then all samplers — matches the prelude
  // layout below and the runtime's BGL/bind-group construction.
  // Atlas-routed bindings drop out — their data flows through the
  // drawHeader instead of through individual texture/sampler slots.
  let nextBinding = HEAP_TEX_BINDING_START;
  const textureBindings = schema.textures
    .filter(t => !atlasTextureBindings.has(t.name))
    .map(t => ({ name: t.name, wgslType: t.wgslType, binding: nextBinding++ }));
  // Samplers: drop the matching one when its texture is atlas-routed.
  // Heuristic: assume position-aligned (nth sampler pairs with nth
  // texture). Heap path's v1 surface only ever has 0..1 of each, so
  // this collapses to "if the lone texture is atlas, drop the lone
  // sampler too".
  const atlasSamplerNames = new Set<string>();
  for (let i = 0; i < schema.textures.length; i++) {
    const t = schema.textures[i]!;
    if (atlasTextureBindings.has(t.name)) {
      const s = schema.samplers[i];
      if (s !== undefined) atlasSamplerNames.add(s.name);
    }
  }
  const samplerBindings = schema.samplers
    .filter(s => !atlasSamplerNames.has(s.name))
    .map(s => ({ name: s.name, wgslType: s.wgslType, binding: nextBinding++ }));

  const isAtlasBucket = atlasTextureBindings.size > 0;
  // Extend the schema's varyings with flat-interpolated atlas threading
  // slots when atlas is in use — the prelude generator emits these into
  // VsOut so user VS code can write them and FS can read them. The
  // location numbers are picked after the existing varyings.
  const schemaForPrelude = isAtlasBucket
    ? extendSchemaWithAtlasVaryings(schema, atlasTextureBindings)
    : schema;
  const preludeWgsl = generatePrelude(
    schemaForPrelude, textureBindings, samplerBindings,
    isAtlasBucket,
  );

  return {
    drawHeaderFields, drawHeaderBytes, preludeWgsl, strideU32,
    perInstanceUniforms,
    perInstanceAttributes,
    textureBindings, samplerBindings,
    atlasTextureBindings,
  };
}

/**
 * Append flat-interpolated atlas threading varyings to a schema.
 * Locations follow the highest existing `@location(N)` so we don't
 * collide with effect-declared varyings.
 */
function extendSchemaWithAtlasVaryings(
  schema: HeapEffectSchema,
  atlasNames: ReadonlySet<string>,
): HeapEffectSchema {
  let nextLoc = 0;
  for (const v of schema.varyings) {
    if (v.location !== undefined) nextLoc = Math.max(nextLoc, v.location + 1);
  }
  const extras: HeapVarying[] = [];
  for (const name of atlasNames) {
    const v = atlasVaryingNames(name);
    extras.push({ name: v.pageRef,    wgslType: "u32",        location: nextLoc++, interpolation: "flat" });
    extras.push({ name: v.formatBits, wgslType: "u32",        location: nextLoc++, interpolation: "flat" });
    extras.push({ name: v.origin,     wgslType: "vec2<f32>",  location: nextLoc++, interpolation: "flat" });
    extras.push({ name: v.size,       wgslType: "vec2<f32>",  location: nextLoc++, interpolation: "flat" });
  }
  return { ...schema, varyings: [...schema.varyings, ...extras] };
}

function lowerFirst(s: string): string {
  return s.length === 0 ? s : s[0]!.toLowerCase() + s.slice(1);
}

function roundUp(value: number, mult: number): number {
  return Math.ceil(value / mult) * mult;
}

// ─── Prelude generation ─────────────────────────────────────────────

function generatePrelude(
  schema: HeapEffectSchema,
  textureBindings: readonly { name: string; wgslType: string; binding: number }[],
  samplerBindings: readonly { name: string; wgslType: string; binding: number }[],
  isAtlasBucket: boolean,
): string {
  // VsOut is generated from the schema's surviving VS outputs (post
  // link + DCE). No hardcoded fields — what the effect declared is
  // what ends up here.
  const vsOutBody = schema.varyings.map(v => `  ${vsOutFieldDecl(v)}: ${v.wgslType},`).join("\n");

  // Textures + samplers — one binding per surviving entry, starting
  // at HEAP_TEX_BINDING_START. The user's WGSL must reference them
  // by their schema name; the rewriter drops any DSL-emitted decls
  // that conflict.
  const texDecls = textureBindings.map(t =>
    `@group(0) @binding(${t.binding}) var ${t.name}: ${t.wgslType};`,
  ).join("\n");
  const smpDecls = samplerBindings.map(s =>
    `@group(0) @binding(${s.binding}) var ${s.name}: ${s.wgslType};`,
  ).join("\n");
  const samplingBlock = (texDecls.length > 0 || smpDecls.length > 0)
    ? `\n${texDecls}${texDecls && smpDecls ? "\n" : ""}${smpDecls}\n`
    : "";

  const megacallDecls = `\n@group(0) @binding(4) var<storage, read> drawTable:       array<u32>;\n@group(0) @binding(5) var<storage, read> indexStorage:    array<u32>;\n@group(0) @binding(6) var<storage, read> firstDrawInTile: array<u32>;`;

  const atlasBlock = isAtlasBucket ? generateAtlasPrelude() : "";

  return /* wgsl */`
@group(0) @binding(0) var<storage, read> heapU32:    array<u32>;
@group(0) @binding(1) var<storage, read> headersU32: array<u32>;
@group(0) @binding(2) var<storage, read> heapF32:    array<f32>;
@group(0) @binding(3) var<storage, read> heapV4f:    array<vec4<f32>>;${megacallDecls}
${samplingBlock}${atlasBlock}
struct VsOut {
${vsOutBody}
};
`;
}

/**
 * Atlas prelude: declares N independent `texture_2d<f32>` bindings
 * per format (linear + srgb), plus a shared atlas sampler. The N-way
 * `switch pageRef` ladder picks the matching texture pair for each
 * draw. Each atlas page is its own `GPUTexture` — adding pages to a
 * format is just a fresh allocation slotted into the next BGL entry,
 * no GPU-side `copyTextureToTexture` required.
 *
 * `binding_array<texture_2d<f32>, N>` would collapse the switch into
 * a runtime index but requires Chrome's experimental bindless
 * feature; the N-consecutive-bindings + switch pattern is native
 * WebGPU 1.0.
 *
 * Mip filtering is software (the GPU's mip walk would walk the
 * texture's mip chain, not our embedded 1.5×1 pyramid). LOD is
 * computed from screen-space derivatives in mip-0 atlas-pixel
 * space; the constant 4096 matches `ATLAS_PAGE_SIZE`.
 *
 * Wrap-mode mirror math: `1 - |((u - floor(u/2)*2) - 1)|` yields a
 * triangle wave with period 2 and amplitude [0,1] — matches WebGPU's
 * `mirror-repeat` address-mode spec. Verified at u={0,0.5,1,1.5,2,-0.5}.
 */
function generateAtlasPrelude(): string {
  return /* wgsl */`
${generateAtlasBindings()}

fn atlasWrap1(u: f32, mode: u32) -> f32 {
  let r = u - floor(u);
  let m = 1.0 - abs((u - floor(u * 0.5) * 2.0) - 1.0);
  let c = clamp(u, 0.0, 1.0);
  return select(select(c, r, mode == 1u), m, mode == 2u);
}

fn atlasApplyWrap(uv: vec2<f32>, addrU: u32, addrV: u32) -> vec2<f32> {
  return vec2<f32>(atlasWrap1(uv.x, addrU), atlasWrap1(uv.y, addrV));
}

fn atlasMipOrigin(origin: vec2<f32>, size: vec2<f32>, k: u32) -> vec2<f32> {
  if (k == 0u) { return origin; }
  let x = origin.x + size.x;
  let y = origin.y + size.y * (1.0 - 1.0 / pow(2.0, f32(k) - 1.0));
  return vec2<f32>(x, y);
}

fn atlasSampleAtMip(
  pageRef: u32, format: u32,
  origin: vec2<f32>, size: vec2<f32>,
  k: u32,
  uvW: vec2<f32>,
) -> vec4<f32> {
  let mipSize = size / pow(2.0, f32(k));
  let mipO = atlasMipOrigin(origin, size, k);
  let atlasUv = mipO + uvW * mipSize;
${generateAtlasSwitch()}
  return vec4<f32>(0.0);
}

fn atlasSample(
  pageRef: u32, formatBits: u32,
  origin: vec2<f32>, size: vec2<f32>,
  uv: vec2<f32>,
) -> vec4<f32> {
  let format  = formatBits & 0x1u;
  let numMips = (formatBits >> 1u) & 0x7u;
  let addrU   = (formatBits >> 4u) & 0x3u;
  let addrV   = (formatBits >> 6u) & 0x3u;
  let uvW = atlasApplyWrap(uv, addrU, addrV);
  // Compute derivatives unconditionally — WGSL forbids dpdx/dpdy in
  // non-uniform control flow, and the per-page switch in
  // atlasSampleAtMip taints anything below it. LOD is clamped so the
  // numMips==1 case still picks mip 0.
  let dx = dpdx(uvW * size);
  let dy = dpdy(uvW * size);
  let rho = max(length(dx), length(dy)) * 4096.0;
  let maxLod = f32(max(numMips, 1u) - 1u);
  let lod = clamp(log2(max(rho, 1e-6)), 0.0, maxLod);
  let lo = u32(floor(lod));
  let hi = min(lo + 1u, max(numMips, 1u) - 1u);
  let t  = lod - f32(lo);
  let a = atlasSampleAtMip(pageRef, format, origin, size, lo, uvW);
  if (numMips <= 1u) { return a; }
  let b = atlasSampleAtMip(pageRef, format, origin, size, hi, uvW);
  return mix(a, b, t);
}
`;
}

/**
 * Emit `2 * ATLAS_ARRAY_SIZE + 1` binding declarations: N linear
 * `texture_2d<f32>`, N srgb `texture_2d<f32>`, then the shared
 * atlas sampler.
 */
export function generateAtlasBindings(): string {
  const lines: string[] = [];
  for (let i = 0; i < ATLAS_ARRAY_SIZE; i++) {
    lines.push(`@group(0) @binding(${ATLAS_LINEAR_BINDING_BASE + i}) var atlasLinear${i}: texture_2d<f32>;`);
  }
  for (let i = 0; i < ATLAS_ARRAY_SIZE; i++) {
    lines.push(`@group(0) @binding(${ATLAS_SRGB_BINDING_BASE + i}) var atlasSrgb${i}: texture_2d<f32>;`);
  }
  lines.push(`@group(0) @binding(${ATLAS_SAMPLER_BINDING}) var atlasSampler: sampler;`);
  return lines.join("\n");
}

/**
 * Emit the `switch pageRef` ladder body for `atlasSampleAtMip`. Each
 * case picks the matching `atlasLinear<i>` / `atlasSrgb<i>` decl and
 * blends on `format == 1u`. The fallback (page out of range) is the
 * function-level `vec4<f32>(0.0)` after the switch.
 */
export function generateAtlasSwitch(): string {
  const lines: string[] = ["  switch pageRef {"];
  for (let i = 0; i < ATLAS_ARRAY_SIZE; i++) {
    lines.push(`    case ${i}u: {`);
    lines.push(`      let lin = textureSampleLevel(atlasLinear${i}, atlasSampler, atlasUv, 0.0);`);
    lines.push(`      let sr  = textureSampleLevel(atlasSrgb${i},   atlasSampler, atlasUv, 0.0);`);
    lines.push(`      return select(lin, sr, format == 1u);`);
    lines.push(`    }`);
  }
  lines.push("    default: {}");
  lines.push("  }");
  return lines.join("\n");
}

/** Render one varying as the WGSL field declaration: `[@builtin(...) | @location(N)] [@interpolate(...)] name`. */
function vsOutFieldDecl(v: HeapVarying): string {
  if (v.builtin !== undefined) return `@builtin(${v.builtin}) ${v.name}`;
  const interp = v.interpolation !== undefined ? ` @interpolate(${v.interpolation})` : "";
  if (v.location === undefined) {
    throw new Error(`heapEffect: varying '${v.name}' has neither @builtin nor @location`);
  }
  return `@location(${v.location})${interp} ${v.name}`;
}

/** Atlas-texture varying names used by the VS→FS threading + FS preamble. */
export function atlasVaryingNames(name: string): {
  pageRef: string; formatBits: string; origin: string; size: string;
} {
  return {
    pageRef:    `_h_${name}PageRef`,
    formatBits: `_h_${name}FormatBits`,
    origin:     `_h_${name}Origin`,
    size:       `_h_${name}Size`,
  };
}


/**
 * Megacall VS prelude: binary-search `drawTable` (one record per slot,
 * 5 u32s: firstEmit, drawIdx, indexStart, indexCount, instanceCount)
 * for the slot owning the given `emitIdx`, then split the in-record
 * offset into `(instId, vid)` so per-RO instancing collapses to one
 * record + one drawIndirect. Defines:
 *   - `heap_drawIdx` — per-RO selector for header reads.
 *   - `instId`         — instance index in [0, instanceCount-1].
 *   - `vid`            — vertex index pulled from indexStorage.
 *
 * Layout: each record emits `indexCount * instanceCount` vertices.
 * `local = emitIdx - firstEmit`, `instId = local / indexCount`,
 * `vid = indices[indexStart + local % indexCount]`.
 */
export function megacallSearchPrelude(): string {
  // The shared megacall values (`heap_drawIdx`, `instId`, `vid`) are
  // declared as `let` locals inside the @vertex fn body. Wombat.shader's
  // composed-stage helper functions (extractFusedEntry) that reference
  // these identifiers are post-processed in `applyMegacallToEmittedVs`
  // to receive them as `u32` parameters — module-scope `var<private>`
  // is rejected by some WGSL parsers (Safari/WebKit), so we thread the
  // values through helper signatures instead.
  return `  let _tileIdx = emitIdx >> 6u;\n  var lo: u32 = firstDrawInTile[_tileIdx];\n  var hi: u32 = firstDrawInTile[_tileIdx + 1u];\n  loop {\n    if (lo >= hi) { break; }\n    let _mid = (lo + hi + 1u) >> 1u;\n    if (drawTable[_mid * 6u] <= emitIdx) { lo = _mid; } else { hi = _mid - 1u; }\n  }\n  let _slot = lo;\n  let _firstEmit  = drawTable[_slot * 6u + 0u];\n  let heap_drawIdx: u32 = drawTable[_slot * 6u + 1u];\n  let _indexStart = drawTable[_slot * 6u + 2u];\n  let _indexCount = drawTable[_slot * 6u + 3u];\n  let _local      = emitIdx - _firstEmit;\n  let instId: u32 = _local / _indexCount;\n  let vid: u32    = indexStorage[_indexStart + (_local % _indexCount)];\n`;
}
