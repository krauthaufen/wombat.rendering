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
//   2. `buildBucketLayout(schema, classifier, hasTextures)` — turns
//      a schema into a concrete per-bucket WGSL prelude + a JS-side
//      layout object that knows where each named uniform / attribute
//      lives, both as a struct field offset (for packing) and as a
//      WGSL access expression (for rewriting the stage bodies).
//
//   3. `rewriteForLayout(rawVs, rawFs, layout)` — rewrites the raw
//      stage WGSL to use the layout's WGSL access expressions instead
//      of the DSL-emitted `_w_uniform.X` / param.X reads.
//
// The CLASSIFIER decides whether a uniform lives in the per-draw
// DrawHeader (one copy per draw) or the per-frame Globals UBO (one
// copy per group). For step 2 it's a tiny hardcoded map; step 4
// drops the distinction entirely (everything is per-draw, sharing
// emerges from aval identity).

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
   * True if this bucket holds an instanced draw (shape 2). The bucket has
   * a single slot (slot=0); WGSL reads `iidx = instance_index` and uses
   * `instanceLoadExpr` for any uniform listed in `perInstanceUniforms`.
   */
  readonly isInstanced: boolean;
  /**
   * Names (schema-side) of uniforms that vary per instance. Populated
   * from `spec.instances.values` at addDraw time; the rewriter pulls
   * these via `instanceLoadExpr` instead of `uniformLoadExpr`.
   */
  readonly perInstanceUniforms: ReadonlySet<string>;
  /**
   * Texture bindings: name + WGSL type + bind-group binding number
   * (allocated starting at TEX_BINDING_START, after the four heap-
   * data bindings 0–3).
   */
  readonly textureBindings: readonly { readonly name: string; readonly wgslType: string; readonly binding: number }[];
  /** Sampler bindings, allocated after textures. */
  readonly samplerBindings: readonly { readonly name: string; readonly wgslType: string; readonly binding: number }[];
  /**
   * Megacall mode: one `pass.draw(totalEmit, 1, 0, 0)` per bucket.
   * The shader binary-searches a per-bucket drawTable to figure out
   * which slot a given vertex emit belongs to, then vertex-pulls the
   * index. Adds two storage-buffer bindings (drawTable at 4,
   * indexStorage at 5); textures shift to 6+.
   */
  readonly megacall: boolean;
  /**
   * Atlas-routed texture binding names, exposed for the FS rewriter so
   * it can substitute `textureSample(name, smp, uv)` with the heap
   * `atlasSample(...)` helper for these names only.
   */
  readonly atlasTextureBindings: ReadonlySet<string>;
  /**
   * Atlas binding-array size (N in `binding_array<texture_2d<f32>, N>`),
   * mirrored from the heapScene's per-format slot count. The prelude
   * substitutes this at codegen time. Always present (>=1) so the
   * shader's BGL declarations match the runtime BGL even for buckets
   * that don't currently route any texture through the atlas.
   */
  readonly atlasArraySize: number;
}

/** First texture binding number; samplers come after the textures. */
export const HEAP_TEX_BINDING_START = 4;

/** Fixed atlas binding numbers — must match the runtime BGL in `heapScene.ts`. */
export const ATLAS_BINDING_LINEAR  = 11;
export const ATLAS_BINDING_SRGB    = 12;
export const ATLAS_BINDING_SAMPLER = 13;

export function buildBucketLayout(
  schema: HeapEffectSchema,
  _hasTextures: boolean,
  opts: {
    isInstanced?: boolean;
    perInstanceUniforms?: ReadonlySet<string>;
    megacall?: boolean;
    /**
     * Names of schema texture bindings routed via the atlas binding-array
     * path. Each such binding is dropped from `textureBindings` /
     * `samplerBindings` and replaced with four `texture-ref` drawHeader
     * fields (pageRef / formatBits / origin / size). Pass an empty set
     * (default) for the standalone texture path.
     */
    atlasTextureBindings?: ReadonlySet<string>;
    /**
     * Atlas binding-array slot count (`binding_array<texture_2d<f32>, N>`).
     * Defaults to 8 if omitted; the runtime always passes the value
     * derived from `device.limits.maxSampledTexturesPerShaderStage`.
     */
    atlasArraySize?: number;
  } = {},
): BucketLayout {
  const isInstanced = opts.isInstanced ?? false;
  const perInstanceUniforms = opts.perInstanceUniforms ?? new Set<string>();
  const megacall = opts.megacall ?? false;
  const atlasTextureBindings = opts.atlasTextureBindings ?? new Set<string>();
  const atlasArraySize = Math.max(1, opts.atlasArraySize ?? 8);
  if (megacall && isInstanced) {
    throw new Error("heapEffect: megacall + isInstanced not supported");
  }
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
  let nextBinding = megacall ? HEAP_TEX_BINDING_START + 2 : HEAP_TEX_BINDING_START;
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
    schemaForPrelude, textureBindings, samplerBindings, megacall,
    isAtlasBucket, atlasArraySize,
  );

  return {
    drawHeaderFields, drawHeaderBytes, preludeWgsl, strideU32,
    isInstanced, perInstanceUniforms,
    textureBindings, samplerBindings,
    megacall,
    atlasTextureBindings,
    atlasArraySize,
  };
}

/** Inline WGSL expression for a per-instance uniform read (indexed by `iidx`). */
function instanceLoadExpr(wgslType: string, refIdent: string, iidxIdent: string): string {
  switch (wgslType) {
    case "mat4x4<f32>": {
      const b = `((${refIdent} + 16u) / 16u + ${iidxIdent} * 4u)`;
      return `mat4x4<f32>(heapV4f[${b}], heapV4f[${b} + 1u], heapV4f[${b} + 2u], heapV4f[${b} + 3u])`;
    }
    case "vec4<f32>": return `heapV4f[(${refIdent} + 16u) / 16u + ${iidxIdent}]`;
    case "vec3<f32>": {
      const b = `((${refIdent} + 16u) / 4u + ${iidxIdent} * 3u)`;
      return `vec3<f32>(heapF32[${b}], heapF32[${b} + 1u], heapF32[${b} + 2u])`;
    }
    case "vec2<f32>": {
      const b = `((${refIdent} + 16u) / 4u + ${iidxIdent} * 2u)`;
      return `vec2<f32>(heapF32[${b}], heapF32[${b} + 1u])`;
    }
    case "f32": return `heapF32[(${refIdent} + 16u) / 4u + ${iidxIdent}]`;
    case "u32": return `heapU32[(${refIdent} + 16u) / 4u + ${iidxIdent}]`;
    default:
      throw new Error(`heapEffect: no inline instance loader for type '${wgslType}'`);
  }
}

/**
 * Inline WGSL expression that loads a uniform of `wgslType` given an
 * already-bound u32 byte ref `refIdent`. The arena's data region
 * starts 16 bytes after the ref; we read via the typed view that
 * matches the size (vec4<f32> for mat4/vec4, f32 for vec3/f32, u32
 * for integer types).
 */
function uniformLoadExpr(wgslType: string, refIdent: string): string {
  switch (wgslType) {
    case "mat4x4<f32>": {
      const b = `((${refIdent} + 16u) / 16u)`;
      return `mat4x4<f32>(heapV4f[${b}], heapV4f[${b} + 1u], heapV4f[${b} + 2u], heapV4f[${b} + 3u])`;
    }
    case "vec4<f32>": return `heapV4f[(${refIdent} + 16u) / 16u]`;
    case "vec3<f32>": {
      const b = `((${refIdent} + 16u) / 4u)`;
      return `vec3<f32>(heapF32[${b}], heapF32[${b} + 1u], heapF32[${b} + 2u])`;
    }
    case "vec2<f32>": {
      const b = `((${refIdent} + 16u) / 4u)`;
      return `vec2<f32>(heapF32[${b}], heapF32[${b} + 1u])`;
    }
    case "f32":  return `heapF32[(${refIdent} + 16u) / 4u]`;
    case "u32":  return `heapU32[(${refIdent} + 16u) / 4u]`;
    default:
      throw new Error(`heapEffect: no inline loader for uniform type '${wgslType}'`);
  }
}

/**
 * Inline WGSL expression for a per-vertex attribute read.
 *
 * Cyclic addressing: every attribute is read at `vid % count` where
 * `count` is the alloc header's `length` field at byte offset 4
 * (= `heapU32[refIdent/4u + 1u]`). For regular per-vertex attribs
 * `count == vertexCount > vid`, so the modulo is a no-op behaviorally.
 * For single-value broadcasts (`BufferView.ofValue`), `count == 1`,
 * so every vertex reads element 0. No special-case path in the
 * runtime — the placement just stores 1 element and lets the shader
 * cycle.
 *
 * Cost: one extra `heapU32` load + one modulo per attribute per
 * vertex. Negligible vs the bandwidth saving on broadcasts (1 elt
 * upload instead of N) and the simplicity (no two-path runtime).
 */
function attributeLoadExpr(wgslType: string, refIdent: string, vidIdent: string): string {
  // `count` = alloc-header length at offset 4 (u32 view index = ref/4 + 1).
  const count = `heapU32[${refIdent} / 4u + 1u]`;
  const idx = `(${vidIdent} % ${count})`;
  switch (wgslType) {
    case "vec3<f32>": {
      const b = `((${refIdent} + 16u) / 4u + ${idx} * 3u)`;
      return `vec3<f32>(heapF32[${b}], heapF32[${b} + 1u], heapF32[${b} + 2u])`;
    }
    case "vec4<f32>": {
      // Source can be V3-tight (12 B/elt → 3 floats, .w = 1.0 implicit)
      // or V4-tight (16 B/elt → 4 floats, .w from data). Header offset 8
      // carries `stride_bytes`; we pick the load path branchless via
      // `select`. The 4th f32 read is OOB-safe under WGSL's zero-init
      // OOB semantics when stride == 12.
      const strideF = `(heapU32[${refIdent} / 4u + 2u] / 4u)`;
      const b = `((${refIdent} + 16u) / 4u + ${idx} * ${strideF})`;
      const w = `select(1.0, heapF32[${b} + 3u], ${strideF} == 4u)`;
      return `vec4<f32>(heapF32[${b}], heapF32[${b} + 1u], heapF32[${b} + 2u], ${w})`;
    }
    case "vec2<f32>": {
      const b = `((${refIdent} + 16u) / 4u + ${idx} * 2u)`;
      return `vec2<f32>(heapF32[${b}], heapF32[${b} + 1u])`;
    }
    default:
      throw new Error(`heapEffect: no inline attribute loader for type '${wgslType}'`);
  }
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
  megacall: boolean,
  isAtlasBucket: boolean,
  atlasArraySize: number,
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

  const megacallDecls = megacall
    ? `\n@group(0) @binding(4) var<storage, read> drawTable:       array<u32>;\n@group(0) @binding(5) var<storage, read> indexStorage:    array<u32>;\n@group(0) @binding(6) var<storage, read> firstDrawInTile: array<u32>;`
    : "";

  const atlasBlock = isAtlasBucket ? generateAtlasPrelude(atlasArraySize) : "";

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
 * Atlas prelude: declares the per-format `binding_array<texture_2d<f32>, N>`s
 * + the shared atlas sampler at fixed bindings 11/12/13, then the
 * `atlasSample` software-mip helper that the FS rewriter calls in place
 * of `textureSample(<atlasName>, smp, uv)`.
 *
 * Mip filtering is software (the GPU's mip walk would walk the wrong
 * pyramid — our embedded 1.5×1 layout isn't visible to the hardware
 * mip filter). LOD is computed from screen-space derivatives in mip-0
 * atlas-pixel space; the constant 4096 matches `ATLAS_PAGE_SIZE`.
 *
 * Wrap-mode mirror math: `1 - |((u - floor(u/2)*2) - 1)|` yields a
 * triangle wave with period 2 and amplitude [0,1] — matches WebGPU's
 * `mirror-repeat` address-mode spec. Verified at u={0,0.5,1,1.5,2,-0.5}.
 */
function generateAtlasPrelude(atlasArraySize: number): string {
  return /* wgsl */`
@group(0) @binding(${ATLAS_BINDING_LINEAR})  var atlasLinear:  binding_array<texture_2d<f32>, ${atlasArraySize}>;
@group(0) @binding(${ATLAS_BINDING_SRGB})    var atlasSrgb:    binding_array<texture_2d<f32>, ${atlasArraySize}>;
@group(0) @binding(${ATLAS_BINDING_SAMPLER}) var atlasSampler: sampler;

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
  let lin = textureSampleLevel(atlasLinear[pageRef], atlasSampler, atlasUv, 0.0);
  let sr  = textureSampleLevel(atlasSrgb[pageRef],   atlasSampler, atlasUv, 0.0);
  return select(lin, sr, format == 1u);
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
  if (numMips <= 1u) {
    return atlasSampleAtMip(pageRef, format, origin, size, 0u, uvW);
  }
  let dx = dpdx(uvW * size);
  let dy = dpdy(uvW * size);
  let rho = max(length(dx), length(dy)) * 4096.0;
  let lod = clamp(log2(max(rho, 1e-6)), 0.0, f32(numMips - 1u));
  let lo = u32(floor(lod));
  let hi = min(lo + 1u, numMips - 1u);
  let t  = lod - f32(lo);
  let a = atlasSampleAtMip(pageRef, format, origin, size, lo, uvW);
  let b = atlasSampleAtMip(pageRef, format, origin, size, hi, uvW);
  return mix(a, b, t);
}
`;
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

// ─── Stage rewriting (layout-driven) ────────────────────────────────

/**
 * FS uniform reads are supported by passing the resolved per-uniform
 * `u32` ref (the arena byte offset) as a flat-interpolated varying.
 * VS computes the ref in its preamble (already does this for its own
 * reads), writes it into `VsOut`; FS reads the varying and goes
 * straight to the arena — no `headersU32` indirection per fragment.
 *
 * Cost: one varying slot per FS-used uniform (a u32 per slot, not a
 * vec4). Wins one load per uniform per fragment, which in practice
 * dominates over the interpolator cost.
 *
 * Per-instance uniforms in FS are not auto-threaded: `iidx` varies
 * within a primitive, so a flat-interpolated ref doesn't address the
 * right element. Throw with explicit guidance.
 */
export function rewriteForLayout(
  rawVs: string,
  rawFs: string,
  layout: BucketLayout,
): { vs: string; fs: string; preludeWgsl: string } {
  const fsUniformsUsed = new Set<string>();
  if (rawFs) {
    for (const m of rawFs.matchAll(/_w_uniform\.(\w+)/g)) fsUniformsUsed.add(m[1]!);
  }
  const fsAtlasUsed = scanFsAtlasTextures(rawFs, layout.atlasTextureBindings);
  // FS reads of per-instance uniforms are auto-threaded too: VS stashes
  // `iidx` as a flat-interpolated `_iidx: u32` varying; FS preamble
  // pulls `let iidx = in._iidx;` and feeds it into `instanceLoadExpr`
  // alongside the same flat-interpolated `_h_FooRef`. All vertices of
  // a primitive belong to one instance, so flat interpolation gives
  // the right iidx for the primitive's fragments.
  const fsNeedsIidx = layout.isInstanced &&
    [...fsUniformsUsed].some(n => layout.perInstanceUniforms.has(n));
  const augmented = augmentPreludeForFsUniforms(layout, fsUniformsUsed, fsNeedsIidx, fsAtlasUsed);
  return {
    vs: rawVs ? rewriteVertex(rawVs, layout, fsUniformsUsed, fsNeedsIidx, fsAtlasUsed) : "",
    fs: rawFs ? rewriteFragment(rawFs, layout, fsUniformsUsed, fsNeedsIidx, fsAtlasUsed) : "",
    preludeWgsl: augmented,
  };
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
 * Find atlas-routed texture binding names actually called via
 * `textureSample(<name>, smp, uv)` in the FS source. Returns the subset
 * of `atlasTextureBindings` that the FS body actually references —
 * non-referenced atlas bindings don't need their drawHeader fields
 * threaded through VsOut.
 */
function scanFsAtlasTextures(fs: string, atlasNames: ReadonlySet<string>): Set<string> {
  const used = new Set<string>();
  if (atlasNames.size === 0 || fs.length === 0) return used;
  for (const name of atlasNames) {
    const re = new RegExp(`textureSample\\s*\\(\\s*${escapeRegExp(name)}\\b`);
    if (re.test(fs)) used.add(name);
  }
  return used;
}

/**
 * Substitute `textureSample(<atlasName>, <smp>, <uv>)` with
 * `atlasSample(_h_<name>PageRef, _h_<name>FormatBits,
 *              _h_<name>Origin, _h_<name>Size, <uv>)`. The atlas-routed
 * `<smp>` argument is dropped — the shared `atlasSampler` is used inside
 * `atlasSampleAtMip`. The user's per-draw sampler state lives in
 * `formatBits` and is consumed by `atlasSample`.
 *
 * Naive paren splitting on commas is fine for the v1 surface (uv args
 * are typically simple identifiers / .xy swizzles); future work: a
 * proper paren-balanced argument splitter if user shaders embed nested
 * parens in the uv expression.
 */
export function rewriteAtlasTextureSamples(
  src: string,
  atlasNames: ReadonlySet<string>,
): string {
  if (atlasNames.size === 0 || src.length === 0) return src;
  let out = src;
  for (const name of atlasNames) {
    const v = atlasVaryingNames(name);
    const re = new RegExp(`textureSample\\s*\\(\\s*${escapeRegExp(name)}\\s*,\\s*([^,]+?)\\s*,\\s*([^)]+?)\\s*\\)`, "g");
    out = out.replace(re, (_m, _smp: string, uv: string) =>
      `atlasSample(${v.pageRef}, ${v.formatBits}, ${v.origin}, ${v.size}, ${uv})`,
    );
  }
  return out;
}

/**
 * Build a prelude that includes the base bindings + base VsOut
 * fields, plus one `@interpolate(flat) _h_<name>Ref: u32` slot per
 * FS-used uniform. The base prelude already exists on the layout —
 * we splice the extra fields into the VsOut struct.
 */
function augmentPreludeForFsUniforms(
  layout: BucketLayout,
  fsUniformsUsed: ReadonlySet<string>,
  fsNeedsIidx: boolean,
  fsAtlasUsed: ReadonlySet<string>,
): string {
  if (fsUniformsUsed.size === 0 && !fsNeedsIidx) {
    return layout.preludeWgsl;
  }
  // Pick locations starting after the highest existing varying location.
  const locs: number[] = [];
  for (const m of layout.preludeWgsl.matchAll(/@location\((\d+)\)/g)) {
    locs.push(parseInt(m[1]!, 10));
  }
  let next = locs.length === 0 ? 0 : Math.max(...locs) + 1;
  const extraFields: string[] = [];
  for (const name of fsUniformsUsed) {
    extraFields.push(`  @location(${next++}) @interpolate(flat) _h_${name}Ref: u32,`);
  }
  if (fsNeedsIidx) {
    extraFields.push(`  @location(${next++}) @interpolate(flat) _iidx: u32,`);
  }
  // Atlas varyings live on the schema (added by extendSchemaWithAtlasVaryings
  // when atlasTextureBindings is non-empty), so they're already in VsOut —
  // no extra slots needed here.
  void fsAtlasUsed;
  return layout.preludeWgsl.replace(
    /(struct\s+VsOut\s*\{[\s\S]*?)(\n\};)/,
    (_m, body: string, close: string) => `${body}\n${extraFields.join("\n")}${close}`,
  );
}

function rewriteVertex(
  wgsl: string,
  layout: BucketLayout,
  fsUniformsUsed: ReadonlySet<string>,
  fsNeedsIidx: boolean,
  fsAtlasUsed: ReadonlySet<string>,
): string {
  let s = wgsl;

  // 1. Drop DSL-emitted UBO struct + binding.
  s = s.replace(/struct\s+_UB_uniform\s*\{[\s\S]*?\};?\s*/g, "");
  s = s.replace(/@group\(\d+\)\s*@binding\(\d+\)\s*var<uniform>\s+_w_uniform\s*:\s*_UB_uniform\s*;?\s*/g, "");

  // 1b. Drop DSL-emitted texture/sampler binding decls — the prelude
  //     re-emits them at our chosen group(0)/binding(N) numbers. The
  //     identifier names survive (we don't rename), so references in
  //     the body still resolve to the prelude-bound variables.
  s = stripTextureSamplerDecls(s);

  // 2. Find @vertex signature.
  const match = s.match(/@vertex\s+fn\s+(\w+)\s*\(\s*(\w+)\s*:\s*(\w+)\s*\)\s*->\s*(\w+)\s*\{/);
  if (match === null) return "";
  const fnHeader     = match[0];
  const paramName    = match[2]!;
  const inputStruct  = match[3]!;
  const outputStruct = match[4]!;

  // 3. Capture body for the preamble pass — uses the ORIGINAL header
  //    so the regex still matches.
  const fnBodyMatch = s.match(/@vertex\s+fn\s+\w+\s*\([^)]*\)\s*->\s*\w+\s*\{([\s\S]*)\}\s*$/);
  if (fnBodyMatch === null) {
    throw new Error("heapEffect: cannot locate VS function body");
  }
  const body = fnBodyMatch[1]!;
  const bindings = buildVsPreamble(body, layout, paramName, fsUniformsUsed, fsNeedsIidx, fsAtlasUsed);

  // 4. Rewrite header BEFORE we rename the output struct everywhere
  //    (otherwise `fnHeader` no longer matches anything in `s`).
  //
  //    Instanced buckets bind `instance_index` to `iidx` and force
  //    `drawIdx = 0u` (the bucket holds a single slot). Non-instanced
  //    keeps the `drawIdx = instance_index` (firstInstance trick) and
  //    leaves `iidx` undefined — the preamble only emits per-instance
  //    reads when the bucket has any.
  if (layout.megacall) {
    s = s.replace(
      fnHeader,
      `@vertex\nfn vs(@builtin(vertex_index) emitIdx: u32) -> VsOut {\n${megacallSearchPrelude()}${bindings.preamble}`,
    );
  } else {
    const idxParam = layout.isInstanced ? "iidx" : "drawIdx";
    const indexLet = layout.isInstanced ? "  let drawIdx = 0u;\n" : "";
    s = s.replace(
      fnHeader,
      `@vertex\nfn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) ${idxParam}: u32) -> VsOut {\n${indexLet}${bindings.preamble}`,
    );
  }

  // 5. Drop input + output structs (we use the prelude's per-effect
  //    VsOut). Rename remaining references inside the body.
  s = s.replace(new RegExp(`struct\\s+${escapeRegExp(inputStruct)}\\s*\\{[\\s\\S]*?\\};?`), "");
  s = s.replace(new RegExp(`struct\\s+${escapeRegExp(outputStruct)}\\s*\\{[\\s\\S]*?\\};?`), "");
  s = s.replace(new RegExp(`\\b${escapeRegExp(outputStruct)}\\b`, "g"), "VsOut");

  // 5b. Inject `out._h_FooRef = _h_FooRef;` lines for every FS-used
  //     uniform, right after the body's `var out: VsOut;` declaration.
  //     These flat-interpolated u32 refs are what the FS preamble
  //     reads to skip the headers indirection per fragment.
  if (bindings.varyingWrites.length > 0) {
    s = s.replace(
      /var\s+out\s*:\s*VsOut\s*;/,
      (m) => `${m}\n${bindings.varyingWrites}`,
    );
  }

  // 6. Substitute uniform / attribute references with the let
  //    identifiers. Order matters: paramName.X must be substituted
  //    before any other paramName cleanup.
  s = s.replace(/_w_uniform\.(\w+)/g, (_m, name: string) => {
    const r = bindings.uniformReplace.get(name);
    if (r === undefined) {
      throw new Error(`heapEffect: VS references uniform '${name}' which is not in the effect schema`);
    }
    return r;
  });
  s = s.replace(new RegExp(`\\b${escapeRegExp(paramName)}\\.(\\w+)`, "g"), (_m, attr: string) => {
    const r = bindings.attrReplace.get(attr);
    if (r === undefined) {
      throw new Error(`heapEffect: VS references attribute '${attr}' which is not in the effect schema`);
    }
    return r;
  });

  // 7. Tidy.
  s = s.replace(/\n{3,}/g, "\n\n").trimStart();
  return s;
}

/**
 * Build the let-binding preamble for a VS body. We need the schema's
 * attribute types here too (to pick the right `attributeLoadExpr`),
 * which the layout doesn't carry — so this helper takes the layout
 * and the schema implicitly via the rewriter's caller.
 */
function buildVsPreamble(
  body: string, layout: BucketLayout, paramName: string,
  fsUniformsUsed: ReadonlySet<string>,
  fsNeedsIidx: boolean,
  fsAtlasUsed: ReadonlySet<string>,
): {
  preamble: string;
  uniformReplace: Map<string, string>;
  attrReplace: Map<string, string>;
  /** `out._h_FooRef = _h_FooRef;` lines to inject into the VS body. */
  varyingWrites: string;
} {
  const uniformsUsed = new Set<string>();
  for (const m of body.matchAll(/_w_uniform\.(\w+)/g)) uniformsUsed.add(m[1]!);
  const attrsUsed = new Set<string>();
  for (const m of body.matchAll(new RegExp(`\\b${escapeRegExp(paramName)}\\.(\\w+)`, "g"))) {
    attrsUsed.add(m[1]!);
  }
  // Refs we must materialize: union of VS-body uses + FS-relayed uses.
  // VS-body uses additionally need the value-let; FS-relayed only need
  // the ref-let so the VS can stash it into VsOut.
  const refNeeded = new Set<string>([...uniformsUsed, ...fsUniformsUsed]);

  const lines: string[] = [];
  const writes: string[] = [];
  const uniformReplace = new Map<string, string>();
  const attrReplace = new Map<string, string>();
  const stride = layout.strideU32;

  for (const f of layout.drawHeaderFields) {
    const ident = `_h_${f.name}`;
    const refIdent = `${ident}Ref`;
    const refOffU32 = f.byteOffset / 4;
    if (f.kind === "uniform-ref" && refNeeded.has(f.name)) {
      lines.push(`  let ${refIdent} = headersU32[drawIdx * ${stride}u + ${refOffU32}u];`);
      if (uniformsUsed.has(f.name)) {
        const loader = layout.perInstanceUniforms.has(f.name)
          ? instanceLoadExpr(f.uniformWgslType ?? "", refIdent, "iidx")
          : uniformLoadExpr(f.uniformWgslType ?? "", refIdent);
        lines.push(`  let ${ident} = ${loader};`);
        uniformReplace.set(f.name, ident);
      }
      if (fsUniformsUsed.has(f.name)) {
        writes.push(`  out.${refIdent} = ${refIdent};`);
      }
    } else if (f.kind === "attribute-ref" && attrsUsed.has(f.name)) {
      lines.push(`  let ${refIdent} = headersU32[drawIdx * ${stride}u + ${refOffU32}u];`);
      lines.push(`  let ${ident} = ${attributeLoadExpr(f.attributeWgslType ?? "vec3<f32>", refIdent, "vid")};`);
      attrReplace.set(f.name, ident);
    } else if (
      f.kind === "texture-ref"
      && f.textureBindingName !== undefined
      && fsAtlasUsed.has(f.textureBindingName)
    ) {
      // Atlas texture-ref: load the inline u32 / vec2<f32> drawHeader
      // values and forward them to FS via flat-interpolated VsOut slots.
      const v = atlasVaryingNames(f.textureBindingName);
      switch (f.textureSub) {
        case "pageRef": {
          lines.push(`  let ${v.pageRef} = headersU32[drawIdx * ${stride}u + ${refOffU32}u];`);
          writes.push(`  out.${v.pageRef} = ${v.pageRef};`);
          break;
        }
        case "formatBits": {
          lines.push(`  let ${v.formatBits} = headersU32[drawIdx * ${stride}u + ${refOffU32}u];`);
          writes.push(`  out.${v.formatBits} = ${v.formatBits};`);
          break;
        }
        case "origin": {
          // origin is vec2<f32> packed at byteOffset; read via heapF32-of-headers
          // is unavailable here (the headers buffer is `array<u32>`), so cast
          // u32→f32 via bitcast.
          lines.push(`  let ${v.origin} = vec2<f32>(bitcast<f32>(headersU32[drawIdx * ${stride}u + ${refOffU32}u]), bitcast<f32>(headersU32[drawIdx * ${stride}u + ${refOffU32}u + 1u]));`);
          writes.push(`  out.${v.origin} = ${v.origin};`);
          break;
        }
        case "size": {
          lines.push(`  let ${v.size} = vec2<f32>(bitcast<f32>(headersU32[drawIdx * ${stride}u + ${refOffU32}u]), bitcast<f32>(headersU32[drawIdx * ${stride}u + ${refOffU32}u + 1u]));`);
          writes.push(`  out.${v.size} = ${v.size};`);
          break;
        }
      }
    }
  }
  if (fsNeedsIidx) writes.push(`  out._iidx = iidx;`);
  return {
    preamble: lines.join("\n"),
    uniformReplace, attrReplace,
    varyingWrites: writes.join("\n"),
  };
}

/**
 * Build the FS uniform-load preamble. For each FS-used uniform, pull
 * the resolved ref from `in._h_FooRef` (the flat-interpolated varying
 * the VS stashed) and load the value via `uniformLoadExpr` — the same
 * expression the VS uses, so identical pixel output regardless of
 * which stage owns the read.
 */
function buildFsPreamble(
  layout: BucketLayout,
  fsUniformsUsed: ReadonlySet<string>,
  fsNeedsIidx: boolean,
  fsAtlasUsed: ReadonlySet<string>,
): { preamble: string; replace: Map<string, string> } {
  if (fsUniformsUsed.size === 0 && fsAtlasUsed.size === 0) {
    return { preamble: "", replace: new Map() };
  }
  const lines: string[] = [];
  const replace = new Map<string, string>();
  if (fsNeedsIidx) lines.push(`  let iidx = in._iidx;`);
  for (const name of fsAtlasUsed) {
    const v = atlasVaryingNames(name);
    lines.push(`  let ${v.pageRef} = in.${v.pageRef};`);
    lines.push(`  let ${v.formatBits} = in.${v.formatBits};`);
    lines.push(`  let ${v.origin} = in.${v.origin};`);
    lines.push(`  let ${v.size} = in.${v.size};`);
  }
  for (const f of layout.drawHeaderFields) {
    if (f.kind !== "uniform-ref") continue;
    if (!fsUniformsUsed.has(f.name)) continue;
    const ident = `_h_${f.name}`;
    const refIdent = `${ident}Ref`;
    lines.push(`  let ${refIdent} = in.${refIdent};`);
    const loader = layout.perInstanceUniforms.has(f.name)
      ? instanceLoadExpr(f.uniformWgslType ?? "", refIdent, "iidx")
      : uniformLoadExpr(f.uniformWgslType ?? "", refIdent);
    lines.push(`  let ${ident} = ${loader};`);
    replace.set(f.name, ident);
  }
  return { preamble: lines.join("\n"), replace };
}

function rewriteFragment(
  wgsl: string,
  layout: BucketLayout,
  fsUniformsUsed: ReadonlySet<string>,
  fsNeedsIidx: boolean,
  fsAtlasUsed: ReadonlySet<string>,
): string {
  let s = wgsl;

  // 1. Drop DSL-emitted UBO struct + binding. FS uniform reads are
  //    auto-threaded via VsOut: the VS preamble computed each ref,
  //    stashed it as a flat-interpolated u32 varying; here we read
  //    `in._h_FooRef` and emit the same `uniformLoadExpr` the VS uses.
  s = s.replace(/struct\s+_UB_uniform\s*\{[\s\S]*?\};?\s*/g, "");
  s = s.replace(/@group\(\d+\)\s*@binding\(\d+\)\s*var<uniform>\s+_w_uniform\s*:\s*_UB_uniform\s*;?\s*/g, "");
  // Drop DSL-emitted texture/sampler binding decls — see rewriteVertex
  // step 1b for rationale.
  s = stripTextureSamplerDecls(s);

  // 2. Find @fragment signature.
  const match = s.match(/@fragment\s+fn\s+(\w+)\s*\(\s*(\w+)\s*:\s*(\w+)\s*\)\s*->\s*(\w+)\s*\{/);
  if (match === null) return "";
  const fnHeader      = match[0];
  const paramName     = match[2]!;
  const inputStruct   = match[3]!;
  const outputStruct  = match[4]!;

  // 3. Drop input + output structs (VsOut comes from the prelude).
  s = s.replace(new RegExp(`struct\\s+${escapeRegExp(inputStruct)}\\s*\\{[\\s\\S]*?\\};?`), "");
  const outputStructRe = new RegExp(`struct\\s+${escapeRegExp(outputStruct)}\\s*\\{([\\s\\S]*?)\\};?`);
  const outputMatch = s.match(outputStructRe);
  if (outputMatch === null) {
    throw new Error(`heapEffect: cannot locate FS output struct '${outputStruct}'`);
  }
  const outBody = outputMatch[1]!;
  const fieldRe = /@location\(0\)\s+(\w+)\s*:\s*vec4<f32>\s*,?/g;
  const fieldMatches = [...outBody.matchAll(fieldRe)];
  if (fieldMatches.length !== 1) {
    throw new Error(`heapEffect: FS must have a single @location(0) vec4<f32> output; got ${fieldMatches.length}`);
  }
  const outputFieldName = fieldMatches[0]![1]!;
  s = s.replace(outputMatch[0], "");

  // 4. Rewrite header. Inject FS uniform preamble: read each `in._h_FooRef`
  //    varying, then load the value via the same expression the VS would
  //    use. Substitution of `_w_uniform.X` references happens in step 6.
  const fsPreamble = buildFsPreamble(layout, fsUniformsUsed, fsNeedsIidx, fsAtlasUsed);
  s = s.replace(
    fnHeader,
    `@fragment\nfn fs(in: VsOut) -> @location(0) vec4<f32> {\n${fsPreamble.preamble}`,
  );
  if (paramName !== "in") {
    s = s.replace(new RegExp(`\\b${escapeRegExp(paramName)}\\.`, "g"), "in.");
  }
  // Substitute FS uniform references with the preamble's let-idents.
  s = s.replace(/_w_uniform\.(\w+)/g, (_m, name: string) => {
    const r = fsPreamble.replace.get(name);
    if (r === undefined) {
      throw new Error(`heapEffect: FS references uniform '${name}' which is not in the effect schema`);
    }
    return r;
  });

  // Rewrite atlas-routed textureSample calls. Must run after the FS
  // preamble injection (which defines the `_h_<name>*` lets we reference)
  // and before the var-out collapse so the rewrite catches calls that
  // appear inside the body. The user's per-draw sampler is dropped —
  // the helper uses the shared atlasSampler internally.
  s = rewriteAtlasTextureSamples(s, fsAtlasUsed);

  // 5. Collapse var-out / out.X = expr / return out → return expr.
  const fieldRef = `out\\.${escapeRegExp(outputFieldName)}`;
  const assignRe = new RegExp(
    `var\\s+out\\s*:\\s*${escapeRegExp(outputStruct)}\\s*;([\\s\\S]*?)${fieldRef}\\s*=\\s*([\\s\\S]*?);([\\s\\S]*?)return\\s+out\\s*;`,
  );
  const assignMatch = s.match(assignRe);
  if (assignMatch === null) {
    throw new Error(`heapEffect: FS body doesn't follow the var-out / out.${outputFieldName} = … / return out pattern`);
  }
  const beforeAssign = assignMatch[1]!;
  const expr         = assignMatch[2]!;
  const afterAssign  = assignMatch[3]!;
  s = s.replace(assignMatch[0], beforeAssign + afterAssign + `return ${expr};`);

  // 6. Tidy.
  s = s.replace(/\n{3,}/g, "\n\n").trimStart();
  return s;
}

/**
 * Megacall VS prelude: binary-search `drawTable` (one u32-quad per slot:
 * firstEmit, drawIdx, indexStart, indexCount) for the slot owning the
 * given `emitIdx`, then vertex-pull the index from `indexStorage`.
 * Defines `drawIdx` + `vid` so the rest of the existing preamble works
 * unchanged.
 */
export function megacallSearchPrelude(): string {
  return `  let _tileIdx = emitIdx >> 6u;\n  var lo: u32 = firstDrawInTile[_tileIdx];\n  var hi: u32 = firstDrawInTile[_tileIdx + 1u];\n  loop {\n    if (lo >= hi) { break; }\n    let _mid = (lo + hi + 1u) >> 1u;\n    if (drawTable[_mid * 4u] <= emitIdx) { lo = _mid; } else { hi = _mid - 1u; }\n  }\n  let _slot = lo;\n  let drawIdx = drawTable[_slot * 4u + 1u];\n  let _indexStart = drawTable[_slot * 4u + 2u];\n  let vid = indexStorage[_indexStart + (emitIdx - drawTable[_slot * 4u])];\n`;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Strip DSL-emitted texture/sampler binding declarations from raw
 * stage WGSL. The prelude re-emits them at our chosen group/binding
 * numbers; identifiers survive so body references still resolve.
 *
 * Matches:
 *   `@group(N) @binding(M) var <name> : texture_*<...>;`
 *   `@group(N) @binding(M) var <name> : sampler[_comparison];`
 *
 * Permissive on whitespace; conservative on the type expression
 * (only the texture_/sampler families) so we don't accidentally
 * eat unrelated bindings.
 */
export function stripTextureSamplerDecls(wgsl: string): string {
  const re = /@group\(\d+\)\s*@binding\(\d+\)\s*var\s+\w+\s*:\s*(?:texture_\w+(?:<[^>]*>)?|sampler(?:_comparison)?)\s*;\s*/g;
  return wgsl.replace(re, "");
}
