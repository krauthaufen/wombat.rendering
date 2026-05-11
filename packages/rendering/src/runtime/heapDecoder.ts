// heapDecoder — synthesise a vertex stage that loads every input/uniform
// the user effect reads from the heap and surfaces them on the inter-
// stage carrier.
//
// Composition model:
//
//   effect(decoderEffect, ...userEffect.stages)
//
// `composeStages` fuses the decoder VS with the user VS into a single
// merged entry. The decoder writes `Position`, `ModelTrafo`, etc.; the
// user VS reads those names from `Input` scope (after `uniformsToInputs`
// renames its `Uniform`-scope reads to `Input`). Stage fusion's
// state-struct path threads the values through helpers naturally — no
// in-place IR rewriting, no `extractFusedEntry` wrapper-init gap, no
// post-emit string mangling.
//
// The decoder is an IR Module: storage-buffer bindings for heap arena +
// megacall lookup buffers, plus a single vertex Entry. Body shape:
//
//   @vertex fn heap_decode(@builtin(vertex_index) emitIdx: u32) -> Out {
//     // megacall binary search → heap_drawIdx / instId / vid
//     ...
//     out.<name> = loadAttributeByRef(...);   // attribute-ref fields
//     out.<name> = loadUniformByRef(...);     // uniform-ref fields
//     return out;
//   }
//
// gl_Position is NOT written here — the user VS owns it, and the fused
// entry's last writer wins (linkCrossStage semantics).
//
// Texture-ref drawHeader fields ARE surfaced through the carrier as
// flat-interpolated varyings — four per atlas-routed texture binding
// (`<name>__atlasPageRef: u32`, `<name>__atlasFormatBits: u32`,
// `<name>__atlasOrigin: vec2<f32>`, `<name>__atlasSize: vec2<f32>`).
// The matching FS-side rewrite turns each `textureSample(t, smp, uv)`
// into `atlasSample(<t>__atlasPageRef, …, uv)` reading from the
// carrier — see `rewriteFsAtlasTexturesViaCarrier` below.

import {
  stage as makeStage,
  type Effect,
} from "@aardworx/wombat.shader";
import {
  type Module, type Expr, type Stmt, type Type, type ValueDef,
  type EntryDef, type EntryParameter, type Var,
  Tu32, Tf32,
} from "@aardworx/wombat.shader/ir";
import type { BucketLayout, DrawHeaderField } from "./heapEffect.js";
import {
  Tvec2, Tvec3, Tvec4, Tmat4, TarrU32,
  constU32, add, mul, div, item, newVec, readScope,
  loadHeaderRef, loadUniformByRef, loadAttributeByRef, loadInstanceByRef,
  buildMegacallPrelude,
  heapArenaStorageDecls, megacallLookupStorageDecls,
  headersU32,
} from "./heapIrBuilders.js";

// Build an Expr that returns the u32 value of a specific inline u32
// slot in the drawHeader: `headersU32[drawIdx * stride + offset_u32]`.
function readHeaderU32(drawIdx: Expr, byteOffset: number, strideU32: number): Expr {
  if ((byteOffset % 4) !== 0) {
    throw new Error(`readHeaderU32: byte offset ${byteOffset} not 4-aligned`);
  }
  const fieldU32 = byteOffset / 4;
  const base = mul(drawIdx, constU32(strideU32), Tu32);
  const idx = fieldU32 === 0 ? base : add(base, constU32(fieldU32), Tu32);
  return item(headersU32, idx, Tu32);
}

// Read two contiguous u32 slots, reinterpret as f32, return as vec2<f32>.
// Used for the inline `origin: vec2<f32>` and `size: vec2<f32>` atlas
// drawHeader entries — the runtime packs them via Float32Array view
// over the same backing buffer, so on the GPU side the bits at u32
// offset are valid f32 values; we just need to address them and
// bitcast.
function readHeaderVec2F(drawIdx: Expr, byteOffset: number, strideU32: number): Expr {
  if ((byteOffset % 4) !== 0) {
    throw new Error(`readHeaderVec2F: byte offset ${byteOffset} not 4-aligned`);
  }
  const fieldU32 = byteOffset / 4;
  const base = mul(drawIdx, constU32(strideU32), Tu32);
  const idx0 = fieldU32 === 0 ? base : add(base, constU32(fieldU32), Tu32);
  const idx1 = add(base, constU32(fieldU32 + 1), Tu32);
  // headersU32 is u32; we need f32. Use the same trick the runtime does:
  // expose a parallel `headersF32` view over the same buffer? That
  // doesn't exist yet — for now bitcast via a CallIntrinsic on `bitcast<f32>`.
  const bitcast = (e: Expr): Expr => ({
    kind: "CallIntrinsic",
    op: {
      name: "bitcast<f32>",
      returnTypeOf: () => Tf32,
      pure: true,
      emit: { wgsl: "bitcast<f32>", glsl: "uintBitsToFloat" },
    },
    args: [e],
    type: Tf32,
  } as Expr);
  return newVec(
    [bitcast(item(headersU32, idx0, Tu32)), bitcast(item(headersU32, idx1, Tu32))],
    Tvec2,
  );
}

/**
 * Decoder-side carrier name for an atlas texture sub-field. Mirrors the
 * existing `_h_<name><Sub>` naming used by `atlasVaryingNames` in
 * heapEffect.ts so the FS-side `textureSample → atlasSample` rewrite
 * can keep reading them by the same names.
 */
export function atlasCarrierName(textureBindingName: string, sub: "pageRef" | "formatBits" | "origin" | "size"): string {
  const cap = sub.charAt(0).toUpperCase() + sub.slice(1);
  return `_h_${textureBindingName}${cap}`;
}

// ─── wgsl-type-string → IR Type ──────────────────────────────────────

/**
 * Convert the WGSL type-string carried on `DrawHeaderField.{uniformWgslType,
 * attributeWgslType}` back into an IR `Type`. The string set is closed:
 * it's whatever `irTypeToWgsl` in heapEffect.ts emits when populating
 * the bucket schema, so the reverse map is finite and exact.
 */
function wgslTypeToIrType(wgsl: string): Type {
  switch (wgsl) {
    case "f32":         return Tf32;
    case "u32":         return Tu32;
    case "i32":         return { kind: "Int", signed: true, width: 32 } as Type;
    case "vec2<f32>":   return Tvec2;
    case "vec3<f32>":   return Tvec3;
    case "vec4<f32>":   return Tvec4;
    case "mat4x4<f32>": return Tmat4;
    default:
      throw new Error(`heapDecoder: cannot translate WGSL type '${wgsl}' to IR Type`);
  }
}

// ─── Decoder synthesis ───────────────────────────────────────────────

/**
 * Decoder emission shape.
 *
 *   - `"standalone"`: the decoder owns the megacall binary search.
 *     `@builtin(vertex_index)` is its sole entry input, and the body's
 *     prelude computes `heap_drawIdx` / `instId` / `vid` as local
 *     `let`s. Used by the canonical heap path.
 *
 *   - `"family-member"`: the family wrapper VS already ran the
 *     megacall search and dispatches into per-effect helpers passing
 *     `heap_drawIdx`, `instId`, `vid` as plain `u32` args. The decoder
 *     consumes those as `EntryParameter`s (non-builtin) and skips the
 *     prelude entirely. After `extractFusedEntry` collapses the
 *     decoder + user stages, the per-effect helper exposed to the
 *     wrapper has exactly three `u32` parameters and one output struct
 *     return.
 */
export type HeapDecoderMode = "standalone" | "family-member";

/**
 * Build a fresh wombat.shader Effect containing exactly one vertex
 * stage — the heap-decoder. Compose it with the user effect's stages
 * via `effect(decoder, ...userEffect.stages)`; let composeStages do
 * the rest.
 *
 * Required input: the bucket layout. Reading the user's iface to
 * derive it happens upstream (heapScene already builds the layout from
 * `effect.compile(...).interface`).
 *
 * Each `attribute-ref` field in the layout produces a per-vertex (or
 * per-instance, depending on `layout.perInstanceAttributes`) load. Each
 * `uniform-ref` field produces a per-draw broadcast load (or per-
 * instance from `layout.perInstanceUniforms`). Atlas texture-ref
 * fields surface as four flat varyings per binding so the FS-side
 * atlas-sample rewrite can read them from the carrier.
 */
export function synthesizeHeapDecoderEffect(layout: BucketLayout, mode: HeapDecoderMode = "standalone"): Effect {
  return makeStage(synthesizeHeapDecoderModule(layout, mode));
}

/**
 * Like `synthesizeHeapDecoderEffect` but returns the raw Module so
 * callers can splice it into an existing module-merge pipeline.
 */
export function synthesizeHeapDecoderModule(layout: BucketLayout, mode: HeapDecoderMode = "standalone"): Module {
  // Storage bindings: arena always required; megacall lookup buffers
  // only when the decoder itself owns the binary search (standalone
  // mode). Family-member mode receives the resolved triple from the
  // wrapper VS via plain u32 fn parameters, so it doesn't need the
  // lookup tables in its own module-scope binding set.
  const storageDecls: ValueDef[] = mode === "standalone"
    ? [...heapArenaStorageDecls(), ...megacallLookupStorageDecls()]
    : [...heapArenaStorageDecls()];

  const inputs: EntryParameter[] = [];
  const outputs: EntryParameter[] = [];
  const stmts: Stmt[] = [];

  // Locals (or input params) that downstream WriteOutput stmts pull
  // from. Populated below depending on `mode`.
  let drawIdxExpr: Expr;
  let instIdExpr:  Expr;
  let vidExpr:     Expr;

  if (mode === "standalone") {
    // Single vertex entry input: the megacall builtin. WGSL emits
    // builtin reads as the *semantic* name; `name === semantic`
    // keeps the @vertex param identifier matching the body.
    inputs.push({
      name: "vertex_index",
      type: Tu32,
      semantic: "vertex_index",
      decorations: [{ kind: "Builtin", value: "vertex_index" }],
    });
    // Read via ReadInput("Builtin", ...): extractFusedEntry detects
    // builtin reads through this scope; a bare `Var` reference would
    // slip past its bodyReads analysis and the fused @vertex
    // wouldn't surface the builtin at all.
    const emitIdxExpr: Expr = { kind: "ReadInput", scope: "Builtin", name: "vertex_index", type: Tu32 };
    // Megacall search → locals heap_drawIdx, instId, vid.
    const { stmts: megacallStmts, locals } = buildMegacallPrelude(emitIdxExpr);
    stmts.push(...megacallStmts);
    drawIdxExpr = { kind: "Var", var: locals.heapDrawIdx, type: Tu32 };
    instIdExpr  = { kind: "Var", var: locals.instId,      type: Tu32 };
    vidExpr     = { kind: "Var", var: locals.vid,         type: Tu32 };
  } else {
    // Family-member mode: receive the megacall outputs as plain u32
    // params from the wrapper. Declare them as EntryParameters
    // (location-decorated so they survive composeStages' attribute
    // classification), then read them via ReadInput("Input", ...)
    // inside the body. After composition the fused entry surfaces
    // them as inputs that the wrapper passes by name.
    const declParam = (name: string, location: number): EntryParameter => ({
      name,
      type: Tu32,
      semantic: name,
      decorations: [
        { kind: "Location", value: location },
        // u32 inter-stage IO must be flat-interpolated.
        { kind: "Interpolation", mode: "flat" },
      ],
    });
    inputs.push(declParam("heap_drawIdx", 0));
    inputs.push(declParam("instId",       1));
    inputs.push(declParam("vid",          2));
    drawIdxExpr = { kind: "ReadInput", scope: "Input", name: "heap_drawIdx", type: Tu32 };
    instIdExpr  = { kind: "ReadInput", scope: "Input", name: "instId",      type: Tu32 };
    vidExpr     = { kind: "ReadInput", scope: "Input", name: "vid",         type: Tu32 };
  }

  // Walk the bucket layout's drawHeader fields. The order doesn't
  // matter for correctness — every surfaced name becomes a unique
  // output and the carrier carries them all.
  let nextLocation = 0;
  // Texture-ref drawHeader fields are grouped by `textureBindingName` —
  // each atlas-routed texture binding produces four contiguous entries
  // (pageRef / formatBits / origin / size). We emit them as four
  // distinct carrier outputs `<name>__atlasPageRef`, etc., so the
  // FS-side `textureSample(t, …)` rewrite can read them by name.
  for (const f of layout.drawHeaderFields) {
    if (f.kind === "uniform-ref") {
      addUniformOutput(f, layout, drawIdxExpr, instIdExpr, outputs, stmts, nextLocation++);
      continue;
    }
    if (f.kind === "attribute-ref") {
      addAttributeOutput(f, layout, drawIdxExpr, instIdExpr, vidExpr, outputs, stmts, nextLocation++);
      continue;
    }
    if (f.kind === "texture-ref") {
      addTextureSubOutput(f, layout, drawIdxExpr, outputs, stmts, nextLocation++);
      continue;
    }
  }

  const entry: EntryDef = {
    name: "heap_decode_vs",
    stage: "vertex",
    inputs,
    outputs,
    arguments: [],
    returnType: { kind: "Void" } as Type,
    body: { kind: "Sequential", body: stmts },
    decorations: [],
  };

  return {
    types: [],
    values: [...storageDecls, { kind: "Entry", entry }],
  };
}

/**
 * Emit a single `WriteOutput(name, loadUniformByRef(...))` for a
 * `uniform-ref` field. Surfaces a per-name output the user VS will
 * read as `ReadInput("Input", name)` post-uniformsToInputs.
 */
function addUniformOutput(
  f: DrawHeaderField,
  layout: BucketLayout,
  drawIdxExpr: Expr,
  instIdExpr: Expr,
  outputs: EntryParameter[],
  stmts: Stmt[],
  location: number,
): void {
  const wgslType = f.uniformWgslType;
  if (wgslType === undefined) {
    throw new Error(`heapDecoder: uniform-ref field '${f.name}' has no uniformWgslType`);
  }
  const irType = wgslTypeToIrType(wgslType);
  const refExpr = loadHeaderRef(drawIdxExpr, f.byteOffset, layout.strideU32);
  const value = layout.perInstanceUniforms.has(f.name)
    ? loadInstanceByRef(refExpr, instIdExpr, wgslType)
    : loadUniformByRef(refExpr, wgslType);

  outputs.push({
    name: f.name,
    type: irType,
    semantic: f.name,
    decorations: [
      { kind: "Location", value: location },
      // Per-draw broadcast — uniform across all vertices of one draw.
      // `flat` interpolation skips the rasterizer's perspective-correct
      // interpolation; required for u32 carriers anyway, harmless for
      // float carriers that are constant across the draw.
      { kind: "Interpolation", mode: "flat" },
    ],
  });
  stmts.push({
    kind: "WriteOutput",
    name: f.name,
    value: { kind: "Expr", value },
  });
}

/**
 * Emit a `WriteOutput` for a single atlas texture-ref sub-field.
 * Atlas drawHeader entries carry the value INLINE (not as a heap ref).
 * Layout (matches `buildBucketLayout` in heapEffect.ts):
 *   pageRef    : u32  — 4 bytes
 *   formatBits : u32  — 4 bytes
 *   (padding to 8-aligned)
 *   origin     : vec2<f32>  — 8 bytes
 *   size       : vec2<f32>  — 8 bytes
 *
 * The carrier output name follows `<textureBindingName>__atlas<Sub>`.
 */
function addTextureSubOutput(
  f: DrawHeaderField,
  layout: BucketLayout,
  drawIdxExpr: Expr,
  outputs: EntryParameter[],
  stmts: Stmt[],
  location: number,
): void {
  const sub = f.textureSub;
  const tex = f.textureBindingName;
  if (sub === undefined || tex === undefined) {
    throw new Error(`heapDecoder: texture-ref field '${f.name}' missing textureSub/textureBindingName`);
  }
  const carrierName = atlasCarrierName(tex, sub);
  let value: Expr;
  let irType: Type;
  switch (sub) {
    case "pageRef":
    case "formatBits": {
      irType = Tu32;
      value = readHeaderU32(drawIdxExpr, f.byteOffset, layout.strideU32);
      break;
    }
    case "origin":
    case "size": {
      irType = Tvec2;
      value = readHeaderVec2F(drawIdxExpr, f.byteOffset, layout.strideU32);
      break;
    }
  }
  outputs.push({
    name: carrierName,
    type: irType,
    semantic: carrierName,
    decorations: [
      { kind: "Location", value: location },
      // Atlas params are constant across all fragments of a primitive
      // — flat interpolation skips the rasterizer's per-pixel math and
      // is mandatory for u32 carriers.
      { kind: "Interpolation", mode: "flat" },
    ],
  });
  stmts.push({
    kind: "WriteOutput",
    name: carrierName,
    value: { kind: "Expr", value },
  });
}

/**
 * Emit a single `WriteOutput(name, loadAttributeByRef(...))` for an
 * `attribute-ref` field. Uses per-vertex `vid` for plain attributes
 * and per-instance `instId` for instance-attributes.
 */
function addAttributeOutput(
  f: DrawHeaderField,
  layout: BucketLayout,
  drawIdxExpr: Expr,
  instIdExpr: Expr,
  vidExpr: Expr,
  outputs: EntryParameter[],
  stmts: Stmt[],
  location: number,
): void {
  const wgslType = f.attributeWgslType;
  if (wgslType === undefined) {
    throw new Error(`heapDecoder: attribute-ref field '${f.name}' has no attributeWgslType`);
  }
  const irType = wgslTypeToIrType(wgslType);
  const refExpr = loadHeaderRef(drawIdxExpr, f.byteOffset, layout.strideU32);
  const idx = layout.perInstanceAttributes.has(f.name) ? instIdExpr : vidExpr;
  const value = loadAttributeByRef(refExpr, idx, wgslType);

  // Per-vertex attributes naturally vary across a primitive — perspective-
  // correct interpolation is the right default. Per-instance attributes
  // are constant across a primitive's vertices but in this fused-stage
  // model they still flow through the carrier, so flat is the safe
  // policy (no semantic difference for genuinely-constant values).
  // Integer-typed inter-stage IO MUST be `flat` (WGSL constraint).
  const isInteger =
    irType.kind === "Int" || irType.kind === "Bool"
    || (irType.kind === "Vector"
        && (irType.element.kind === "Int" || irType.element.kind === "Bool"));
  const interp: "smooth" | "flat" =
    layout.perInstanceAttributes.has(f.name) || isInteger ? "flat" : "smooth";

  outputs.push({
    name: f.name,
    type: irType,
    semantic: f.name,
    decorations: [
      { kind: "Location", value: location },
      { kind: "Interpolation", mode: interp },
    ],
  });
  stmts.push({
    kind: "WriteOutput",
    name: f.name,
    value: { kind: "Expr", value },
  });
}
