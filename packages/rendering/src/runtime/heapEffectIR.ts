// heapEffectIR — IR-level rewriter for the heap render path.
//
// Takes a wombat.shader Effect, transforms its IR Module so that:
//   - Vertex attribute reads are replaced with heap-buffer pulls
//     (cyclic via `vid % length`).
//   - Uniform reads (in any stage) are replaced with heap-buffer
//     decodes via per-draw refs from the headers buffer.
//   - FS uniform decodes thread the per-uniform refs through VsOut
//     as flat-interpolated `_h_<name>Ref: u32` varyings.
//   - The heap arena's four typed views (heapU32, headersU32, heapF32,
//     heapV4f) are added as storage-buffer bindings.
//
// The result is a fresh Module that compiles via the existing
// wombat.shader pipeline (DCE, cross-stage link, fragment-output
// linker, WGSL emit) — no string-regex post-processing.
//
// This module knows nothing about the runtime; the BucketLayout it
// consumes is identical to the one heapScene.ts already builds.

import { compileModule, stage as makeStage, effect as makeEffect } from "@aardworx/wombat.shader";
import type { Effect, CompileOptions } from "@aardworx/wombat.shader";
import { substituteInputsInStage, readInputs, mapExpr, mapStmt, liftReturns, uniformsToInputs } from "@aardworx/wombat.shader/passes";
import { synthesizeHeapDecoderModule } from "./heapDecoder.js";
import {
  type Module, type Expr, type Stmt, type Type, type ValueDef,
  type EntryDef, type EntryParameter, type ParamDecoration,
  Tu32, Tf32, Vec,
} from "@aardworx/wombat.shader/ir";
import {
  megacallSearchPrelude, atlasVaryingNames,
  generateAtlasBindings, generateAtlasSwitch, generateAtlasPrelude,
  HEAP_PERSIST_VERSION, persistKey, lsLoad, lsStore,
  type BucketLayout,
} from "./heapEffect.js";
import type { IntrinsicRef } from "@aardworx/wombat.shader/ir";

// ─── IR builders ────────────────────────────────────────────────────

const Tbool: Type = { kind: "Bool" };

const constU32 = (n: number): Expr => ({
  kind: "Const", type: Tu32, value: { kind: "Int", signed: false, value: n },
});
const constF32 = (n: number): Expr => ({
  kind: "Const", type: Tf32, value: { kind: "Float", value: n },
});
const readScope = (scope: "Uniform" | "Input" | "Builtin", name: string, type: Type): Expr => ({
  kind: "ReadInput", scope, name, type,
});
const item = (target: Expr, index: Expr, elemType: Type): Expr => ({
  kind: "Item", target, index, type: elemType,
});
const add = (lhs: Expr, rhs: Expr, type: Type): Expr => ({ kind: "Add", lhs, rhs, type });
const mul = (lhs: Expr, rhs: Expr, type: Type): Expr => ({ kind: "Mul", lhs, rhs, type });
const div = (lhs: Expr, rhs: Expr, type: Type): Expr => ({ kind: "Div", lhs, rhs, type });
const mod = (lhs: Expr, rhs: Expr, type: Type): Expr => ({ kind: "Mod", lhs, rhs, type });
const eqU32 = (lhs: Expr, rhs: Expr): Expr => ({ kind: "Eq", lhs, rhs, type: Tbool });
const select = (cond: Expr, ifTrue: Expr, ifFalse: Expr, type: Type): Expr => ({
  kind: "Conditional", cond, ifTrue, ifFalse, type,
});
const newVec = (components: Expr[], type: Type): Expr => ({
  kind: "NewVector", components, type,
});
const matFromRows = (rows: Expr[], type: Type): Expr => ({
  kind: "MatrixFromRows", rows, type,
});

// Runtime-sized array types for the heap storage buffers.
const arrU32: Type    = { kind: "Array", element: Tu32, length: "runtime" } as Type;
const arrF32: Type    = { kind: "Array", element: Tf32, length: "runtime" } as Type;
const arrVec4: Type   = { kind: "Array", element: Vec(Tf32, 4), length: "runtime" } as Type;
const Tvec2 = Vec(Tf32, 2);
const Tvec3 = Vec(Tf32, 3);
const Tvec4 = Vec(Tf32, 4);

// ─── Storage-buffer bindings ────────────────────────────────────────
//
// Bindings 0..3 are the four typed views of the heap arena. Bindings
// 4+ (textures/samplers) are added by the layout / runtime as needed.

function heapStorageBufferDecls(): ValueDef[] {
  return [
    { kind: "StorageBuffer", binding: { group: 0, slot: 0 }, name: "heapU32",    layout: arrU32,  access: "read" },
    { kind: "StorageBuffer", binding: { group: 0, slot: 1 }, name: "headersU32", layout: arrU32,  access: "read" },
    { kind: "StorageBuffer", binding: { group: 0, slot: 2 }, name: "heapF32",    layout: arrF32,  access: "read" },
    { kind: "StorageBuffer", binding: { group: 0, slot: 3 }, name: "heapV4f",    layout: arrVec4, access: "read" },
  ];
}

// ─── Heap-load IR builders ─────────────────────────────────────────
//
// Storage buffers are exposed in IR as `ReadInput("Uniform", name)`
// (the same scope WGSL uses for all bindings). Indexing is `Item`.
// Address arithmetic is plain Add/Mul on u32.

const heapU32     = readScope("Uniform", "heapU32",    arrU32);
const headersU32  = readScope("Uniform", "headersU32", arrU32);
const heapF32     = readScope("Uniform", "heapF32",    arrF32);
const heapV4f     = readScope("Uniform", "heapV4f",    arrVec4);

/** `headersU32[drawIdx * stride + offset]`. */
function loadHeaderRef(drawIdx: Expr, fieldByteOffset: number, strideU32: number): Expr {
  // u32 index = drawIdx * strideU32 + (fieldByteOffset / 4).
  const offU32 = fieldByteOffset / 4;
  const idx = add(mul(drawIdx, constU32(strideU32), Tu32), constU32(offU32), Tu32);
  return item(headersU32, idx, Tu32);
}

/**
 * Cyclic per-element index: `(elemIdx % count) * floatsPerElement`,
 * where count = alloc-header length (u32 at offset 4, i.e. ref/4 + 1).
 * For per-vertex attributes elemIdx = vid; for instance arrays =
 * iidx (length is then the instance count, no cycling needed but the
 * mod is harmless).
 */
function cyclicElemIdx(refIdent: Expr, elemIdx: Expr, floatsPerElem: number): Expr {
  const count = item(heapU32, add(div(refIdent, constU32(4), Tu32), constU32(1), Tu32), Tu32);
  const cycled = mod(elemIdx, count, Tu32);
  return mul(cycled, constU32(floatsPerElem), Tu32);
}

/**
 * Load a per-draw uniform value from the heap given its `u32` ref.
 * Type-driven: matrix → 4 vec4 columns from heapV4f; vec4 → one
 * heapV4f read; vec3/2/scalar → heapF32 reads + NewVector.
 */
function loadUniformByRef(refIdent: Expr, wgslType: string): Expr {
  switch (wgslType) {
    case "mat4x4<f32>": {
      // base vec4 index = (ref + 16) / 16
      const base = div(add(refIdent, constU32(16), Tu32), constU32(16), Tu32);
      const cols: Expr[] = [];
      for (let i = 0; i < 4; i++) {
        const idx = i === 0 ? base : add(base, constU32(i), Tu32);
        cols.push(item(heapV4f, idx, Tvec4));
      }
      return matFromRows(cols, { kind: "Matrix", element: Tf32, rows: 4, cols: 4 } as Type);
    }
    case "vec4<f32>":
      return item(heapV4f, div(add(refIdent, constU32(16), Tu32), constU32(16), Tu32), Tvec4);
    case "vec3<f32>": {
      const base = div(add(refIdent, constU32(16), Tu32), constU32(4), Tu32);
      return newVec([
        item(heapF32, base, Tf32),
        item(heapF32, add(base, constU32(1), Tu32), Tf32),
        item(heapF32, add(base, constU32(2), Tu32), Tf32),
      ], Tvec3);
    }
    case "vec2<f32>": {
      const base = div(add(refIdent, constU32(16), Tu32), constU32(4), Tu32);
      return newVec([
        item(heapF32, base, Tf32),
        item(heapF32, add(base, constU32(1), Tu32), Tf32),
      ], Tvec2);
    }
    case "f32":
      return item(heapF32, div(add(refIdent, constU32(16), Tu32), constU32(4), Tu32), Tf32);
    case "u32":
      return item(heapU32, div(add(refIdent, constU32(16), Tu32), constU32(4), Tu32), Tu32);
    default:
      throw new Error(`heapEffectIR: no IR loader for uniform type '${wgslType}'`);
  }
}

/**
 * Load an attribute value with cyclic addressing (`(idx % length) *
 * stride`). The mod lets a length-1 broadcast replicate across all
 * vertices. For vec4 the source can be V3-tight (12 B, .w = 1.0) or
 * V4-tight (16 B, .w from data); we read the actual stride from the
 * alloc header at offset 8, then use `select` for the .w fill.
 */
function loadAttributeByRef(refIdent: Expr, idx: Expr, wgslType: string): Expr {
  const dataF32Base = div(add(refIdent, constU32(16), Tu32), constU32(4), Tu32);
  switch (wgslType) {
    case "vec3<f32>": {
      const off = cyclicElemIdx(refIdent, idx, 3);
      const base = add(dataF32Base, off, Tu32);
      return newVec([
        item(heapF32, base, Tf32),
        item(heapF32, add(base, constU32(1), Tu32), Tf32),
        item(heapF32, add(base, constU32(2), Tu32), Tf32),
      ], Tvec3);
    }
    case "vec4<f32>": {
      // strideF = stride_bytes / 4  (header u32 at ref/4 + 2).
      const strideBytes = item(heapU32, add(div(refIdent, constU32(4), Tu32), constU32(2), Tu32), Tu32);
      const strideF = div(strideBytes, constU32(4), Tu32);
      const cycled = mod(idx, item(heapU32, add(div(refIdent, constU32(4), Tu32), constU32(1), Tu32), Tu32), Tu32);
      const off = mul(cycled, strideF, Tu32);
      const base = add(dataF32Base, off, Tu32);
      const w = select(eqU32(strideF, constU32(4)), item(heapF32, add(base, constU32(3), Tu32), Tf32), constF32(1.0), Tf32);
      return newVec([
        item(heapF32, base, Tf32),
        item(heapF32, add(base, constU32(1), Tu32), Tf32),
        item(heapF32, add(base, constU32(2), Tu32), Tf32),
        w,
      ], Tvec4);
    }
    case "vec2<f32>": {
      const off = cyclicElemIdx(refIdent, idx, 2);
      const base = add(dataF32Base, off, Tu32);
      return newVec([
        item(heapF32, base, Tf32),
        item(heapF32, add(base, constU32(1), Tu32), Tf32),
      ], Tvec2);
    }
    default:
      throw new Error(`heapEffectIR: no IR loader for attribute type '${wgslType}'`);
  }
}

// ─── Instance-index / vertex-index built-in plumbing ────────────────
//
// The user's stages don't have @builtin(instance_index)/(vertex_index)
// inputs — they read inputs by name (vertexAttributes, uniforms).
// We need those builtins to address the heap. Inject them as
// EntryParameters on every vertex stage.

function entryHasBuiltin(e: EntryDef, semantic: string): boolean {
  return e.inputs.some(p =>
    p.decorations.some(d => d.kind === "Builtin" && d.value === semantic),
  );
}

function builtinParam(semantic: "vertex_index" | "instance_index"): EntryParameter {
  return {
    name: semantic,
    type: Tu32,
    semantic,
    decorations: [{ kind: "Builtin", value: semantic } as ParamDecoration],
  };
}

function injectVsBuiltins(m: Module): Module {
  const values = m.values.map((v): typeof v => {
    if (v.kind !== "Entry" || v.entry.stage !== "vertex") return v;
    let inputs = v.entry.inputs;
    if (!entryHasBuiltin(v.entry, "vertex_index")) {
      inputs = [...inputs, builtinParam("vertex_index")];
    }
    if (!entryHasBuiltin(v.entry, "instance_index")) {
      inputs = [...inputs, builtinParam("instance_index")];
    }
    return { ...v, entry: { ...v.entry, inputs } };
  });
  return { ...m, values };
}

// ─── VS body rewrites ──────────────────────────────────────────────
//
// For each surviving uniform field, build a heap-load expression
// using `instance_index` as drawIdx. For each surviving attribute,
// use `vertex_index` and cyclic addressing.
//
// These map builders extract the (name → Expr) mapping that the VS
// substitution needs; the actual rewrite is delegated to
// `Effect.substitute({ vertex: { uniforms, inputs } })` in
// `compileHeapEffectIR`.

/**
 * Per-RO selector expression. Always reads the `heap_drawIdx` local
 * defined by the megacall prelude.
 */
function drawIdxExprFor(_layout: BucketLayout): Expr {
  return { kind: "Var", var: { name: "heap_drawIdx", type: Tu32, mutable: false } } as Expr;
}

/**
 * `(uniformName → loadUniformByRef(loadHeaderRef(...), wgslType))` for
 * every uniform-ref field in `layout`. Used as the `vertex.uniforms`
 * argument to `Effect.substitute`.
 */
function buildVertexUniformMap(layout: BucketLayout): Map<string, Expr> {
  const drawIdx: Expr = drawIdxExprFor(layout);
  const iidx: Expr = readScope("Builtin", "instance_index", Tu32);
  const stride = layout.strideU32;
  const out = new Map<string, Expr>();
  for (const f of layout.drawHeaderFields) {
    if (f.kind !== "uniform-ref") continue;
    const refExpr = loadHeaderRef(drawIdx, f.byteOffset, stride);
    const value = layout.perInstanceUniforms.has(f.name)
      ? loadInstanceByRef(refExpr, iidx, f.uniformWgslType ?? "")
      : loadUniformByRef(refExpr, f.uniformWgslType ?? "");
    out.set(f.name, value);
  }
  return out;
}

/**
 * `(attributeName → loadAttributeByRef(loadHeaderRef(...), idx, wgslType))`
 * for every attribute-ref field in `layout`. Used as the
 * `vertex.inputs` argument to `Effect.substitute`.
 */
function buildVertexAttributeMap(layout: BucketLayout): Map<string, Expr> {
  const drawIdx: Expr = drawIdxExprFor(layout);
  const iidx: Expr = readScope("Builtin", "instance_index", Tu32);
  const vid = readScope("Builtin", "vertex_index", Tu32);
  const stride = layout.strideU32;
  const out = new Map<string, Expr>();
  for (const f of layout.drawHeaderFields) {
    if (f.kind !== "attribute-ref") continue;
    const refExpr = loadHeaderRef(drawIdx, f.byteOffset, stride);
    const value = layout.perInstanceAttributes.has(f.name)
      ? loadAttributeByRef(refExpr, iidx, f.attributeWgslType ?? "")
      : loadAttributeByRef(refExpr, vid, f.attributeWgslType ?? "");
    out.set(f.name, value);
  }
  return out;
}

function mapVertexEntryBodies(m: Module, fn: (body: Stmt) => Stmt): Module {
  return mapStageEntryBodies(m, "vertex", fn);
}

function mapStageEntryBodies(
  m: Module,
  stage: "vertex" | "fragment" | "compute",
  fn: (body: Stmt) => Stmt,
): Module {
  const values = m.values.map((v): typeof v => {
    if (v.kind !== "Entry" || v.entry.stage !== stage) return v;
    return { ...v, entry: { ...v.entry, body: fn(v.entry.body) } };
  });
  return { ...m, values };
}

/** Append `extras` to every VS entry's outputs and every FS entry's inputs. */
function addInterstageParams(
  m: Module,
  extras: readonly EntryParameter[],
): Module {
  if (extras.length === 0) return m;
  const values = m.values.map((v): typeof v => {
    if (v.kind !== "Entry") return v;
    if (v.entry.stage === "vertex") {
      return { ...v, entry: { ...v.entry, outputs: [...v.entry.outputs, ...extras] } };
    }
    if (v.entry.stage === "fragment") {
      return { ...v, entry: { ...v.entry, inputs: [...v.entry.inputs, ...extras] } };
    }
    return v;
  });
  return { ...m, values };
}

/**
 * Prepend `stmts` to every VS entry's body. Flatten when the body is
 * already a `Sequential` — CSE numbers `_cse0..N` per Sequential, so
 * nested Sequentials produce colliding declarations in the emitted
 * WGSL.
 */
function prependVsBodyStmts(m: Module, stmts: readonly Stmt[]): Module {
  if (stmts.length === 0) return m;
  return mapVertexEntryBodies(m, body => {
    if (body.kind === "Sequential") {
      return { kind: "Sequential", body: [...stmts, ...body.body] };
    }
    return { kind: "Sequential", body: [...stmts, body] };
  });
}

// ─── FS uniform threading ──────────────────────────────────────────
//
// FS can't access `instance_index` directly, so we thread the per-
// uniform refs through VsOut as flat-interpolated u32 varyings:
//
//   - For each FS-used uniform `X` whose layout entry is uniform-ref:
//       VS output: `_h_<X>Ref: u32 @interpolate(flat)`
//       VS body:   `WriteOutput("_h_<X>Ref", loadHeaderRef(drawIdx, X))`
//       FS input:  `_h_<X>Ref: u32 @interpolate(flat)`
//       FS body:   `ReadInput("Uniform", X)` →
//                  `loadUniformByRef(ReadInput("Input", "_h_<X>Ref"), …)`
//
// For per-instance uniforms (layout.perInstanceUniforms), iidx is also
// threaded as `_iidx: u32 @interpolate(flat)` and the FS uses
// `loadInstanceByRef` instead.

function wgslTypeFromHeapField(wgslType: string): Type {
  switch (wgslType) {
    case "mat4x4<f32>": return { kind: "Matrix", element: Tf32, rows: 4, cols: 4 } as Type;
    case "vec4<f32>":   return Tvec4;
    case "vec3<f32>":   return Tvec3;
    case "vec2<f32>":   return Tvec2;
    case "f32":         return Tf32;
    case "u32":         return Tu32;
    default: throw new Error(`heapEffectIR: unknown uniform wgsl type '${wgslType}'`);
  }
}

/**
 * Family-member FS uniform rewrite (direct heap reads).
 *
 * The wrapper module threads `heap_drawIdx` and `instId` as plain u32
 * flat varyings into the FS, then passes them to each per-effect FS
 * helper. Inside the helper, every `ReadInput("Uniform", X)` is
 * substituted with `loadUniformByRef(loadHeaderRef(heap_drawIdx, …), …)`
 * (or `loadInstanceByRef(refExpr, instId, …)` for per-instance
 * uniforms). No VS-side WriteOutput stmts and no `_h_*Ref` varyings —
 * the carrier is the single `heap_drawIdx` u32.
 *
 * `heap_drawIdx` and `instId` are referenced as `Var` exprs; the post-
 * emit `applyFamilyMemberFsShape` rewires the FS entry signature to
 * surface them as plain u32 fn parameters.
 */
function rewriteFsUniformsDirect(m: Module, layout: BucketLayout): Module {
  const used = new Set<string>();
  for (const v of m.values) {
    if (v.kind !== "Entry" || v.entry.stage !== "fragment") continue;
    for (const sn of readInputs(v.entry.body).values()) {
      if (sn.scope === "Uniform") used.add(sn.name);
    }
  }
  if (used.size === 0) return m;

  const fieldByName = new Map(layout.drawHeaderFields.map(f => [f.name, f]));
  const drawIdxExpr: Expr = { kind: "Var", var: { name: "heap_drawIdx", type: Tu32, mutable: false } } as Expr;
  const instIdExpr:  Expr = { kind: "Var", var: { name: "instId",       type: Tu32, mutable: false } } as Expr;
  const fsSubst = new Map<string, Expr>();
  for (const name of used) {
    const f = fieldByName.get(name);
    if (f === undefined || f.kind !== "uniform-ref") continue;
    const refExpr = loadHeaderRef(drawIdxExpr, f.byteOffset, layout.strideU32);
    const value = layout.perInstanceUniforms.has(name)
      ? loadInstanceByRef(refExpr, instIdExpr, f.uniformWgslType ?? "")
      : loadUniformByRef(refExpr, f.uniformWgslType ?? "");
    fsSubst.set(name, value);
  }
  return substituteInputsInStage(m, "fragment", "Uniform", n => fsSubst.get(n));
}

/**
 * Family-member FS atlas rewrite (direct heap reads).
 *
 * Like `rewriteFsAtlasTextures` but instead of threading
 * `_h_<name>{PageRef,FormatBits,Origin,Size}` flat varyings from VS,
 * the substituted `atlasSample(pageRef, formatBits, origin, size, uv)`
 * call reads each header field inline from `headersU32` via
 * `heap_drawIdx`. The wrapper-supplied `heap_drawIdx` u32 fn parameter
 * carries the per-draw header index.
 */
function rewriteFsAtlasTexturesDirect(m: Module, layout: BucketLayout): Module {
  if (layout.atlasTextureBindings.size === 0) return m;
  const used = new Set<string>();
  for (const v of m.values) {
    if (v.kind !== "Entry" || v.entry.stage !== "fragment") continue;
    visitStmtExprs(v.entry.body, e => {
      const hit = isAtlasSampleCall(e, layout.atlasTextureBindings);
      if (hit !== null) used.add(hit.name);
    });
  }
  if (used.size === 0) return m;

  const fieldByAtlas = new Map<string, Map<string, ReturnType<typeof byteOffsetOf>>>();
  for (const f of layout.drawHeaderFields) {
    if (f.kind !== "texture-ref" || f.textureBindingName === undefined) continue;
    const sub = f.textureSub!;
    let inner = fieldByAtlas.get(f.textureBindingName);
    if (inner === undefined) { inner = new Map(); fieldByAtlas.set(f.textureBindingName, inner); }
    inner.set(sub, byteOffsetOf(f.byteOffset));
  }

  const drawIdxExpr: Expr = { kind: "Var", var: { name: "heap_drawIdx", type: Tu32, mutable: false } } as Expr;
  const stride = layout.strideU32;
  const headerU32At = (offU32: number): Expr =>
    item(headersU32, add(mul(drawIdxExpr, constU32(stride), Tu32), constU32(offU32), Tu32), Tu32);
  const headerF32At = (offU32: number): Expr => ({
    kind: "CallIntrinsic",
    op: { name: "bitcast<f32>", returnTypeOf: () => Tf32, pure: true,
          emit: { glsl: "intBitsToFloat", wgsl: "bitcast<f32>" } } as IntrinsicRef,
    args: [headerU32At(offU32)],
    type: Tf32,
  });

  return mapStageEntryBodies(m, "fragment", body => mapStmt(body, {
    expr: e => mapExpr(e, sub => {
      const hit = isAtlasSampleCall(sub, layout.atlasTextureBindings);
      if (hit === null) return sub;
      const fields = fieldByAtlas.get(hit.name)!;
      const pageRefOff    = fields.get("pageRef")!.u32;
      const formatBitsOff = fields.get("formatBits")!.u32;
      const originOff     = fields.get("origin")!.u32;
      const sizeOff       = fields.get("size")!.u32;
      const args: Expr[] = [
        headerU32At(pageRefOff),
        headerU32At(formatBitsOff),
        newVec([headerF32At(originOff),     headerF32At(originOff + 1)],     Tvec2),
        newVec([headerF32At(sizeOff),       headerF32At(sizeOff + 1)],       Tvec2),
        hit.uv,
      ];
      return { kind: "CallIntrinsic", op: ATLAS_SAMPLE_INTRINSIC, args, type: Tvec4 };
    }),
  }));
}

/**
 * FS uniform threading: thread `_h_<name>Ref` (and `_iidx`, when
 * needed) varyings from VS to FS, and rewrite FS `ReadInput("Uniform",
 * X)` into a heap-load through the threaded ref.
 *
 * This pass stays at Module-level rather than going through the
 * Effect.substitute API for two reasons:
 *
 *   1. The VS side adds *Stmts* (WriteOutput) and entry parameters,
 *      not Expr substitutions — outside the substitute API's scope.
 *   2. Even the FS Expr substitution can't run pre-merge: the
 *      frontend hides `return { outColor: ... }` field exprs in a
 *      `_record` carrier that mapExpr doesn't traverse, so any
 *      uniform read inside the returned record is invisible until
 *      `liftReturns` rewrites it into explicit WriteOutput stmts.
 *      We run after liftReturns; the matching VS-side write injection
 *      lives in the same pass to keep the `_h_<X>Ref` names paired.
 */
function rewriteFsUniforms(m: Module, layout: BucketLayout): Module {
  // Scan FS bodies for uniform-scope reads.
  const used = new Set<string>();
  for (const v of m.values) {
    if (v.kind !== "Entry" || v.entry.stage !== "fragment") continue;
    for (const sn of readInputs(v.entry.body).values()) {
      if (sn.scope === "Uniform") used.add(sn.name);
    }
  }
  if (used.size === 0) return m;

  const fieldByName = new Map(layout.drawHeaderFields.map(f => [f.name, f]));
  const drawIdxExpr = drawIdxExprFor(layout);
  const threadParams: EntryParameter[] = [];
  const vsWrites: Stmt[] = [];
  const fsSubst = new Map<string, Expr>();

  for (const name of used) {
    const f = fieldByName.get(name);
    if (f === undefined || f.kind !== "uniform-ref") continue;
    const refParamName = `_h_${name}Ref`;
    const refExprWriter = loadHeaderRef(drawIdxExpr, f.byteOffset, layout.strideU32);
    threadParams.push({
      name: refParamName, type: Tu32, semantic: refParamName,
      decorations: [{ kind: "Interpolation", mode: "flat" } as ParamDecoration],
    });
    vsWrites.push({
      kind: "WriteOutput", name: refParamName,
      value: { kind: "Expr", value: refExprWriter },
    });
    const refReadFs = readScope("Input", refParamName, Tu32);
    const value = layout.perInstanceUniforms.has(name)
      ? loadInstanceByRef(refReadFs, readScope("Input", "_iidx", Tu32), f.uniformWgslType ?? "")
      : loadUniformByRef(refReadFs, f.uniformWgslType ?? "");
    fsSubst.set(name, value);
  }

  // Per-instance: also thread iidx if any FS-used uniform is per-instance.
  const fsHasPerInstance = [...used].some(n => layout.perInstanceUniforms.has(n));
  if (fsHasPerInstance) {
    threadParams.push({
      name: "_iidx", type: Tu32, semantic: "_iidx",
      decorations: [{ kind: "Interpolation", mode: "flat" } as ParamDecoration],
    });
    vsWrites.push({
      kind: "WriteOutput", name: "_iidx",
      value: { kind: "Expr", value: readScope("Builtin", "instance_index", Tu32) },
    });
  }

  let out = m;
  out = addInterstageParams(out, threadParams);
  out = prependVsBodyStmts(out, vsWrites);
  out = substituteInputsInStage(out, "fragment", "Uniform", n => fsSubst.get(n));
  return out;
}

/**
 * Per-instance uniform load: indexes into a packed array allocation
 * by `iidx` (instance index in [0, instanceCount-1]).
 */
function loadInstanceByRef(refIdent: Expr, iidx: Expr, wgslType: string): Expr {
  const baseV4 = div(add(refIdent, constU32(16), Tu32), constU32(16), Tu32);
  const baseF32 = div(add(refIdent, constU32(16), Tu32), constU32(4), Tu32);
  switch (wgslType) {
    case "mat4x4<f32>": {
      const start = add(baseV4, mul(iidx, constU32(4), Tu32), Tu32);
      const cols: Expr[] = [];
      for (let i = 0; i < 4; i++) {
        cols.push(item(heapV4f, i === 0 ? start : add(start, constU32(i), Tu32), Tvec4));
      }
      return matFromRows(cols, { kind: "Matrix", element: Tf32, rows: 4, cols: 4 } as Type);
    }
    case "vec4<f32>":
      return item(heapV4f, add(baseV4, iidx, Tu32), Tvec4);
    case "vec3<f32>": {
      const off = mul(iidx, constU32(3), Tu32);
      const base = add(baseF32, off, Tu32);
      return newVec([
        item(heapF32, base, Tf32),
        item(heapF32, add(base, constU32(1), Tu32), Tf32),
        item(heapF32, add(base, constU32(2), Tu32), Tf32),
      ], Tvec3);
    }
    case "vec2<f32>": {
      const off = mul(iidx, constU32(2), Tu32);
      const base = add(baseF32, off, Tu32);
      return newVec([
        item(heapF32, base, Tf32),
        item(heapF32, add(base, constU32(1), Tu32), Tf32),
      ], Tvec2);
    }
    case "f32":
      return item(heapF32, add(baseF32, iidx, Tu32), Tf32);
    case "u32":
      return item(heapU32, add(baseF32, iidx, Tu32), Tu32);
    default:
      throw new Error(`heapEffectIR: no IR loader for per-instance uniform type '${wgslType}'`);
  }
}

// `wgslTypeFromHeapField` will be used once we propagate proper types
// onto the synthesised threading params; for now we only emit u32-typed
// refs, so the type lookup isn't called from the generated IR. Keep
// the import alive.
void wgslTypeFromHeapField;

// ─── FS atlas-texture rewrite (textureSample → atlasSample) ─────────
//
// For each atlas-routed binding actually called from FS as
// `textureSample(<name>, smp, uv)`:
//   - Thread 4 flat-interpolated VsOut varyings: `_h_<name>PageRef`,
//     `_h_<name>FormatBits`, `_h_<name>Origin`, `_h_<name>Size`.
//   - VS body writes them from the drawHeader (`headersU32` reads,
//     bitcast for the f32 fields since the headers buffer is u32).
//   - FS substitutes the textureSample call with a `CallIntrinsic`
//     emitting `atlasSample(pageRef, formatBits, origin, size, uv)`.
//
// The `atlasSample` helper is declared by the prelude string the post-
// emit injector prepends to the FS source — the IR doesn't need to
// know about its body.

/** Synthetic `atlasSample` intrinsic for the IR's CallIntrinsic emit hook. */
const ATLAS_SAMPLE_INTRINSIC: IntrinsicRef = {
  name: "atlasSample",
  returnTypeOf: () => Tvec4,
  pure: true,
  emit: { glsl: "atlasSample", wgsl: "atlasSample" },
};

function isAtlasSampleCall(e: Expr, atlasNames: ReadonlySet<string>): { name: string; uv: Expr } | null {
  if (e.kind !== "CallIntrinsic") return null;
  if (e.op.emit.wgsl !== "textureSample") return null;
  if (e.args.length < 2) return null;
  const tex = e.args[0]!;
  // Pre-legaliseTypes the sampler arg is a Var or ReadInput referencing
  // the Sampler ValueDef by its schema name. Match either shape.
  let name: string | undefined;
  if (tex.kind === "ReadInput" && tex.scope === "Uniform") name = tex.name;
  else if (tex.kind === "Var") name = (tex as { var: { name: string } }).var.name;
  if (name === undefined) return null;
  // The schema (extracted post-legaliseTypes) names the binding
  // `<sampler>_view`; the IR pre-pass sees the user-written sampler
  // name. Match either form so the rewrite fires regardless of which
  // shape the layout's atlas-binding set carries.
  const matched =
    atlasNames.has(name) ? name :
    atlasNames.has(`${name}_view`) ? `${name}_view` :
    undefined;
  if (matched === undefined) return null;
  // texture(sampler, uv) parses as a 2-arg call (no separate sampler/
  // texture pair until splitWgslSamplers runs). The uv is args[1].
  const uv = e.args[1]!;
  return { name: matched, uv };
}

function rewriteFsAtlasTextures(m: Module, layout: BucketLayout): Module {
  if (layout.atlasTextureBindings.size === 0) return m;
  // Scan FS bodies to find which atlas bindings are actually used.
  const used = new Set<string>();
  for (const v of m.values) {
    if (v.kind !== "Entry" || v.entry.stage !== "fragment") continue;
    visitStmtExprs(v.entry.body, e => {
      const hit = isAtlasSampleCall(e, layout.atlasTextureBindings);
      if (hit !== null) used.add(hit.name);
    });
  }
  if (used.size === 0) return m;

  const fieldByAtlas = new Map<string, Map<string, ReturnType<typeof byteOffsetOf>>>();
  for (const f of layout.drawHeaderFields) {
    if (f.kind !== "texture-ref" || f.textureBindingName === undefined) continue;
    const sub = f.textureSub!;
    let inner = fieldByAtlas.get(f.textureBindingName);
    if (inner === undefined) { inner = new Map(); fieldByAtlas.set(f.textureBindingName, inner); }
    inner.set(sub, byteOffsetOf(f.byteOffset));
  }

  const drawIdxExpr = drawIdxExprFor(layout);
  const threadParams: EntryParameter[] = [];
  const vsWrites: Stmt[] = [];

  for (const name of used) {
    const v = atlasVaryingNames(name);
    const fields = fieldByAtlas.get(name)!;
    const stride = layout.strideU32;
    const headerU32At = (offU32: number): Expr =>
      item(headersU32, add(mul(drawIdxExpr, constU32(stride), Tu32), constU32(offU32), Tu32), Tu32);
    const headerF32At = (offU32: number): Expr => ({
      kind: "CallIntrinsic",
      op: { name: "bitcast<f32>", returnTypeOf: () => Tf32, pure: true,
            emit: { glsl: "intBitsToFloat", wgsl: "bitcast<f32>" } } as IntrinsicRef,
      args: [headerU32At(offU32)],
      type: Tf32,
    });

    const pageRefOff    = fields.get("pageRef")!.u32;
    const formatBitsOff = fields.get("formatBits")!.u32;
    const originOff     = fields.get("origin")!.u32;
    const sizeOff       = fields.get("size")!.u32;

    threadParams.push({
      name: v.pageRef, type: Tu32, semantic: v.pageRef,
      decorations: [{ kind: "Interpolation", mode: "flat" } as ParamDecoration],
    });
    threadParams.push({
      name: v.formatBits, type: Tu32, semantic: v.formatBits,
      decorations: [{ kind: "Interpolation", mode: "flat" } as ParamDecoration],
    });
    threadParams.push({
      name: v.origin, type: Tvec2, semantic: v.origin,
      decorations: [{ kind: "Interpolation", mode: "flat" } as ParamDecoration],
    });
    threadParams.push({
      name: v.size, type: Tvec2, semantic: v.size,
      decorations: [{ kind: "Interpolation", mode: "flat" } as ParamDecoration],
    });

    vsWrites.push({ kind: "WriteOutput", name: v.pageRef,
      value: { kind: "Expr", value: headerU32At(pageRefOff) } });
    vsWrites.push({ kind: "WriteOutput", name: v.formatBits,
      value: { kind: "Expr", value: headerU32At(formatBitsOff) } });
    vsWrites.push({ kind: "WriteOutput", name: v.origin,
      value: { kind: "Expr", value: newVec(
        [headerF32At(originOff), headerF32At(originOff + 1)], Tvec2) } });
    vsWrites.push({ kind: "WriteOutput", name: v.size,
      value: { kind: "Expr", value: newVec(
        [headerF32At(sizeOff), headerF32At(sizeOff + 1)], Tvec2) } });
  }

  // Rewrite FS textureSample(...) → atlasSample(...) for atlas-routed names.
  let out = m;
  out = addInterstageParams(out, threadParams);
  out = prependVsBodyStmts(out, vsWrites);
  out = mapStageEntryBodies(out, "fragment", body => mapStmt(body, {
    expr: e => mapExpr(e, sub => {
      const hit = isAtlasSampleCall(sub, layout.atlasTextureBindings);
      if (hit === null) return sub;
      const v = atlasVaryingNames(hit.name);
      const args: Expr[] = [
        readScope("Input", v.pageRef,    Tu32),
        readScope("Input", v.formatBits, Tu32),
        readScope("Input", v.origin,     Tvec2),
        readScope("Input", v.size,       Tvec2),
        hit.uv,
      ];
      return { kind: "CallIntrinsic", op: ATLAS_SAMPLE_INTRINSIC, args, type: Tvec4 };
    }),
  }));
  return out;
}

/** Visit every Expr inside a Stmt tree, read-only. */
function visitStmtExprs(s: Stmt, fn: (e: Expr) => void): void {
  mapStmt(s, {
    expr: e => mapExpr(e, sub => { fn(sub); return sub; }),
  });
}

function byteOffsetOf(byteOffset: number): { u32: number } {
  return { u32: byteOffset / 4 };
}

// ─── Public entry point ────────────────────────────────────────────

/**
 * Compile a wombat.shader Effect into heap-rendering WGSL via IR-level
 * substitution.
 *
 * v1 scope:
 *   - Adds heap storage buffers + builtin VS params.
 *   - Substitutes VS attribute / VS uniform reads with heap loads.
 *   - FS uniform substitution still TBD — falls back to current
 *     regex rewriter for FS until the cross-stage threading is wired.
 */
/**
 * Emission mode for `compileHeapEffectIR`.
 *
 *   - `"standalone"` (default): emit a self-contained pipeline. The VS
 *     entry takes `@builtin(vertex_index) emitIdx` and runs the megacall
 *     binary-search prelude inline; storage-buffer bindings 4..6 for
 *     drawTable / indexStorage / firstDrawInTile are appended.
 *   - `"family-member"`: emit a VS entry shaped to be invoked from a
 *     family wrapper. The wrapper performs the megacall search and
 *     passes `heap_drawIdx`, `instId`, `vid` down. The emitted entry
 *     becomes `@vertex fn vs(heap_drawIdx: u32, instId: u32, vid: u32)`;
 *     no megacall prelude or bindings 4..6 are emitted at this level.
 */
export type HeapEffectEmitMode = "standalone" | "family-member";

export interface HeapEffectIR { vs: string; fs: string; preludeWgsl: string; vsEntry: string; fsEntry: string }

// ─── Compiled-heap-effect-IR cache (in-memory tier 1 + localStorage tier 2)
//
// `compileHeapEffectIRViaDecoder` is the IR-rewrite + WGSL-emit step —
// the ~15% of cold boot in the profile. Its output is fully determined
// by `(userEffect.id, layout.id, mode, compileOptions)`, all stable
// content hashes/values, and is plain JSON (five WGSL strings). tier 1 =
// module Map (no re-walk within a run); tier 2 = localStorage (no
// re-walk across reloads, keyed under `HEAP_PERSIST_VERSION` — see
// heapEffect.ts; cold start unchanged).

function compileOptionsKey(o: CompileOptions): string {
  return o.target +
    (o.skipOptimisations ? ":raw" : "") +
    (o.fragmentOutputLayout !== undefined
      ? ":fbo[" + [...o.fragmentOutputLayout.locations.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([n, l]) => `${n}=${l}`).join("|") + "]"
      : "") +
    (o.instanceAttributes && o.instanceAttributes.size > 0 ? ":inst[" + [...o.instanceAttributes].sort().join(",") + "]" : "");
}

const _heapIrMemCache = new Map<string, HeapEffectIR>();

function isHeapEffectIR(v: unknown): v is HeapEffectIR {
  return typeof v === "object" && v !== null
    && typeof (v as HeapEffectIR).vs === "string" && typeof (v as HeapEffectIR).fs === "string"
    && typeof (v as HeapEffectIR).preludeWgsl === "string"
    && typeof (v as HeapEffectIR).vsEntry === "string" && typeof (v as HeapEffectIR).fsEntry === "string";
}

export function compileHeapEffectIR(
  userEffect: Effect,
  layout: BucketLayout,
  compileOptions: CompileOptions,
  mode: HeapEffectEmitMode = "standalone",
): HeapEffectIR {
  const contentKey = `${userEffect.id}|${layout.id}|${mode}|${compileOptionsKey(compileOptions)}`;
  const mem = _heapIrMemCache.get(contentKey);
  if (mem !== undefined) return mem;
  const lsKey = persistKey(HEAP_PERSIST_VERSION, "ir", contentKey);
  const persisted = lsLoad(lsKey, isHeapEffectIR);
  if (persisted !== undefined) { _heapIrMemCache.set(contentKey, persisted); return persisted; }

  // Both modes run through the composition-based heap rewrite.
  // Family-member mode skips the megacall prelude in the decoder
  // (wrapper VS owns it) and re-shapes the emitted @vertex into a
  // regular function so heapShaderFamily can splice it into the family
  // wrapper.
  const result = compileHeapEffectIRViaDecoder(userEffect, layout, compileOptions, mode);
  _heapIrMemCache.set(contentKey, result);
  lsStore(lsKey, result);
  return result;

  // ────────── legacy path (no longer reached) ──────────
  // Kept below temporarily for reference during the migration. The
  // unreachable code is dead-code-eliminated by the TS compiler.
  // eslint-disable-next-line no-unreachable
  const heapShaped = userEffect.substitute({
    vertex: {
      inputs:   buildVertexAttributeMap(layout),
      uniforms: buildVertexUniformMap(layout),
    },
  });

  // Build a single Module from all of the effect's stages by
  // composing them. Then apply the remaining heap rewrites to that
  // combined module. The composed module preserves stage entry
  // boundaries.
  let combined: Module = mergeStages(heapShaped);
  // Lift `return { ... }` into explicit WriteOutput stmts so the FS
  // uniform pass below can see (and rewrite) ReadInputs that feed the
  // returned record fields. Without this, ObjectLiteral exprs hide
  // their fields on a `_record` carrier that mapExpr doesn't traverse.
  combined = liftReturns(combined);
  combined = injectVsBuiltins(combined);
  // FS uniform threading must run BEFORE atlas FS rewriting in source
  // order? No — order historically: atlas first, then uniforms. Both
  // add interstage params and VS WriteOutput stmts; their write sets
  // are disjoint, so the order is interchangeable in practice. Keep
  // the original order for byte-for-byte equivalence.
  if (mode === "family-member") {
    // Direct heap-read FS path: substitute uniforms / atlas samples
    // directly using the wrapper-supplied `heap_drawIdx` / `instId`
    // u32 fn parameters. No VS→FS varying threading on the heap side.
    combined = rewriteFsAtlasTexturesDirect(combined, layout);
    combined = rewriteFsUniformsDirect(combined, layout);
  } else {
    combined = rewriteFsAtlasTextures(combined, layout);
    combined = rewriteFsUniforms(combined, layout);
  }
  // Add heap storage buffers as bindings.
  combined = { ...combined, values: [...heapStorageBufferDecls(), ...combined.values] };

  // Skip frontend resolveHoles / liftReturns since they expect the
  // template not to have been pre-modified. Compile via compileModule
  // directly which still runs assignLocations + linkCrossStage + DCE +
  // emit. (`makeStage` wrapper goes through Effect.compile which calls
  // compileModule too.)
  const compiled = compileModule(combined, compileOptions);
  const vsStage = compiled.stages.find(s => s.stage === "vertex");
  const fsStage = compiled.stages.find(s => s.stage === "fragment");
  let vsSrc = vsStage?.source ?? "";
  let fsSrc = fsStage?.source ?? "";
  if (vsSrc.length > 0) {
    vsSrc = mode === "family-member"
      ? applyFamilyMemberShape(vsSrc)
      : applyMegacallToEmittedVs(vsSrc);
  }
  if (mode === "family-member" && fsSrc.length > 0) {
    fsSrc = applyFamilyMemberFsShape(fsSrc);
  }
  if (layout.atlasTextureBindings.size > 0) {
    // Strip user-emitted texture/sampler binding decls for atlas-routed
    // names — the IR's FS still emits `var <name>: texture_2d<f32>;` even
    // though we've replaced every textureSample call referencing it.
    // The render BGL has no slot for these bindings.
    if (vsSrc.length > 0) vsSrc = stripAtlasTextureSamplerDecls(vsSrc, layout.atlasTextureBindings);
    if (fsSrc.length > 0) {
      fsSrc = stripAtlasTextureSamplerDecls(fsSrc, layout.atlasTextureBindings);
      fsSrc = prependAtlasPrelude(fsSrc);
    }
  }
  return {
    vs: vsSrc,
    fs: fsSrc,
    vsEntry: vsStage?.entryName ?? "vs",
    fsEntry: fsStage?.entryName ?? "fs",
    preludeWgsl: "",
  };
}

/**
 * Composition-based heap rewrite (standalone mode).
 *
 * Pipeline:
 *   1. Merge user effect stages into one Module.
 *   2. `uniformsToInputs` — every per-RO uniform name from the bucket
 *      layout gets renamed from Uniform scope to Input scope (so the
 *      decoder VS writes it into the carrier, and the rename is visible
 *      to BOTH the VS and the FS sides simultaneously). The pass also
 *      surfaces matching `EntryParameter` declarations on every entry
 *      whose body reads them — required so `extractFusedEntry`'s State-
 *      struct construction sees them as ports.
 *   3. Synthesise the heap decoder VS Module from the layout
 *      (`synthesizeHeapDecoderModule`). The decoder owns the megacall
 *      search and writes every drawHeader field — per-vertex
 *      attributes, per-RO uniforms, per-instance variants, and atlas
 *      texture sub-fields — onto the inter-stage carrier.
 *   4. Concatenate values: `[heapDecls, decoderEntry, ...userModule.values]`.
 *      `composeStages` (run by `compileModule`) fuses the decoder VS
 *      with the user VS via the standard State-struct path.
 *   5. `rewriteFsAtlasTexturesViaCarrier` — FS-side substitution only:
 *      every `textureSample(t, smp, uv)` for an atlas-routed binding
 *      becomes `atlasSample(Input._h_<t>PageRef, …, uv)`. The decoder
 *      already wrote those carrier slots in step 3 so no VS-side writes
 *      are needed.
 *   6. `liftReturns` + `injectVsBuiltins` + `compileModule`.
 *
 * No post-emit string mangling — `applyMegacallToEmittedVs` /
 * `applyFamilyMemberShape` are gone for this path.
 */
function compileHeapEffectIRViaDecoder(
  userEffect: Effect,
  layout: BucketLayout,
  compileOptions: CompileOptions,
  mode: HeapEffectEmitMode = "standalone",
): { vs: string; fs: string; preludeWgsl: string; vsEntry: string; fsEntry: string } {
  // 1. Merge user stages.
  let userModule: Module = mergeStages(userEffect);

  // 2. Rename per-RO uniforms (Uniform → Input). The set includes:
  //
  //   - every uniform-ref drawHeader name (post-RO uniforms the decoder
  //     loads per draw),
  //   - every attribute-ref drawHeader name (Sg.instanced's
  //     `instanceUniforms` rewrites these from Uniform → Input scope
  //     at Effect.compile time, but `mergeStages` accesses the raw
  //     stage template that still carries the original `Uniform` decl
  //     + Uniform-scope reads; renaming covers both shapes so neither
  //     a stale Uniform decl nor a stale Uniform-scope read survives).
  //
  // Names not in the drawHeader stay untouched (currently none in the
  // heap-everything path).
  const uniformRefNames = new Set<string>();
  for (const f of layout.drawHeaderFields) {
    if (f.kind === "uniform-ref" || f.kind === "attribute-ref") uniformRefNames.add(f.name);
  }
  userModule = uniformsToInputs(userModule, uniformRefNames);

  // 3. Build the decoder Module. Mode flows through: standalone owns
  // the megacall search; family-member receives drawIdx/instId/vid
  // from the wrapper VS as inputs and skips the search.
  const decoderModule = synthesizeHeapDecoderModule(layout, mode);

  // 4. Concatenate. Decoder values come first so the storage-buffer
  // bindings land at @binding(0)..@binding(6) consistently across
  // every emitted shader (the WGSL emit walks values in order and
  // assigns binding slots).
  let combined: Module = {
    types: [...decoderModule.types, ...userModule.types],
    values: [...decoderModule.values, ...userModule.values],
  };

  // 5. Standard preprocessing.
  combined = liftReturns(combined);
  combined = injectVsBuiltins(combined);

  // 6. FS-side atlas rewrite (texture-sample → atlas-sample, reading
  // from the carrier varyings the decoder wrote).
  combined = rewriteFsAtlasTexturesViaCarrier(combined, layout);

  // 7. Compile.
  const compiled = compileModule(combined, compileOptions);
  if ((globalThis as { __HEAP_DEBUG_FULL__?: boolean }).__HEAP_DEBUG_FULL__) {
    for (const s of compiled.stages) console.log(`[heap-debug-full] ${s.stage}:\n${s.source}`);
  }
  const vsStage = compiled.stages.find(s => s.stage === "vertex");
  const fsStage = compiled.stages.find(s => s.stage === "fragment");
  let vsSrc = vsStage?.source ?? "";
  let fsSrc = fsStage?.source ?? "";

  // 8. Family-member mode: re-shape the @vertex / @fragment entries
  // into plain function helpers the family wrapper can call. The body
  // is already correct (decoder + user-VS fused, all carrier values
  // computed); the entry headers just need to lose their stage
  // decorations and let location-decorated params become plain u32
  // function arguments.
  if (mode === "family-member") {
    if (vsSrc.length > 0) vsSrc = reshapeFamilyMemberVs(vsSrc);
    if (fsSrc.length > 0) fsSrc = reshapeFamilyMemberFs(fsSrc);
  }

  // 9. Final WGSL hygiene: atlas-routed texture/sampler bindings are
  // semantically gone (all sampling now goes through the atlas
  // helpers + page arrays), but emit may still produce decls for
  // them. Strip + prepend the atlas-sample helper library on FS.
  if (layout.atlasTextureBindings.size > 0) {
    if (vsSrc.length > 0) vsSrc = stripAtlasTextureSamplerDecls(vsSrc, layout.atlasTextureBindings);
    if (fsSrc.length > 0) {
      fsSrc = stripAtlasTextureSamplerDecls(fsSrc, layout.atlasTextureBindings);
      fsSrc = prependAtlasPrelude(fsSrc);
    }
  }

  return {
    vs: vsSrc,
    fs: fsSrc,
    vsEntry: vsStage?.entryName ?? "vs",
    fsEntry: fsStage?.entryName ?? "fs",
    preludeWgsl: "",
  };
}

/**
 * Family-member VS reshape — surface `(heap_drawIdx, instId, vid)` as
 * plain `u32` parameters of the helper fn so the family wrapper VS
 * can call it directly with the megacall-search outputs.
 *
 * The decoder synthesised the three names as `@location(N)`-decorated
 * inputs. After composeStages they live inside the entry's input
 * struct as `in.heap_drawIdx` / `in.instId` / `in.vid`. This pass:
 *
 *   1. Replaces the `(in: <InputStruct>)` parameter with three
 *      plain `u32` parameters by the same names.
 *   2. Inserts a body prologue that builds the input struct locally
 *      and seeds its three megacall fields from the new params, so
 *      the rest of the body (referring to `in.X`) keeps working.
 *
 * `@vertex`/`@fragment` decorations are NOT stripped here —
 * `heapShaderFamily.stripStageDecoration` does that during wrapper
 * assembly. Leaving them lets `readEntryName` find the helper name.
 *
 * DCE may have dropped one or both of `instId` / `vid` from the
 * input struct when nothing in the fused body reads them. The
 * pass detects this and emits a parameter only for fields that
 * survived; the wrapper can still pass three args (extras are
 * ignored by WGSL only if the signature accepts them — actually
 * they're not, so we DO need to keep all three names so the
 * wrapper's `family_X(drawIdx, instId, vid)` call typechecks).
 *
 * The three params are always emitted regardless of liveness; an
 * unused param is harmless and keeps the wrapper call site stable.
 */
function reshapeFamilyMemberVs(src: string): string {
  return reshapeFamilyMember(src, /^@vertex\s/m);
}

/** Family-member FS reshape — currently same shape as VS. */
function reshapeFamilyMemberFs(src: string): string {
  return reshapeFamilyMember(src, /^@fragment\s/m);
}

function reshapeFamilyMember(src: string, stageRe: RegExp): string {
  // Locate the @vertex / @fragment fn declaration.
  const stageMatch = stageRe.exec(src);
  if (stageMatch === null) return src;
  const fnRe = /fn\s+(\w+)\s*\(([^)]*)\)\s*(->\s*\w+)?\s*\{/y;
  fnRe.lastIndex = stageMatch.index + stageMatch[0].length;
  // Walk back over whitespace between the stage attr and `fn`.
  let i = fnRe.lastIndex;
  while (/\s/.test(src[i - 1] ?? "")) i--;
  fnRe.lastIndex = i;
  const fnMatch = fnRe.exec(src);
  if (fnMatch === null) {
    // Try without leading-whitespace assumption — find the next `fn ...`.
    const loose = /fn\s+(\w+)\s*\(([^)]*)\)\s*(->\s*\w+)?\s*\{/.exec(src.slice(stageMatch.index));
    if (loose === null) return src;
    return rewriteFn(src, stageMatch.index + loose.index!, loose);
  }
  return rewriteFn(src, fnMatch.index, fnMatch);
}

function rewriteFn(src: string, fnStart: number, fnMatch: RegExpExecArray): string {
  const fnName = fnMatch[1]!;
  const paramsRaw = fnMatch[2]!;
  const returnPart = fnMatch[3] ?? "";
  const params = paramsRaw.split(",").map((p) => p.trim()).filter((p) => p.length > 0);

  // Identify the `in: <InputStruct>` param (if any).
  let inStructName: string | undefined;
  const otherParams: string[] = [];
  for (const p of params) {
    const m = /^in\s*:\s*(\w+)$/.exec(p);
    if (m !== null && inStructName === undefined) {
      inStructName = m[1]!;
      continue;
    }
    otherParams.push(p);
  }

  // New parameter list: three megacall params first, then any other
  // params (rare; usually only `in`).
  const newParams = [
    "heap_drawIdx: u32",
    "instId: u32",
    "vid: u32",
    ...otherParams,
  ].join(", ");

  // Body prologue: only needed when the original entry actually had an
  // input struct that the body reads via `in.X`. We inject a local
  // `in` and seed its three megacall fields. Any other fields the
  // entry declared (vertex attribute Inputs, post-rename uniforms) are
  // satisfied via the fused State struct — they never read from `in`.
  const prologue = inStructName === undefined
    ? ""
    : `\n    var in: ${inStructName};\n    in.heap_drawIdx = heap_drawIdx;\n    in.instId = instId;\n    in.vid = vid;\n`;

  // Splice the new signature in.
  const before = src.slice(0, fnStart);
  const headerLen = fnMatch[0]!.length;
  // The match captured up to and including the `{`. Anything inside the
  // body follows.
  const bodyStart = fnStart + headerLen;
  const newHeader = `fn ${fnName}(${newParams})${returnPart ? " " + returnPart : ""} {${prologue}`;
  return before + newHeader + src.slice(bodyStart);
}

/**
 * FS-side atlas rewrite for the decoder-composition path.
 *
 * The decoder VS has already written `_h_<name>{PageRef,FormatBits,
 * Origin,Size}` onto the inter-stage carrier. This pass:
 *   1. Rewrites each `textureSample(t, smp, uv)` (or shipped
 *      `texture(...)` shorthand) for an atlas-routed binding into the
 *      atlas-sample intrinsic reading from `Input.<carrier name>`.
 *   2. Adds matching `EntryParameter`s on every FS Entry so the WGSL
 *      emit produces an Input struct that actually contains the
 *      `_h_<name><Sub>` fields the body reads.
 *
 * Unlike the legacy `rewriteFsAtlasTextures`, this does NOT inject
 * VS-side WriteOutputs (the decoder writes them already).
 */
function rewriteFsAtlasTexturesViaCarrier(m: Module, layout: BucketLayout): Module {
  if (layout.atlasTextureBindings.size === 0) return m;

  // Find which atlas-routed names the FS actually samples — only those
  // need carrier declarations and rewrites.
  const used = new Set<string>();
  for (const v of m.values) {
    if (v.kind !== "Entry" || v.entry.stage !== "fragment") continue;
    visitStmtExprs(v.entry.body, e => {
      const hit = isAtlasSampleCall(e, layout.atlasTextureBindings);
      if (hit !== null) used.add(hit.name);
    });
  }
  if (used.size === 0) return m;

  // Build the four carrier inputs per used texture. Flat interpolation
  // — integer types require it and the floats are constant per draw.
  const fsExtras: EntryParameter[] = [];
  for (const name of used) {
    const v = atlasVaryingNames(name);
    const flat: ParamDecoration = { kind: "Interpolation", mode: "flat" };
    fsExtras.push({ name: v.pageRef,    type: Tu32,  semantic: v.pageRef,    decorations: [flat] });
    fsExtras.push({ name: v.formatBits, type: Tu32,  semantic: v.formatBits, decorations: [flat] });
    fsExtras.push({ name: v.origin,     type: Tvec2, semantic: v.origin,     decorations: [flat] });
    fsExtras.push({ name: v.size,       type: Tvec2, semantic: v.size,       decorations: [flat] });
  }

  // Add the EntryParameter declarations to every FS entry's inputs.
  let out = m;
  out = {
    ...out,
    values: out.values.map((vv) => {
      if (vv.kind !== "Entry" || vv.entry.stage !== "fragment") return vv;
      const haveInput = new Set(vv.entry.inputs.map(p => p.name));
      const additions = fsExtras.filter(p => !haveInput.has(p.name));
      if (additions.length === 0) return vv;
      return { ...vv, entry: { ...vv.entry, inputs: [...vv.entry.inputs, ...additions] } };
    }),
  };

  // Body rewrite: textureSample → atlasSample(Input.pageRef, …, uv).
  out = mapStageEntryBodies(out, "fragment", body => mapStmt(body, {
    expr: e => mapExpr(e, sub => {
      const hit = isAtlasSampleCall(sub, layout.atlasTextureBindings);
      if (hit === null) return sub;
      const v = atlasVaryingNames(hit.name);
      const args: Expr[] = [
        readScope("Input", v.pageRef,    Tu32),
        readScope("Input", v.formatBits, Tu32),
        readScope("Input", v.origin,     Tvec2),
        readScope("Input", v.size,       Tvec2),
        hit.uv,
      ];
      return { kind: "CallIntrinsic", op: ATLAS_SAMPLE_INTRINSIC, args, type: Tvec4 };
    }),
  }));
  return out;
}

/**
 * Drop `@group(N) @binding(M) var <name>: texture_*<...>;` and the
 * matching sampler decl(s) for atlas-routed binding names. We keep
 * other texture/sampler decls intact since they belong to the standalone
 * texture path.
 *
 * Sampler-name detection: WGSL emit pairs samplers with their textures
 * positionally in the IR; the user's WGSL can name them anything. To
 * be safe we also strip any sampler decl that is *only* referenced by
 * a now-rewritten textureSample call. Rather than do data-flow, we
 * pattern-match: if the FS has zero remaining textureSample calls AND
 * a single sampler decl, drop it. Otherwise leave it. Practically the
 * v1 surface has 0..1 atlas-routed binding per shader, so this is
 * sufficient.
 */
function stripAtlasTextureSamplerDecls(src: string, atlasNames: ReadonlySet<string>): string {
  let out = src;
  for (const name of atlasNames) {
    const re = new RegExp(`@group\\(\\d+\\)\\s*@binding\\(\\d+\\)\\s*var\\s+${name}\\s*:\\s*texture_\\w+(?:<[^>]*>)?\\s*;\\s*`, "g");
    out = out.replace(re, "");
  }
  // If textureSample no longer appears anywhere, drop the (single)
  // sampler decl too. Conservative: we leave samplers alone if they
  // could still be in use.
  if (!/textureSample\s*\(/.test(out)) {
    out = out.replace(/@group\(\d+\)\s*@binding\(\d+\)\s*var\s+\w+\s*:\s*sampler(?:_comparison)?\s*;\s*/g, "");
  }
  return out;
}

/**
 * Prepend the atlas texture-array decls + `atlasSample` helper to
 * the emitted FS source. The helper signatures match the post-
 * rewrite `atlasSample(pageRef, formatBits, origin, size, uv)`
 * calls the IR pass produces.
 */
function prependAtlasPrelude(fs: string): string {
  // Delegate to the VS prelude generator — both stages need the same
  // atlas-sample helper, and consolidating prevents the two copies
  // from drifting.
  return generateAtlasPrelude() + fs;
}

/**
 * Megacall post-processing for the IR-emitted VS source. The IR uses
 * `@builtin(vertex_index) vertex_index` + `@builtin(instance_index)
 * instance_index` as the param names; rather than rewrite the IR
 * (which would require building the binary-search loop in IR), we
 * post-process the emitted WGSL: strip instance_index from the
 * @vertex signature, rename vertex_index to emitIdx, and inject a
 * prelude that defines `instance_index` + `vertex_index` as locals
 * via the drawTable binary search so existing body references keep
 * resolving. Also injects the drawTable/indexStorage binding decls.
 */
function applyMegacallToEmittedVs(vs: string): string {
  let s = vs;
  // Megacall storage bindings only — the shared values (`heap_drawIdx`,
  // `instId`, `vid`) are declared as `let` locals in the @vertex body
  // and threaded as parameters into any helper fn that references them.
  // No module-scope `var<private>` (Safari/WebKit rejects reading
  // those from helper fn bodies).
  const decl = `\n@group(0) @binding(4) var<storage, read> drawTable:       array<u32>;\n@group(0) @binding(5) var<storage, read> indexStorage:    array<u32>;\n@group(0) @binding(6) var<storage, read> firstDrawInTile: array<u32>;\n`;
  // Locate the @vertex fn header, then balance parens manually since
  // params can carry `@builtin(name)` decorations (regex `[^)]*` would
  // halt at the first inner `)`).
  const startRe = /@vertex\s+fn\s+(\w+)\s*\(/;
  const startMatch = s.match(startRe);
  if (startMatch === null) return s;
  const fnName = startMatch[1]!;
  const paramOpen = startMatch.index! + startMatch[0]!.length;
  let depth = 1;
  let i = paramOpen;
  for (; i < s.length && depth > 0; i++) {
    const c = s[i];
    if (c === "(") depth++;
    else if (c === ")") depth--;
  }
  if (depth !== 0) return s;
  const paramList = s.slice(paramOpen, i - 1);
  // After the closing `)` we expect `\s*->\s*<retType>\s*\{`.
  const tailRe = /\s*->\s*([\w<>,\s]+?)\s*\{/y;
  tailRe.lastIndex = i;
  const tailMatch = tailRe.exec(s);
  if (tailMatch === null) return s;
  const retType = tailMatch[1]!.trim();
  const headerEnd = tailRe.lastIndex;
  const headerStart = startMatch.index!;
  const params = paramList.split(",").map(p => p.trim()).filter(p => p.length > 0);
  const kept: string[] = [];
  for (const p of params) {
    if (/@builtin\(\s*instance_index\s*\)/.test(p)) continue;
    if (/@builtin\(\s*vertex_index\s*\)/.test(p)) {
      kept.push("@builtin(vertex_index) emitIdx: u32");
      continue;
    }
    kept.push(p);
  }
  // megacallSearchPrelude exposes `heap_drawIdx`, `instId`, `vid`. The
  // IR rewrites bind header-selector reads to `heap_drawIdx` directly
  // (loadHeaderRef takes an Expr arg); the remaining bindings expose
  // `instance_index` (= per-RO instance idx) and `vertex_index` for
  // user code that reads them by builtin.
  const newHeader = `@vertex fn ${fnName}(${kept.join(", ")}) -> ${retType} {\n${megacallSearchPrelude()}  let instance_index: u32 = instId;\n  let vertex_index: u32 = vid;\n`;
  s = s.slice(0, headerStart) + newHeader + s.slice(headerEnd);
  s = threadMegacallParamsThroughHelpers(s);
  return decl + s;
}

/**
 * Family-member shape: the wrapper performs the megacall binary search
 * and invokes this VS as a regular function, passing `heap_drawIdx`,
 * `instId`, and `vid` as plain `u32` parameters. We rewrite the emitted
 * @vertex signature so its first three params are exactly those names
 * (no `@builtin` decorations), and inject local aliases so any body
 * references to `vertex_index` / `instance_index` (the names produced
 * by IR `Builtin` reads + `injectVsBuiltins`) resolve to the new
 * parameters.
 *
 * The megacall storage-buffer bindings (drawTable / indexStorage /
 * firstDrawInTile) are NOT emitted — the wrapper module owns them.
 *
 * `threadMegacallParamsThroughHelpers` is run unchanged: helper fns
 * extracted by composeStages may still reference `heap_drawIdx`,
 * `instId`, `vid` after CSE; threading the params is a pure text walk
 * and does not depend on the entry shape.
 */
function applyFamilyMemberShape(vs: string): string {
  let s = vs;
  // Locate the @vertex fn header, then balance parens manually since
  // params can carry `@builtin(name)` decorations.
  const startRe = /@vertex\s+fn\s+(\w+)\s*\(/;
  const startMatch = s.match(startRe);
  if (startMatch === null) return s;
  const fnName = startMatch[1]!;
  const paramOpen = startMatch.index! + startMatch[0]!.length;
  let depth = 1;
  let i = paramOpen;
  for (; i < s.length && depth > 0; i++) {
    const c = s[i];
    if (c === "(") depth++;
    else if (c === ")") depth--;
  }
  if (depth !== 0) return s;
  const paramList = s.slice(paramOpen, i - 1);
  const tailRe = /\s*->\s*([\w<>,\s]+?)\s*\{/y;
  tailRe.lastIndex = i;
  const tailMatch = tailRe.exec(s);
  if (tailMatch === null) return s;
  const retType = tailMatch[1]!.trim();
  const headerEnd = tailRe.lastIndex;
  const headerStart = startMatch.index!;
  // Drop ALL @vertex fn params: the megacall builtins
  // (instance_index, vertex_index) are not needed because we re-bind
  // them from incoming u32 args below, and any @location(...) attribute
  // params have already had their body reads substituted with
  // heap-load expressions by `substituteInputsInStage` — the params
  // themselves are dead. The wrapper calls the helper with exactly
  // three megacall args.
  const newParamList = ["heap_drawIdx: u32", "instId: u32", "vid: u32"].join(", ");
  void paramList;
  // Body aliases: IR-emitted body code reads `vertex_index` and
  // `instance_index` (the names of the @builtin params standalone mode
  // adds). Re-bind those names to the incoming `vid` / `instId`.
  const bodyAliases =
    `  let instance_index: u32 = instId;\n` +
    `  let vertex_index: u32 = vid;\n`;
  const newHeader = `@vertex fn ${fnName}(${newParamList}) -> ${retType} {\n${bodyAliases}`;
  s = s.slice(0, headerStart) + newHeader + s.slice(headerEnd);
  s = threadMegacallParamsThroughHelpers(s);
  return s;
}

/**
 * Family-member FS shape: rewrite the emitted `@fragment fn` signature
 * to take `heap_drawIdx: u32, instId: u32` as additional plain u32 fn
 * parameters (after the existing FsIn struct param). The body uses
 * `heap_drawIdx` / `instId` as `Var` reads — the wrapper passes them
 * in from flat-interpolated u32 inputs on the family FsIn.
 *
 * `threadMegacallParamsThroughHelpers` is run as well in case any FS
 * helper fn extracted by composeStages references those identifiers.
 */
function applyFamilyMemberFsShape(fs: string): string {
  let s = fs;
  const startRe = /@fragment\s+fn\s+(\w+)\s*\(/;
  const startMatch = s.match(startRe);
  if (startMatch === null) return s;
  const fnName = startMatch[1]!;
  const paramOpen = startMatch.index! + startMatch[0]!.length;
  let depth = 1;
  let i = paramOpen;
  for (; i < s.length && depth > 0; i++) {
    const c = s[i];
    if (c === "(") depth++;
    else if (c === ")") depth--;
  }
  if (depth !== 0) return s;
  const paramList = s.slice(paramOpen, i - 1);
  const tailRe = /\s*->\s*([\w<>,\s]+?)\s*\{/y;
  tailRe.lastIndex = i;
  const tailMatch = tailRe.exec(s);
  if (tailMatch === null) return s;
  const retType = tailMatch[1]!.trim();
  const headerEnd = tailRe.lastIndex;
  const headerStart = startMatch.index!;
  const existing = paramList.split(",").map(p => p.trim()).filter(p => p.length > 0);
  const newParamList = [...existing, "heap_drawIdx: u32", "instId: u32"].join(", ");
  const newHeader = `@fragment fn ${fnName}(${newParamList}) -> ${retType} {\n`;
  s = s.slice(0, headerStart) + newHeader + s.slice(headerEnd);
  s = threadMegacallParamsThroughHelpers(s);
  return s;
}

/**
 * Post-emit text rewrite that threads `heap_drawIdx`, `instId`, and
 * `vid` through any helper function whose body references them. The
 * IR's composeStages pass extracts same-stage entries into helper fns
 * (`fn _<name>(s_in: <State>) -> <State>`) called from the wrapper
 * `@vertex` fn. Those helpers can carry CSE-extracted expressions that
 * reference `heap_drawIdx`/`instId`/`vid` — declared as locals in the
 * wrapper, those identifiers are out of scope inside helpers, so we
 * append them as `u32` parameters and pass them at every call site.
 *
 * Module-scope `var<private>` would also work but Safari/WebKit's WGSL
 * parser rejects helper-fn reads of module-scope private vars in some
 * configurations; explicit parameter passing is portable across all
 * conforming WGSL implementations.
 */
function threadMegacallParamsThroughHelpers(src: string): string {
  const idents = ["heap_drawIdx", "instId", "vid"] as const;
  // Find every `fn _<name>(...) -> <ret> { ... }` declaration. We only
  // touch fns whose name starts with `_` (the convention used by
  // extractFusedEntry's helpers); user code with a leading-underscore
  // name is not expected in IR-emitted WGSL.
  const fnRe = /\bfn\s+(_\w+)\s*\(/g;
  // Collect (helperName, paramRange, bodyRange, neededIdents).
  type Edit = { name: string; needed: string[]; paramOpen: number; paramClose: number };
  const edits: Edit[] = [];
  let m: RegExpExecArray | null;
  while ((m = fnRe.exec(src)) !== null) {
    const name = m[1]!;
    const paramOpen = m.index + m[0]!.length; // position after `(`
    // Balance parens to find the matching `)`.
    let depth = 1;
    let i = paramOpen;
    for (; i < src.length && depth > 0; i++) {
      const c = src[i];
      if (c === "(") depth++;
      else if (c === ")") depth--;
    }
    if (depth !== 0) continue;
    const paramClose = i - 1; // position of the matching `)`
    // Find the function body `{ ... }` after the return type.
    const afterParens = src.slice(i);
    const braceIdx = afterParens.indexOf("{");
    if (braceIdx < 0) continue;
    const bodyOpen = i + braceIdx;
    let bdepth = 1;
    let j = bodyOpen + 1;
    for (; j < src.length && bdepth > 0; j++) {
      const c = src[j];
      if (c === "{") bdepth++;
      else if (c === "}") bdepth--;
    }
    if (bdepth !== 0) continue;
    const bodyClose = j - 1;
    const body = src.slice(bodyOpen + 1, bodyClose);
    const needed = idents.filter(id => new RegExp(`\\b${id}\\b`).test(body));
    if (needed.length === 0) continue;
    edits.push({ name, needed, paramOpen, paramClose });
  }
  if (edits.length === 0) return src;
  // Apply param-list edits from the back so earlier offsets stay valid.
  const helperNeeds = new Map<string, string[]>();
  let out = src;
  for (let k = edits.length - 1; k >= 0; k--) {
    const e = edits[k]!;
    helperNeeds.set(e.name, e.needed);
    const existing = out.slice(e.paramOpen, e.paramClose).trim();
    const extra = e.needed.map(id => `${id}: u32`).join(", ");
    const newParams = existing.length === 0 ? extra : `${existing}, ${extra}`;
    out = out.slice(0, e.paramOpen) + newParams + out.slice(e.paramClose);
  }
  // Rewrite call sites for each touched helper. Match `<name>(<args>)`
  // and append the corresponding identifiers. Match the bare name so
  // we don't accidentally hit a substring inside another identifier.
  for (const [name, needed] of helperNeeds) {
    const callRe = new RegExp(`\\b${name}\\s*\\(`, "g");
    let result = "";
    let lastIdx = 0;
    let cm: RegExpExecArray | null;
    while ((cm = callRe.exec(out)) !== null) {
      const callOpen = cm.index + cm[0]!.length;
      // Skip if this is the `fn <name>(` declaration itself (preceded
      // by `fn`).
      const before = out.slice(0, cm.index).trimEnd();
      if (/\bfn$/.test(before)) continue;
      // Balance parens to find matching `)`.
      let depth = 1;
      let p = callOpen;
      for (; p < out.length && depth > 0; p++) {
        const c = out[p];
        if (c === "(") depth++;
        else if (c === ")") depth--;
      }
      if (depth !== 0) continue;
      const callClose = p - 1;
      const args = out.slice(callOpen, callClose).trim();
      const extras = needed.join(", ");
      const newArgs = args.length === 0 ? extras : `${args}, ${extras}`;
      result += out.slice(lastIdx, callOpen) + newArgs + ")";
      lastIdx = p; // p is one past the `)`
    }
    if (lastIdx > 0) {
      result += out.slice(lastIdx);
      out = result;
    }
  }
  return out;
}

function mergeStages(eff: Effect): Module {
  // Combine all stages' templates into one module by concatenating
  // their values + types arrays. The compile pipeline's composeStages
  // pass will fuse same-stage entries afterwards.
  const types = eff.stages.flatMap(s => [...s.template.types]);
  const values = eff.stages.flatMap(s => [...s.template.values]);
  return { types, values };
}
