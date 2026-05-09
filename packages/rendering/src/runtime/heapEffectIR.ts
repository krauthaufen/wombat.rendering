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
import { substituteInputs, readInputs, mapExpr, mapStmt, liftReturns } from "@aardworx/wombat.shader/passes";
import {
  type Module, type Expr, type Stmt, type Type, type ValueDef,
  type EntryDef, type EntryParameter, type ParamDecoration,
  Tu32, Tf32, Vec,
} from "@aardworx/wombat.shader/ir";
import {
  megacallSearchPrelude, atlasVaryingNames,
  ATLAS_BINDING_LINEAR, ATLAS_BINDING_SRGB, ATLAS_BINDING_SAMPLER,
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
 * Load a per-vertex attribute value with cyclic addressing. For vec4
 * the source can be V3-tight (12 B, .w = 1.0) or V4-tight (16 B, .w
 * from data); we read the actual stride from the alloc header at
 * offset 8, then use `select` for the .w fill.
 */
function loadAttributeByRef(refIdent: Expr, vid: Expr, wgslType: string): Expr {
  const dataF32Base = div(add(refIdent, constU32(16), Tu32), constU32(4), Tu32);
  switch (wgslType) {
    case "vec3<f32>": {
      const off = cyclicElemIdx(refIdent, vid, 3);
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
      const cycled = mod(vid, item(heapU32, add(div(refIdent, constU32(4), Tu32), constU32(1), Tu32), Tu32), Tu32);
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
      const off = cyclicElemIdx(refIdent, vid, 2);
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

function rewriteVertexBodies(m: Module, layout: BucketLayout): Module {
  // For instanced buckets the bucket holds a single slot — drawIdx is
  // baked to 0u and `instance_index` becomes iidx (per-instance index
  // within the array uniform). For non-instanced buckets drawIdx ==
  // instance_index (firstInstance trick) and iidx == 0u.
  const drawIdx = layout.isInstanced
    ? constU32(0)
    : readScope("Builtin", "instance_index", Tu32);
  const iidx = layout.isInstanced
    ? readScope("Builtin", "instance_index", Tu32)
    : constU32(0);
  const vid = readScope("Builtin", "vertex_index", Tu32);
  const uniformMapping = new Map<string, Expr>();
  const attrMapping    = new Map<string, Expr>();
  const stride = layout.strideU32;
  for (const f of layout.drawHeaderFields) {
    const refExpr = loadHeaderRef(drawIdx, f.byteOffset, stride);
    if (f.kind === "uniform-ref") {
      const value = layout.perInstanceUniforms.has(f.name)
        ? loadInstanceByRef(refExpr, iidx, f.uniformWgslType ?? "")
        : loadUniformByRef(refExpr, f.uniformWgslType ?? "");
      uniformMapping.set(f.name, value);
    } else if (f.kind === "attribute-ref") {
      attrMapping.set(f.name, loadAttributeByRef(refExpr, vid, f.attributeWgslType ?? ""));
    }
  }
  // Restrict to vertex stages: the FS substitution needs the threaded
  // varyings, not raw heap loads. We pre-filter by walking values
  // and only substituting inside vertex Entry bodies.
  return mapVertexEntryBodies(m, body => {
    let s = body;
    s = mapStmtSubstInputScope(s, "Uniform", n => uniformMapping.get(n));
    s = mapStmtSubstInputScope(s, "Input",   n => attrMapping.get(n));
    return s;
  });
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

function mapStmtSubstInputScope(
  body: Stmt,
  scope: "Uniform" | "Input" | "Builtin",
  mapping: (name: string) => Expr | undefined,
): Stmt {
  // Wrap a single-stage substitution in a synthetic single-entry
  // module so we can reuse `substituteInputs` from wombat.shader.
  const tempEntry: EntryDef = {
    name: "__tmp", stage: "vertex", inputs: [], outputs: [], arguments: [],
    returnType: { kind: "Void" } as Type, body, decorations: [],
  };
  const tempModule: Module = {
    types: [], values: [{ kind: "Entry", entry: tempEntry }],
  };
  const out = substituteInputs(tempModule, scope, mapping);
  const ent = out.values.find(v => v.kind === "Entry");
  if (ent === undefined || ent.kind !== "Entry") return body;
  return ent.entry.body;
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
  const drawIdxExpr = readScope("Builtin", "instance_index", Tu32);
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
  out = mapStageEntryBodies(out, "fragment", body =>
    mapStmtSubstInputScope(body, "Uniform", n => fsSubst.get(n)));
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
  if (e.args.length < 3) return null;
  const tex = e.args[0]!;
  if (tex.kind !== "ReadInput" || tex.scope !== "Uniform") return null;
  if (!atlasNames.has(tex.name)) return null;
  return { name: tex.name, uv: e.args[2]! };
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

  const drawIdxExpr = readScope("Builtin", "instance_index", Tu32);
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
export function compileHeapEffectIR(
  userEffect: Effect,
  layout: BucketLayout,
  compileOptions: CompileOptions,
): { vs: string; fs: string; preludeWgsl: string; vsEntry: string; fsEntry: string } {
  // Build a single Module from all of the effect's stages by
  // composing them. Then apply heap rewrites to that combined module.
  // The composed module preserves stage entry boundaries.
  let combined: Module = mergeStages(userEffect);
  // Lift `return { ... }` into explicit WriteOutput stmts so the
  // substitution passes below see (and rewrite) the ReadInputs that
  // feed the record fields. Without this, ObjectLiteral exprs hide
  // their fields on a `_record` carrier that mapExpr doesn't traverse.
  combined = liftReturns(combined);
  combined = injectVsBuiltins(combined);
  // FS uniform threading must run BEFORE VS body rewriting, since it
  // adds WriteOutput stmts that the VS rewriter shouldn't touch (the
  // refs they write reference `instance_index` directly, not the
  // substituted heap-load expressions).
  combined = rewriteFsAtlasTextures(combined, layout);
  combined = rewriteFsUniforms(combined, layout);
  combined = rewriteVertexBodies(combined, layout);
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
  if (layout.megacall && vsSrc.length > 0) {
    vsSrc = applyMegacallToEmittedVs(vsSrc);
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
  const decls = `
@group(0) @binding(${ATLAS_BINDING_LINEAR})  var atlasLinear:  texture_2d_array<f32>;
@group(0) @binding(${ATLAS_BINDING_SRGB})    var atlasSrgb:    texture_2d_array<f32>;
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
fn atlasSampleAtMip(pageRef: u32, format: u32, origin: vec2<f32>, size: vec2<f32>, k: u32, uvW: vec2<f32>) -> vec4<f32> {
  let mipSize = size / pow(2.0, f32(k));
  let mipO = atlasMipOrigin(origin, size, k);
  let atlasUv = mipO + uvW * mipSize;
  let lin = textureSampleLevel(atlasLinear, atlasSampler, atlasUv, pageRef, 0.0);
  let sr  = textureSampleLevel(atlasSrgb,   atlasSampler, atlasUv, pageRef, 0.0);
  return select(lin, sr, format == 1u);
}
fn atlasSample(pageRef: u32, formatBits: u32, origin: vec2<f32>, size: vec2<f32>, uv: vec2<f32>) -> vec4<f32> {
  let format  = formatBits & 0x1u;
  let numMips = (formatBits >> 1u) & 0x7u;
  let addrU   = (formatBits >> 4u) & 0x3u;
  let addrV   = (formatBits >> 6u) & 0x3u;
  let uvW = atlasApplyWrap(uv, addrU, addrV);
  if (numMips <= 1u) { return atlasSampleAtMip(pageRef, format, origin, size, 0u, uvW); }
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
  return decls + fs;
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
  const newHeader = `@vertex fn ${fnName}(${kept.join(", ")}) -> ${retType} {\n${megacallSearchPrelude()}  let instance_index: u32 = drawIdx;\n  let vertex_index: u32 = vid;\n`;
  s = s.slice(0, headerStart) + newHeader + s.slice(headerEnd);
  return decl + s;
}

function mergeStages(eff: Effect): Module {
  // Combine all stages' templates into one module by concatenating
  // their values + types arrays. The compile pipeline's composeStages
  // pass will fuse same-stage entries afterwards.
  const types = eff.stages.flatMap(s => [...s.template.types]);
  const values = eff.stages.flatMap(s => [...s.template.values]);
  return { types, values };
}
