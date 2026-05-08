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
import { substituteInputs, readInputs } from "@aardworx/wombat.shader/passes";
import {
  type Module, type Expr, type Stmt, type Type, type ValueDef,
  type EntryDef, type EntryParameter, type ParamDecoration,
  Tu32, Tf32, Vec,
} from "@aardworx/wombat.shader/ir";
import type { BucketLayout } from "./heapEffect.js";

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
    } else {
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
  combined = injectVsBuiltins(combined);
  // FS uniform threading must run BEFORE VS body rewriting, since it
  // adds WriteOutput stmts that the VS rewriter shouldn't touch (the
  // refs they write reference `instance_index` directly, not the
  // substituted heap-load expressions).
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
  return {
    vs: vsStage?.source ?? "",
    fs: fsStage?.source ?? "",
    vsEntry: vsStage?.entryName ?? "vs",
    fsEntry: fsStage?.entryName ?? "fs",
    preludeWgsl: "",
  };
}

function mergeStages(eff: Effect): Module {
  // Combine all stages' templates into one module by concatenating
  // their values + types arrays. The compile pipeline's composeStages
  // pass will fuse same-stage entries afterwards.
  const types = eff.stages.flatMap(s => [...s.template.types]);
  const values = eff.stages.flatMap(s => [...s.template.values]);
  return { types, values };
}
