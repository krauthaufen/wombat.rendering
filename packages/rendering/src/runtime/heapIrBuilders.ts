// Shared IR builders for the heap-rendering path.
//
// Centralises the tiny utility helpers (constU32 / item / add / …) plus
// the heap-load expressions (loadHeaderRef / loadUniformByRef /
// loadAttributeByRef / loadInstanceByRef). Both `heapEffectIR.ts` (the
// substitute-based legacy path) and `heapDecoder.ts` (the
// composition-based decoder synthesis) consume these.
//
// Storage buffers heapU32 / headersU32 / heapF32 / heapV4f are exposed
// as `ReadInput("Uniform", name)` of array<...> type; indexing is
// `Item(buffer, index)`. The buffers themselves are added as
// StorageBuffer ValueDefs (see `heapStorageBufferDecls`) so the WGSL
// emitter materialises the `@group(0) @binding(N) var<storage, read>`
// declarations.

import {
  type Expr, type Stmt, type Type, type ValueDef,
  Tu32, Tf32, Vec,
} from "@aardworx/wombat.shader/ir";

// ─── Type shorthands ───────────────────────────────────────────────────

export const Tbool: Type = { kind: "Bool" };
export const Tvec2 = Vec(Tf32, 2);
export const Tvec3 = Vec(Tf32, 3);
export const Tvec4 = Vec(Tf32, 4);
export const Tmat4 = { kind: "Matrix" as const, element: Tf32, rows: 4, cols: 4 } as Type;
export const TarrU32:  Type = { kind: "Array", element: Tu32, length: "runtime" } as Type;
export const TarrF32:  Type = { kind: "Array", element: Tf32, length: "runtime" } as Type;
export const TarrVec4: Type = { kind: "Array", element: Tvec4, length: "runtime" } as Type;

// ─── Const + leaf expression builders ─────────────────────────────────

export const constU32 = (n: number): Expr => ({
  kind: "Const", type: Tu32, value: { kind: "Int", signed: false, value: n },
});
export const constF32 = (n: number): Expr => ({
  kind: "Const", type: Tf32, value: { kind: "Float", value: n },
});
export const readScope = (
  scope: "Uniform" | "Input" | "Builtin",
  name: string,
  type: Type,
): Expr => ({ kind: "ReadInput", scope, name, type });

// ─── Compound expression builders ─────────────────────────────────────

export const item = (target: Expr, index: Expr, elemType: Type): Expr => ({
  kind: "Item", target, index, type: elemType,
});
export const add = (lhs: Expr, rhs: Expr, type: Type): Expr => ({ kind: "Add", lhs, rhs, type });
export const sub = (lhs: Expr, rhs: Expr, type: Type): Expr => ({ kind: "Sub", lhs, rhs, type });
export const mul = (lhs: Expr, rhs: Expr, type: Type): Expr => ({ kind: "Mul", lhs, rhs, type });
export const div = (lhs: Expr, rhs: Expr, type: Type): Expr => ({ kind: "Div", lhs, rhs, type });
export const mod = (lhs: Expr, rhs: Expr, type: Type): Expr => ({ kind: "Mod", lhs, rhs, type });
export const eqU32 = (lhs: Expr, rhs: Expr): Expr => ({ kind: "Eq", lhs, rhs, type: Tbool });
export const geU32 = (lhs: Expr, rhs: Expr): Expr => ({ kind: "Ge", lhs, rhs, type: Tbool });
export const leU32 = (lhs: Expr, rhs: Expr): Expr => ({ kind: "Le", lhs, rhs, type: Tbool });
export const shr = (lhs: Expr, rhs: Expr, type: Type): Expr => ({
  kind: "ShiftRight", lhs, rhs, type,
});
export const select = (cond: Expr, ifTrue: Expr, ifFalse: Expr, type: Type): Expr => ({
  kind: "Conditional", cond, ifTrue, ifFalse, type,
});
export const newVec = (components: Expr[], type: Type): Expr => ({
  kind: "NewVector", components, type,
});
export const matFromRows = (rows: Expr[], type: Type): Expr => ({
  kind: "MatrixFromRows", rows, type,
});

// ─── Storage-buffer references (Expr-form) ────────────────────────────

export const heapU32     = readScope("Uniform", "heapU32",     TarrU32);
export const headersU32  = readScope("Uniform", "headersU32",  TarrU32);
export const heapF32     = readScope("Uniform", "heapF32",     TarrF32);
export const heapV4f     = readScope("Uniform", "heapV4f",     TarrVec4);
export const drawTable        = readScope("Uniform", "drawTable",       TarrU32);
export const indexStorage     = readScope("Uniform", "indexStorage",    TarrU32);
export const firstDrawInTile  = readScope("Uniform", "firstDrawInTile", TarrU32);

// ─── Storage-buffer ValueDefs ─────────────────────────────────────────
//
// Bindings 0..3 are heap-arena typed views; 4..6 are megacall lookup
// buffers. Order matches the BindGroupLayout entries the runtime sets
// up — both sides walk module ValueDefs in declaration order and increment
// a slot counter to derive `@binding(N)`.

// `keep: true` on every heap-rendering binding: the host-side
// BindGroupLayout pins these at slots 0..6, so the WGSL must declare
// them at the same slots whether or not any code path on the current
// effect happens to reference each one (e.g. a shader with no mat4
// uniforms doesn't touch heapV4f, but the BGL still has slot 3).
export function heapArenaStorageDecls(): ValueDef[] {
  return [
    { kind: "StorageBuffer", binding: { group: 0, slot: 0 }, name: "heapU32",    layout: TarrU32,  access: "read", keep: true },
    { kind: "StorageBuffer", binding: { group: 0, slot: 1 }, name: "headersU32", layout: TarrU32,  access: "read", keep: true },
    { kind: "StorageBuffer", binding: { group: 0, slot: 2 }, name: "heapF32",    layout: TarrF32,  access: "read", keep: true },
    { kind: "StorageBuffer", binding: { group: 0, slot: 3 }, name: "heapV4f",    layout: TarrVec4, access: "read", keep: true },
  ];
}

export function megacallLookupStorageDecls(): ValueDef[] {
  return [
    { kind: "StorageBuffer", binding: { group: 0, slot: 4 }, name: "drawTable",       layout: TarrU32, access: "read", keep: true },
    { kind: "StorageBuffer", binding: { group: 0, slot: 5 }, name: "indexStorage",    layout: TarrU32, access: "read", keep: true },
    { kind: "StorageBuffer", binding: { group: 0, slot: 6 }, name: "firstDrawInTile", layout: TarrU32, access: "read", keep: true },
  ];
}

// ─── Heap-load address builders ───────────────────────────────────────

/** `headersU32[drawIdx * stride + offset]` — a u32 ref slot in the drawHeader. */
export function loadHeaderRef(drawIdx: Expr, fieldByteOffset: number, strideU32: number): Expr {
  const fieldU32 = fieldByteOffset / 4;
  if (!Number.isInteger(fieldU32)) {
    throw new Error(`loadHeaderRef: byteOffset ${fieldByteOffset} is not 4-byte aligned`);
  }
  const drawOff = mul(drawIdx, constU32(strideU32), Tu32);
  const idx = fieldU32 === 0 ? drawOff : add(drawOff, constU32(fieldU32), Tu32);
  return item(headersU32, idx, Tu32);
}

/**
 * Load a per-draw uniform value from the heap given its `u32` ref.
 * Matrix → 4 vec4 columns from heapV4f; vec4 → one heapV4f read;
 * vec3/2/scalar → heapF32 reads + NewVector.
 */
export function loadUniformByRef(refIdent: Expr, wgslType: string): Expr {
  switch (wgslType) {
    case "mat4x4<f32>": {
      const base = div(add(refIdent, constU32(16), Tu32), constU32(16), Tu32);
      const cols: Expr[] = [];
      for (let i = 0; i < 4; i++) {
        const idx = i === 0 ? base : add(base, constU32(i), Tu32);
        cols.push(item(heapV4f, idx, Tvec4));
      }
      return matFromRows(cols, Tmat4);
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
      throw new Error(`heap: no IR loader for uniform type '${wgslType}'`);
  }
}

/**
 * Cyclic per-vertex/per-instance attribute load. `idx` is the vertex or
 * instance index. The allocation header at `ref/4 + 1` holds the
 * element length so `idx % length` broadcasts length-1 allocations.
 */
export function loadAttributeByRef(refIdent: Expr, idx: Expr, wgslType: string): Expr {
  const dataF32Base = div(add(refIdent, constU32(16), Tu32), constU32(4), Tu32);
  const cyclic = (elemSize: number): Expr => {
    const length = item(heapU32, add(div(refIdent, constU32(4), Tu32), constU32(1), Tu32), Tu32);
    return mul(mod(idx, length, Tu32), constU32(elemSize), Tu32);
  };
  switch (wgslType) {
    case "vec3<f32>": {
      const base = add(dataF32Base, cyclic(3), Tu32);
      return newVec([
        item(heapF32, base, Tf32),
        item(heapF32, add(base, constU32(1), Tu32), Tf32),
        item(heapF32, add(base, constU32(2), Tu32), Tf32),
      ], Tvec3);
    }
    case "vec4<f32>": {
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
      const base = add(dataF32Base, cyclic(2), Tu32);
      return newVec([
        item(heapF32, base, Tf32),
        item(heapF32, add(base, constU32(1), Tu32), Tf32),
      ], Tvec2);
    }
    case "f32": {
      const base = add(dataF32Base, cyclic(1), Tu32);
      return item(heapF32, base, Tf32);
    }
    case "u32": {
      const base = add(div(refIdent, constU32(4), Tu32), add(constU32(4), cyclic(1), Tu32), Tu32);
      return item(heapU32, base, Tu32);
    }
    default:
      throw new Error(`heap: no IR loader for attribute type '${wgslType}'`);
  }
}

/**
 * Per-instance uniform/attribute load: read element `iidx` from the
 * allocation (no cyclic-modulo wrap — instance count is exact).
 */
export function loadInstanceByRef(refIdent: Expr, iidx: Expr, wgslType: string): Expr {
  const dataF32Base = div(add(refIdent, constU32(16), Tu32), constU32(4), Tu32);
  const dataVec4Base = div(add(refIdent, constU32(16), Tu32), constU32(16), Tu32);
  switch (wgslType) {
    case "mat4x4<f32>": {
      const off = mul(iidx, constU32(4), Tu32);
      const base = add(dataVec4Base, off, Tu32);
      const cols: Expr[] = [];
      for (let i = 0; i < 4; i++) {
        cols.push(item(heapV4f, add(base, constU32(i), Tu32), Tvec4));
      }
      return matFromRows(cols, Tmat4);
    }
    case "vec4<f32>":
      return item(heapV4f, add(dataVec4Base, iidx, Tu32), Tvec4);
    case "vec3<f32>": {
      const off = mul(iidx, constU32(3), Tu32);
      const base = add(dataF32Base, off, Tu32);
      return newVec([
        item(heapF32, base, Tf32),
        item(heapF32, add(base, constU32(1), Tu32), Tf32),
        item(heapF32, add(base, constU32(2), Tu32), Tf32),
      ], Tvec3);
    }
    case "vec2<f32>": {
      const off = mul(iidx, constU32(2), Tu32);
      const base = add(dataF32Base, off, Tu32);
      return newVec([
        item(heapF32, base, Tf32),
        item(heapF32, add(base, constU32(1), Tu32), Tf32),
      ], Tvec2);
    }
    case "f32":
      return item(heapF32, add(dataF32Base, iidx, Tu32), Tf32);
    case "u32":
      return item(heapU32, add(div(add(refIdent, constU32(16), Tu32), constU32(4), Tu32), iidx, Tu32), Tu32);
    default:
      throw new Error(`heap: no IR loader for per-instance type '${wgslType}'`);
  }
}

// ─── Megacall search prelude (IR-level) ───────────────────────────────
//
// Reproduces the binary-search prelude (currently emitted as raw WGSL by
// `megacallSearchPrelude()` in heapEffect.ts) as a sequence of IR Stmts.
// Returns:
//   - the Stmt list to prepend at the top of the entry body
//   - Var handles for `heap_drawIdx`, `instId`, `vid` that body code can
//     reference via `{kind: "Var", var: ...}` expressions
//
// The bit-tile cadence (tile = 64 emits) matches the prefix-sum scan
// kernel's tile size; if that changes upstream this MUST change too.

import type { Var } from "@aardworx/wombat.shader/ir";

export interface MegacallLocals {
  readonly heapDrawIdx: Var;
  readonly instId: Var;
  readonly vid: Var;
}

const TILE_BITS = 6; // 64 emits per tile — must match the scan kernel.
const RECORD_U32 = 5; // drawTable record stride — must match heapScene.

export function buildMegacallPrelude(emitIdx: Expr): { stmts: Stmt[]; locals: MegacallLocals } {
  // Local Var declarations. `mutable: false` for the result locals;
  // `mutable: true` for the lo/hi loop indices.
  const tileIdx: Var       = { name: "_tileIdx",  type: Tu32, mutable: false };
  const lo: Var            = { name: "lo",        type: Tu32, mutable: true  };
  const hi: Var            = { name: "hi",        type: Tu32, mutable: true  };
  const mid: Var           = { name: "_mid",      type: Tu32, mutable: false };
  const slot: Var          = { name: "_slot",     type: Tu32, mutable: false };
  const firstEmit: Var     = { name: "_firstEmit", type: Tu32, mutable: false };
  const heapDrawIdx: Var   = { name: "heap_drawIdx", type: Tu32, mutable: false };
  const indexStart: Var    = { name: "_indexStart", type: Tu32, mutable: false };
  const indexCount: Var    = { name: "_indexCount", type: Tu32, mutable: false };
  const localOff: Var      = { name: "_local",    type: Tu32, mutable: false };
  const instId: Var        = { name: "instId",    type: Tu32, mutable: false };
  const vid: Var           = { name: "vid",       type: Tu32, mutable: false };

  const varExpr = (v: Var): Expr => ({ kind: "Var", var: v, type: v.type });

  const stmts: Stmt[] = [];
  // let _tileIdx = emitIdx >> 6u;
  stmts.push({ kind: "Declare", var: tileIdx, init: { kind: "Expr", value: shr(emitIdx, constU32(TILE_BITS), Tu32) } });
  // var lo: u32 = firstDrawInTile[_tileIdx];
  stmts.push({ kind: "Declare", var: lo, init: { kind: "Expr", value: item(firstDrawInTile, varExpr(tileIdx), Tu32) } });
  // var hi: u32 = firstDrawInTile[_tileIdx + 1u];
  stmts.push({ kind: "Declare", var: hi, init: { kind: "Expr", value: item(firstDrawInTile, add(varExpr(tileIdx), constU32(1), Tu32), Tu32) } });

  // loop {
  //   if (lo >= hi) { break; }
  //   let _mid = (lo + hi + 1u) >> 1u;
  //   if (drawTable[_mid * 5u] <= emitIdx) { lo = _mid; } else { hi = _mid - 1u; }
  // }
  const loopBody: Stmt[] = [];
  loopBody.push({
    kind: "If",
    cond: geU32(varExpr(lo), varExpr(hi)),
    then: { kind: "Break" },
  });
  // (lo + hi + 1) >> 1 — half-up midpoint
  const midInit = shr(
    add(add(varExpr(lo), varExpr(hi), Tu32), constU32(1), Tu32),
    constU32(1),
    Tu32,
  );
  loopBody.push({ kind: "Declare", var: mid, init: { kind: "Expr", value: midInit } });
  // drawTable[_mid * 5u]
  const midDrawStart = item(drawTable, mul(varExpr(mid), constU32(RECORD_U32), Tu32), Tu32);
  loopBody.push({
    kind: "If",
    cond: leU32(midDrawStart, emitIdx),
    then: { kind: "Write", target: { kind: "LVar", var: lo, type: Tu32 }, value: varExpr(mid) },
    else: { kind: "Write", target: { kind: "LVar", var: hi, type: Tu32 }, value: sub(varExpr(mid), constU32(1), Tu32) },
  });
  stmts.push({ kind: "Loop", body: { kind: "Sequential", body: loopBody } });

  // let _slot = lo;
  stmts.push({ kind: "Declare", var: slot, init: { kind: "Expr", value: varExpr(lo) } });
  const slotBase = mul(varExpr(slot), constU32(RECORD_U32), Tu32);
  // let _firstEmit  = drawTable[_slot * 5u + 0u];
  stmts.push({ kind: "Declare", var: firstEmit, init: { kind: "Expr", value: item(drawTable, slotBase, Tu32) } });
  // let heap_drawIdx = drawTable[_slot * 5u + 1u];
  stmts.push({ kind: "Declare", var: heapDrawIdx, init: { kind: "Expr", value: item(drawTable, add(slotBase, constU32(1), Tu32), Tu32) } });
  // let _indexStart = drawTable[_slot * 5u + 2u];
  stmts.push({ kind: "Declare", var: indexStart, init: { kind: "Expr", value: item(drawTable, add(slotBase, constU32(2), Tu32), Tu32) } });
  // let _indexCount = drawTable[_slot * 5u + 3u];
  stmts.push({ kind: "Declare", var: indexCount, init: { kind: "Expr", value: item(drawTable, add(slotBase, constU32(3), Tu32), Tu32) } });
  // let _local = emitIdx - _firstEmit;
  stmts.push({ kind: "Declare", var: localOff, init: { kind: "Expr", value: sub(emitIdx, varExpr(firstEmit), Tu32) } });
  // let instId = _local / _indexCount;
  stmts.push({ kind: "Declare", var: instId, init: { kind: "Expr", value: div(varExpr(localOff), varExpr(indexCount), Tu32) } });
  // let vid = indexStorage[_indexStart + (_local % _indexCount)];
  stmts.push({
    kind: "Declare",
    var: vid,
    init: { kind: "Expr", value: item(indexStorage, add(varExpr(indexStart), mod(varExpr(localOff), varExpr(indexCount), Tu32), Tu32), Tu32) },
  });

  return { stmts, locals: { heapDrawIdx, instId, vid } };
}
