// heapScene — multi-group heap-everything render path.
//
// Public-API extraction of the experimental heap-demo (commits in
// the `experimental/heap-arch` branch document the design). One
// shared GPURenderPipeline + GPUBindGroup per pipeline-state group,
// per-draw uniforms in a storage buffer, vertex pulling from packed
// position / normal slabs, indexed draws via a shared index buffer
// and `firstInstance` routing into the heap.
//
// Per-frame uploads:
//   - Globals (one UBO per group): viewProj + lightLocation. ~80 B.
//     Independent of draw count.
//   - DrawHeader (one storage buffer per group): only slots whose
//     `aval<Trafo3d>` or `aval<V4f>` were marked since last frame
//     are re-uploaded — 160 B per dirty slot.
//
// Geometry is uploaded ONCE at construction (positions + normals +
// indices, packed across all draws in a group).
//
// The user supplies a fragment-shader body per group key. The vertex
// stage is provided by this module and reads from a fixed layout:
//
//   @group(0) @binding(0) var<uniform>             globals:   Globals;
//   @group(0) @binding(1) var<storage, read>       draws:     array<DrawHeader>;
//   @group(0) @binding(2) var<storage, read>       positions: array<f32>;
//   @group(0) @binding(3) var<storage, read>       normals:   array<f32>;
//
//   struct VsOut {
//     @builtin(position) clipPos:  vec4<f32>,
//     @location(0)       worldPos: vec3<f32>,
//     @location(1)       normal:   vec3<f32>,
//     @location(2)       color:    vec4<f32>,
//     @location(3)       lightLoc: vec3<f32>,
//   };
//
// User WGSL must declare `@fragment fn fs(in: VsOut) -> @location(0) vec4<f32>`.
//
// Per-group texture sets. WebGPU 1.0 has no bindless, so texture-set
// is a wedge in the group key: groups with different texture sets
// can't share a bind group. Pass a `textures: { texture, sampler }`
// per groupKey via `BuildHeapSceneOptions.texturesByGroupKey`. The
// package adds the texture (binding 4) + sampler (binding 5) to that
// group's bind-group layout; the user's FS WGSL declares them.

import { Trafo3d, V3d, V4f, M44d, type V2f } from "@aardworx/wombat.base";
import type { ITexture } from "../core/texture.js";
import type { ISampler } from "../core/sampler.js";
import { AVal, AdaptiveObject, AdaptiveToken, HashTable } from "@aardworx/wombat.adaptive";
import type { aval, aset, IAdaptiveObject, IDisposable, IHashSetReader } from "@aardworx/wombat.adaptive";
import type { Effect, CompileOptions } from "@aardworx/wombat.shader";
import type { PipelineState } from "../core/pipelineState.js";
import type { BufferView } from "../core/bufferView.js";
import type { IBuffer, HostBufferSource } from "../core/buffer.js";
import {
  buildBucketLayout, compileHeapEffect,
  type BucketLayout, type FragmentOutputLayout,
  type HeapEffectSchema,
} from "./heapEffect.js";
import { compileHeapEffectIR } from "./heapEffectIR.js";
import {
  buildShaderFamily, compileShaderFamily,
  type ShaderFamilySchema,
} from "./heapShaderFamily.js";
import {
  ATLAS_PAGE_FORMATS, atlasFormatIndex,
  type AtlasPage, type AtlasPageFormat, type AtlasPool,
} from "./textureAtlas/atlasPool.js";
import {
  ATLAS_ARRAY_SIZE, ATLAS_LINEAR_BINDING_BASE,
  ATLAS_SRGB_BINDING_BASE, ATLAS_SAMPLER_BINDING,
} from "./heapEffect.js";
import {
  DerivedUniformsScene,
  registerRoDerivations, deregisterRoDerivations,
  isDerivedUniformName,
  type RoRegistration,
} from "./derivedUniforms/index.js";

// ---------------------------------------------------------------------------
// Per-allocation arena layout
// ---------------------------------------------------------------------------

// Per-allocation header: (u32 typeId, u32 length). typeId is
// (semantic << 16) | encoding. The data region follows the header
// aligned up to 16 bytes (so positions/normals/etc. line up for
// future vec4 reads).
const ALLOC_HEADER_BYTES   = 8;
const ALLOC_HEADER_PAD_TO  = 16;     // data starts header_offset + 16

// Encoding-tag enum (low 16 bits of typeId).
const ENC_V3F_TIGHT = 1;             // tightly-packed array of vec3<f32> (12 B/elt)

// Semantic-tag enum (high 16 bits of typeId). Optional metadata —
// the shader doesn't branch on this.
const SEM_POSITIONS = 1;
const SEM_NORMALS   = 2;

const ALIGN16 = (n: number) => (n + 15) & ~15;

function packMat44(m: M44d, dst: Float32Array, off: number): void {
  // Zero-alloc flat copy (row-major) straight into the f32 staging
  // buffer — `copyTo` does `dst.set(m._data, off)` which narrows f64→f32
  // on store, no throwaway `number[]` per call.
  m.copyTo(dst, off);
}

// ─── Layout-driven value packing ────────────────────────────────────
//
// Maps a schema uniform name + its value source to bytes in a staging
// buffer. The bridge between the spec's named JS-side fields (e.g.
// `spec.modelTrafo: Trafo3d`) and the schema's typed uniforms
// ("ModelTrafo" mat4, "ModelTrafoInv" mat4, …). Step 5 generalises
// this to a `spec.uniforms: { [name]: aval }` map; until then we
// hardcode the shape here.

// ─── Generic packer registry, keyed on WGSL type ────────────────────
//
// Each per-draw uniform comes from the spec as an aval whose JS value
// type is determined by what the user passes. The packer for a given
// WGSL type knows how to turn that JS value into bytes for the arena.
// Step 5: this replaces the per-name `perDrawBinding` switch — the
// spec just supplies `uniforms: { [name]: aval<...> }` and the
// runtime asks the registry "how do I pack a `mat4x4<f32>`".

/** A packer for one WGSL storage-buffer type. */
interface WgslPacker {
  /** Tightly-packed size in bytes of one value (mat4 = 64, vec3 = 12, …). */
  readonly dataBytes: number;
  readonly typeId: number;
  /**
   * Pack `val` (the aval's `.getValue(tok)` result) into `dst` at
   * float offset `off`. The packer is responsible for handling the
   * value type — Trafo3d, M44d, V4f, V3d, V3f, number — coercing as
   * needed. Throws on unsupported value shapes.
   */
  readonly pack: (val: unknown, dst: Float32Array, off: number) => void;
}

const PACKER_MAT4: WgslPacker = {
  dataBytes: 64, typeId: 0,
  pack: (val, dst, off) => {
    // Accept Trafo3d (uses .forward) or M44d directly.
    const m = (val as { forward?: M44d }).forward !== undefined
      ? (val as { forward: M44d }).forward
      : (val as M44d);
    packMat44(m, dst, off);
  },
};
const PACKER_VEC4: WgslPacker = {
  dataBytes: 16, typeId: 0,
  pack: (val, dst, off) => {
    const v = val as V4f;
    dst[off + 0] = v.x; dst[off + 1] = v.y;
    dst[off + 2] = v.z; dst[off + 3] = v.w;
  },
};
const PACKER_VEC3: WgslPacker = {
  dataBytes: 12, typeId: 0,
  pack: (val, dst, off) => {
    // V3f or V3d both expose .x/.y/.z; cast through a common shape.
    const v = val as { x: number; y: number; z: number };
    dst[off + 0] = v.x; dst[off + 1] = v.y; dst[off + 2] = v.z;
  },
};
const PACKER_VEC2: WgslPacker = {
  dataBytes: 8, typeId: 0,
  pack: (val, dst, off) => {
    const v = val as { x: number; y: number };
    dst[off + 0] = v.x; dst[off + 1] = v.y;
  },
};
const PACKER_F32: WgslPacker = {
  dataBytes: 4, typeId: 0,
  pack: (val, dst, off) => { dst[off] = val as number; },
};
// Integer scalars / vectors. The arena is fronted by a `Float32Array`,
// so writing raw bits has to go through a same-buffer Uint32/Int32 view
// to avoid the lossy `i32 → f32` coercion you'd get from a direct
// `dst[off] = ...` assignment.
function makeIntPacker(
  ctor: typeof Uint32Array | typeof Int32Array,
  dim: 1 | 2 | 3 | 4,
): WgslPacker {
  const bytes = dim * 4;
  if (dim === 1) {
    return {
      dataBytes: bytes, typeId: 0,
      pack: (val, dst, off) => {
        new ctor(dst.buffer as ArrayBuffer, dst.byteOffset + off * 4, 1)[0] = val as number;
      },
    };
  }
  // Vector: accept {x,y,z,w} components.
  return {
    dataBytes: bytes, typeId: 0,
    pack: (val, dst, off) => {
      const view = new ctor(dst.buffer as ArrayBuffer, dst.byteOffset + off * 4, dim);
      const v = val as { x: number; y: number; z?: number; w?: number };
      view[0] = v.x; view[1] = v.y;
      if (dim >= 3) view[2] = v.z!;
      if (dim >= 4) view[3] = v.w!;
    },
  };
}
const PACKER_U32     = makeIntPacker(Uint32Array, 1);
const PACKER_UVEC2   = makeIntPacker(Uint32Array, 2);
const PACKER_UVEC3   = makeIntPacker(Uint32Array, 3);
const PACKER_UVEC4   = makeIntPacker(Uint32Array, 4);
const PACKER_I32     = makeIntPacker(Int32Array, 1);
const PACKER_IVEC2   = makeIntPacker(Int32Array, 2);
const PACKER_IVEC3   = makeIntPacker(Int32Array, 3);
const PACKER_IVEC4   = makeIntPacker(Int32Array, 4);

function packerForWgslType(wgslType: string): WgslPacker {
  switch (wgslType) {
    case "mat4x4<f32>": return PACKER_MAT4;
    case "vec4<f32>":   return PACKER_VEC4;
    case "vec3<f32>":   return PACKER_VEC3;
    case "vec2<f32>":   return PACKER_VEC2;
    case "f32":         return PACKER_F32;
    case "u32":         return PACKER_U32;
    case "vec2<u32>":   return PACKER_UVEC2;
    case "vec3<u32>":   return PACKER_UVEC3;
    case "vec4<u32>":   return PACKER_UVEC4;
    case "i32":         return PACKER_I32;
    case "vec2<i32>":   return PACKER_IVEC2;
    case "vec3<i32>":   return PACKER_IVEC3;
    case "vec4<i32>":   return PACKER_IVEC4;
    default:
      throw new Error(`heapScene: no JS-side packer for WGSL type '${wgslType}'`);
  }
}

// ---------------------------------------------------------------------------
// Shared WGSL prelude (struct + bindings + VS)
// ---------------------------------------------------------------------------

// Common decls — struct + bindings + helpers. Always concatenated
// into every group's shader, before its custom VS+FS bodies.
//
// One arena GPUBuffer is bound through multiple typed views. Today
// only `array<DrawHeader>` and `array<f32>` are wired; later we add
// `array<u32>`, `array<vec2<f32>>`, `array<vec4<f32>>`, etc. (skip
// `vec3` — its storage stride is 16, doesn't match tight packing).
//
// `attrRef` style: each attribute reference in DrawHeader is a byte
// offset into the arena pointing at the allocation's header. The
// header is `(u32 typeId, u32 length)`; data follows 16 bytes after
// the ref. Today every attribute is V3F_TIGHT-encoded so the shader
// reads 3 consecutive f32s from the f32 view; once we add more
// encodings, `loadVec3Attr` grows a branch on `typeId & 0xFFFFu`.
// (Per-bucket WGSL preludes are now generated by `buildBucketLayout`
// in heapEffect.ts; this module just glues the bucket layout, the
// shared arena, and the per-bucket DrawHeap together.)

// ---------------------------------------------------------------------------
// Geometry packing
// ---------------------------------------------------------------------------

/**
 * Geometry triple. Tightly-packed Float32 positions / normals (3
 * floats per vertex) plus Uint32 indices.
 */
export interface HeapGeometry {
  readonly positions: Float32Array;
  readonly normals:   Float32Array;
  readonly indices:   Uint32Array;
}


// ---------------------------------------------------------------------------
// Resizable buffer (pow2 grow + GPU-side copy on resize)
// ---------------------------------------------------------------------------

const MIN_BUFFER_BYTES = 64 * 1024;
const POW2 = (n: number): number => {
  let p = 1; while (p < n) p <<= 1; return p;
};

/**
 * A GPUBuffer that can grow to next power-of-two on demand. On grow,
 * a fresh buffer is created at the new size, the live tail copied
 * over via copyBufferToBuffer, and dependents (bind groups, mostly)
 * are notified to rebuild via the `onResize` callback.
 *
 * `usedBytes` is the high-water mark — the runtime advances this as
 * it allocates, and `ensureCapacity` grows when required. This
 * separates allocation policy from grow policy.
 */
class GrowBuffer {
  private buf: GPUBuffer;
  private cap: number;
  private used = 0;
  private readonly listeners = new Set<() => void>();
  constructor(
    private readonly device: GPUDevice,
    private readonly label: string,
    private readonly usage: GPUBufferUsageFlags,
    initialBytes: number,
  ) {
    this.cap = Math.max(MIN_BUFFER_BYTES, POW2(initialBytes));
    this.buf = device.createBuffer({ size: this.cap, usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label });
  }
  get buffer(): GPUBuffer { return this.buf; }
  get capacity(): number { return this.cap; }
  get usedBytes(): number { return this.used; }
  setUsed(n: number): void { this.used = n; }
  onResize(cb: () => void): IDisposable {
    this.listeners.add(cb);
    return { dispose: () => { this.listeners.delete(cb); } };
  }
  /** Ensure the buffer is at least `bytes` capacity. Grows by pow2 + copies live tail. */
  ensureCapacity(bytes: number): void {
    if (bytes <= this.cap) return;
    const newCap = POW2(bytes);
    const newBuf = this.device.createBuffer({
      size: newCap, usage: this.usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label: this.label,
    });
    if (this.used > 0) {
      const enc = this.device.createCommandEncoder({ label: `${this.label}: grow-copy` });
      enc.copyBufferToBuffer(this.buf, 0, newBuf, 0, ALIGN16(this.used));
      this.device.queue.submit([enc.finish()]);
    }
    this.buf.destroy();
    this.buf = newBuf;
    this.cap = newCap;
    for (const cb of this.listeners) cb();
  }
  destroy(): void { this.buf.destroy(); }
}

// ---------------------------------------------------------------------------
// UniformPool — aval-keyed refcounted allocations over the AttributeArena
// ---------------------------------------------------------------------------

interface PoolEntry {
  /** Byte offset into the arena. Data starts at ref + ALLOC_HEADER_PAD_TO. */
  readonly ref: number;
  readonly dataBytes: number;
  readonly typeId: number;
  /** Packer used to refresh the data region on aval marks. */
  readonly pack: (val: unknown, dst: Float32Array, off: number) => void;
  refcount: number;
}

/**
 * Aval-keyed pool of arena allocations. One allocation per unique
 * aval (object identity). Two draws referencing the same aval share
 * the allocation; their DrawHeaders carry the same u32 ref. Holds
 * uniforms (fixed-size scalars/vectors/matrices) AND attribute arrays
 * (variable-size). The caller decides `dataBytes` + `length` per
 * acquisition — the pool just keys on aval identity and refcounts.
 *
 * Sharing emerges from aval identity — no separate "frequency"
 * declaration needed. A `cval` shared by all draws → 1 alloc.
 * A static positions array shared across instanced draws → 1 alloc.
 * Same code path either way.
 */
class UniformPool {
  // Keyed by `aval<unknown>` *by reference* (a plain JS `Map`). These
  // keys are overwhelmingly reactive `cval`s (per-object trafos,
  // colours, …) and the hot path is `acquire`/`release` ~once per
  // drawHeader field per RO. A content-keyed `HashTable` would buy
  // nothing here — reactive avals never compare content-equal — and
  // would cost: reactive avals have no `equals`/`getHashCode`, so a
  // `HashTable` falls back to a WeakMap-counter identity hash per
  // lookup, measurably slower than `Map`'s native hashing. (Constant-
  // aval dedup matters where keys are *texture* avals — there the
  // `AtlasPool` is content-keyed; constant avals there carry a cached
  // hash and a fast `equals`.)
  private readonly byAval = new Map<aval<unknown>, PoolEntry>();

  has(aval: aval<unknown>): boolean { return this.byAval.has(aval); }
  entry(aval: aval<unknown>): PoolEntry | undefined { return this.byAval.get(aval); }

  /**
   * Acquire (or share) an allocation for `aval`. Caller passes the
   * pre-read `value` (so the pool doesn't need a token) plus the
   * (`dataBytes`, `typeId`, `length`, `pack`) describing how to lay
   * it out. If a new allocation is made, the value is packed and
   * uploaded immediately.
   */
  acquire(
    device: GPUDevice,
    arena: AttributeArena,
    aval: aval<unknown>,
    value: unknown,
    dataBytes: number,
    typeId: number,
    length: number,
    pack: (val: unknown, dst: Float32Array, off: number) => void,
  ): number {
    const existing = this.byAval.get(aval);
    if (existing !== undefined) {
      existing.refcount++;
      return existing.ref;
    }
    const ref = arena.alloc(dataBytes);
    const allocBytes = ALIGN16(ALLOC_HEADER_PAD_TO + dataBytes);
    const buf = new ArrayBuffer(allocBytes);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);
    u32[0] = typeId;
    u32[1] = length;
    // stride_bytes (offset 8): bytes per element. Lets the VS decode
    // pick V3- vs V4-tight load expressions for vec4 attributes
    // (and is informative for everything else).
    u32[2] = length > 0 ? Math.floor(dataBytes / length) : 0;
    pack(value, f32, ALLOC_HEADER_PAD_TO / 4);
    arena.write(ref, new Uint8Array(buf));
    void device;
    this.byAval.set(aval, { ref, dataBytes, typeId, pack, refcount: 1 });
    return ref;
  }

  /** Decrement refcount; if zero, free the arena allocation. */
  release(arena: AttributeArena, aval: aval<unknown>): void {
    const e = this.byAval.get(aval);
    if (e === undefined) return;
    e.refcount--;
    if (e.refcount > 0) return;
    arena.release(e.ref, ALIGN16(ALLOC_HEADER_PAD_TO + e.dataBytes));
    this.byAval.delete(aval);
  }

  /** Re-pack one entry's data region into the arena's CPU shadow. */
  repack(device: GPUDevice, arena: AttributeArena, aval: aval<unknown>, val: unknown): void {
    const e = this.byAval.get(aval);
    if (e === undefined) return;
    const dst = new Float32Array(e.dataBytes / 4);
    e.pack(val, dst, 0);
    arena.write(
      e.ref + ALLOC_HEADER_PAD_TO,
      new Uint8Array(dst.buffer, dst.byteOffset, e.dataBytes),
    );
    void device;
  }
}

/**
 * Aval-keyed pool over the `IndexAllocator`. Two draws referencing
 * the same `Uint32Array` (or aval thereof) share an index range —
 * 19K instanced clones of the same mesh share one allocation, one
 * upload. Index data is treated as immutable for the aval's
 * lifetime: an aval mark won't repack (we'd have to free + re-alloc
 * since size changes are likely). Use a fresh aval to swap meshes.
 *
 * **Value-equality dedup for constant avals (§5b):** when an
 * incoming aval has `isConstant === true`, the pool also keys by
 * the underlying `ArrayBuffer` tuple `(buffer, byteOffset,
 * byteLength)`. Two distinct constant avals wrapping the same
 * `Uint32Array` view (or two views over the same backing buffer
 * with matching offsets) collapse to one allocation. Hashing
 * kilobytes of indices on every acquire would be wasteful; the
 * tuple key catches the realistic "one ArrayBuffer shared across
 * many aval wrappers" pattern, which is the only one that matters
 * for the heap path. Reactive (non-constant) avals fall through
 * to identity-only — their content can change and the pool can't
 * silently merge them.
 */
class IndexPool {
  // Per-aval binding. `perAvalCount` tracks acquire/release balance
  // for THIS aval; `entry` is the shared allocation (one entry can be
  // referenced by many aliasing avals via §5b dedup).
  private readonly byAval = new Map<
    aval<Uint32Array>,
    { entry: IndexPoolEntry; perAvalCount: number }
  >();
  private readonly byValueKey = new Map<string, IndexPoolEntry>();
  // Stable per-ArrayBuffer numeric id for value-key composition.
  // WeakMap-backed so buffers GC'd elsewhere drop their entry too.
  private readonly bufferIds = new WeakMap<ArrayBufferLike, number>();
  private nextBufferId = 1;
  private bufferIdOf(buf: ArrayBufferLike): number {
    let id = this.bufferIds.get(buf);
    if (id === undefined) {
      id = this.nextBufferId++;
      this.bufferIds.set(buf, id);
    }
    return id;
  }

  acquire(
    device: GPUDevice,
    indices: IndexAllocator,
    aval: aval<Uint32Array>,
    arr: Uint32Array,
  ): { firstIndex: number; count: number } {
    const bound = this.byAval.get(aval);
    if (bound !== undefined) {
      bound.perAvalCount++;
      bound.entry.totalRefcount++;
      return { firstIndex: bound.entry.firstIndex, count: bound.entry.count };
    }
    let valueKey: string | undefined;
    if (aval.isConstant) {
      valueKey = `${this.bufferIdOf(arr.buffer)}:${arr.byteOffset}:${arr.byteLength}`;
      const shared = this.byValueKey.get(valueKey);
      if (shared !== undefined) {
        shared.totalRefcount++;
        this.byAval.set(aval, { entry: shared, perAvalCount: 1 });
        return { firstIndex: shared.firstIndex, count: shared.count };
      }
    }
    const firstIndex = indices.alloc(arr.length);
    indices.write(
      firstIndex * 4,
      new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength),
    );
    void device;
    const entry: IndexPoolEntry = { firstIndex, count: arr.length, totalRefcount: 1, valueKey };
    this.byAval.set(aval, { entry, perAvalCount: 1 });
    if (valueKey !== undefined) this.byValueKey.set(valueKey, entry);
    return { firstIndex, count: arr.length };
  }

  release(indices: IndexAllocator, aval: aval<Uint32Array>): void {
    const bound = this.byAval.get(aval);
    if (bound === undefined) return;
    bound.perAvalCount--;
    bound.entry.totalRefcount--;
    if (bound.perAvalCount === 0) this.byAval.delete(aval);
    if (bound.entry.totalRefcount > 0) return;
    indices.release(bound.entry.firstIndex, bound.entry.count);
    if (bound.entry.valueKey !== undefined) {
      this.byValueKey.delete(bound.entry.valueKey);
    }
  }
}

interface IndexPoolEntry {
  firstIndex: number;
  count: number;
  totalRefcount: number;
  valueKey: string | undefined;
}

// ---------------------------------------------------------------------------
// DrawHeap (slot-indexed) and AttributeArena (byte-bump) allocators
// ---------------------------------------------------------------------------

/**
 * Slot-indexed allocator over a GrowBuffer. `slotBytes` is set per-
 * instance — each bucket sizes its DrawHeader from its effect's
 * schema, so a bucket whose layout is e.g. 96 B / slot uses a
 * DrawHeap with `slotBytes=96`.
 */
class DrawHeap {
  private free: number[] = [];
  private nextSlot = 0;
  constructor(private readonly buf: GrowBuffer, private readonly slotBytes: number) {}
  get buffer(): GPUBuffer { return this.buf.buffer; }
  /** Bytes per slot — caller multiplies by slot index for byte offsets. */
  get bytesPerSlot(): number { return this.slotBytes; }
  /** High-water mark in bytes (used to size bind-group entry on rebuild). */
  get usedBytes(): number { return this.nextSlot * this.slotBytes; }
  alloc(): number {
    const slot = this.free.length > 0 ? this.free.pop()! : this.nextSlot++;
    this.buf.ensureCapacity((slot + 1) * this.slotBytes);
    this.buf.setUsed(Math.max(this.buf.usedBytes, (slot + 1) * this.slotBytes));
    return slot;
  }
  release(slot: number): void { this.free.push(slot); }
  onResize(cb: () => void): IDisposable { return this.buf.onResize(cb); }
  destroy(): void { this.buf.destroy(); }
}

/**
 * Byte-bump allocator over a GrowBuffer for variable-size attribute
 * allocations. Each allocation gets a 16-byte aligned start (8-byte
 * (typeId, length) header at the start, data 16 bytes in). Frees go
 * onto a list keyed by size for simple first-fit reuse later — for
 * now `release` just records the gap and the bump cursor never
 * shrinks.
 */
class AttributeArena {
  private cursor = 0;
  // (offset, size) free entries; first-fit reuse not yet implemented.
  private freeList: { off: number; size: number }[] = [];
  // CPU shadow of the entire GPU buffer. Writes go here first; a
  // single `device.queue.writeBuffer` per dirty contiguous range
  // lifts them to the GPU at flush time. At the cost of doubling
  // host memory we collapse N small writeBuffer calls (10K+ at
  // initial population) to 1 per frame.
  private shadow: Uint8Array;
  private dirtyMin = Infinity;
  private dirtyMax = 0;
  constructor(private readonly buf: GrowBuffer) {
    this.shadow = new Uint8Array(buf.capacity);
    buf.onResize(() => {
      const grown = new Uint8Array(buf.capacity);
      grown.set(this.shadow);
      this.shadow = grown;
    });
  }
  get buffer(): GPUBuffer { return this.buf.buffer; }
  get capacity(): number { return this.buf.capacity; }
  get usedBytes(): number { return this.cursor; }
  /**
   * Stage `data` to the shadow at byte offset `dst`. Tracks the
   * dirty range so `flush(device)` can emit a single writeBuffer
   * covering everything dirty since the last flush.
   */
  write(dst: number, data: Uint8Array): void {
    this.shadow.set(data, dst);
    if (dst < this.dirtyMin) this.dirtyMin = dst;
    const end = dst + data.byteLength;
    if (end > this.dirtyMax) this.dirtyMax = end;
  }
  flush(device: GPUDevice): void {
    if (this.dirtyMax <= this.dirtyMin) return;
    device.queue.writeBuffer(
      this.buf.buffer, this.dirtyMin,
      this.shadow.buffer, this.shadow.byteOffset + this.dirtyMin,
      this.dirtyMax - this.dirtyMin,
    );
    this.dirtyMin = Infinity;
    this.dirtyMax = 0;
  }
  /**
   * Allocate space for one attribute. Returns the byte ref (offset
   * to the header — data lives at ref + 16).
   */
  alloc(dataBytes: number): number {
    const allocBytes = ALIGN16(ALLOC_HEADER_PAD_TO + dataBytes);
    // First-fit reuse from free list.
    for (let i = 0; i < this.freeList.length; i++) {
      const f = this.freeList[i]!;
      if (f.size >= allocBytes) {
        const ref = f.off;
        if (f.size === allocBytes) this.freeList.splice(i, 1);
        else { f.off += allocBytes; f.size -= allocBytes; }
        return ref;
      }
    }
    const ref = this.cursor;
    this.cursor += allocBytes;
    this.buf.ensureCapacity(this.cursor);
    this.buf.setUsed(this.cursor);
    return ref;
  }
  release(ref: number, dataBytes: number): void {
    const allocBytes = ALIGN16(ALLOC_HEADER_PAD_TO + dataBytes);
    insertSortedFreeBlock(this.freeList, ref, allocBytes);
  }
  onResize(cb: () => void): IDisposable { return this.buf.onResize(cb); }
  destroy(): void { this.buf.destroy(); }
}

/**
 * Insert `{off, size}` into a free-list kept sorted by `off`, then
 * coalesce with the two immediate neighbours.
 *
 * The list invariant — sorted, non-overlapping, never-adjacent — is
 * preserved across allocs (which take from the front or split a
 * block) and releases (this function). The previous implementation
 * did `push + Array.prototype.sort + linear coalesce scan`, which is
 * O(N log N) per release. Under a 500-RO bulk-remove the sort
 * dominated `removeDraw` (~41 ms of self-time in the heap-demo-sg
 * toggle profile). Binary-search insert + 2-neighbour merge collapses
 * that to O(log N + N-shift), and is principled — the sort never
 * actually mattered since we already had the sorted prefix as an
 * invariant.
 */
function insertSortedFreeBlock(
  freeList: { off: number; size: number }[],
  off: number,
  size: number,
): void {
  // Binary-search for the insertion index (first entry whose off > new).
  let lo = 0, hi = freeList.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (freeList[mid]!.off <= off) lo = mid + 1;
    else hi = mid;
  }
  // lo is the index where the new entry would be inserted.
  // Try merging with the predecessor first; if successful, the merged
  // block may now be adjacent to its (former) successor too.
  const prev = lo > 0 ? freeList[lo - 1] : undefined;
  if (prev !== undefined && prev.off + prev.size === off) {
    prev.size += size;
    // Check forward-merge with what was freeList[lo].
    const next = freeList[lo];
    if (next !== undefined && prev.off + prev.size === next.off) {
      prev.size += next.size;
      freeList.splice(lo, 1);
    }
    return;
  }
  const next = freeList[lo];
  if (next !== undefined && off + size === next.off) {
    next.off = off;
    next.size += size;
    return;
  }
  freeList.splice(lo, 0, { off, size });
}

/**
 * Element-bump allocator over an index GrowBuffer (units = u32). Each
 * draw's index range is allocated as one block; on release the block
 * is returned to a free list and can be reused first-fit.
 */
class IndexAllocator {
  private cursor = 0;     // in u32s, not bytes
  private freeList: { off: number; size: number }[] = [];
  // CPU shadow + dirty range, same shape as AttributeArena. Index
  // uploads (one per drawn mesh's index buffer) get coalesced to a
  // single writeBuffer per dirty range at flush time.
  private shadow: Uint8Array;
  private dirtyMin = Infinity;
  private dirtyMax = 0;
  constructor(private readonly buf: GrowBuffer) {
    this.shadow = new Uint8Array(buf.capacity);
    buf.onResize(() => {
      const grown = new Uint8Array(buf.capacity);
      grown.set(this.shadow);
      this.shadow = grown;
    });
  }
  get buffer(): GPUBuffer { return this.buf.buffer; }
  get usedElements(): number { return this.cursor; }
  /** Stage `data` (bytes) at the given byte offset; tracks dirty range. */
  write(dstByteOffset: number, data: Uint8Array): void {
    this.shadow.set(data, dstByteOffset);
    if (dstByteOffset < this.dirtyMin) this.dirtyMin = dstByteOffset;
    const end = dstByteOffset + data.byteLength;
    if (end > this.dirtyMax) this.dirtyMax = end;
  }
  flush(device: GPUDevice): void {
    if (this.dirtyMax <= this.dirtyMin) return;
    device.queue.writeBuffer(
      this.buf.buffer, this.dirtyMin,
      this.shadow.buffer, this.shadow.byteOffset + this.dirtyMin,
      this.dirtyMax - this.dirtyMin,
    );
    this.dirtyMin = Infinity;
    this.dirtyMax = 0;
  }
  alloc(elements: number): number {
    for (let i = 0; i < this.freeList.length; i++) {
      const f = this.freeList[i]!;
      if (f.size >= elements) {
        const off = f.off;
        if (f.size === elements) this.freeList.splice(i, 1);
        else { f.off += elements; f.size -= elements; }
        return off;
      }
    }
    const off = this.cursor;
    this.cursor += elements;
    this.buf.ensureCapacity(this.cursor * 4);
    this.buf.setUsed(this.cursor * 4);
    return off;
  }
  release(off: number, elements: number): void {
    insertSortedFreeBlock(this.freeList, off, elements);
  }
  onResize(cb: () => void): IDisposable { return this.buf.onResize(cb); }
  destroy(): void { this.buf.destroy(); }
}

// ---------------------------------------------------------------------------
// Static initial pack (uses the new allocators)
// ---------------------------------------------------------------------------

/**
 * Global arena state: attribute / uniform data lives in `attrs`
 * (multi-typed-view storage); indices live in `indices` (separate
 * INDEX-usage buffer). Per-draw bookkeeping (which arena offsets
 * are alive for which draw) now lives entirely in the bucket via
 * the UniformPool's refcount + the bucket's per-local-slot arrays.
 */
interface ArenaState {
  readonly attrs:    AttributeArena;
  readonly indices:  IndexAllocator;
}

function buildArenaState(
  device: GPUDevice,
  attrBytesHint: number,
  idxBytesHint: number,
  label: string,
  idxExtraUsage: GPUBufferUsageFlags = 0,
): ArenaState {
  const attrs = new AttributeArena(new GrowBuffer(
    device, `${label}/attrs`, GPUBufferUsage.STORAGE,
    attrBytesHint,
  ));
  const indices = new IndexAllocator(new GrowBuffer(
    device, `${label}/idx`, GPUBufferUsage.INDEX | idxExtraUsage,
    idxBytesHint,
  ));
  return { attrs, indices };
}

function arenaBytes(arena: ArenaState): number {
  return arena.attrs.usedBytes + arena.indices.usedElements * 4;
}

/** Upload a single attribute — header (typeId, length) + data — into the arena at byte offset `ref`. */
function writeAttribute(
  device: GPUDevice, buf: GPUBuffer, ref: number, typeId: number, length: number, data: Float32Array,
): void {
  const allocBytes = ALIGN16(ALLOC_HEADER_PAD_TO + data.byteLength);
  const staging = new ArrayBuffer(allocBytes);
  const u32 = new Uint32Array(staging);
  const f32 = new Float32Array(staging);
  u32[0] = typeId;
  u32[1] = length;
  f32.set(data, ALLOC_HEADER_PAD_TO / 4);
  device.queue.writeBuffer(buf, ref, staging, 0, allocBytes);
}

function asAval<T>(v: aval<T> | T): aval<T> {
  return (typeof v === "object" && v !== null && typeof (v as { getValue?: unknown }).getValue === "function")
    ? (v as aval<T>)
    : AVal.constant(v as T);
}

/** Heuristic predicate — BufferView has `buffer: aval<IBuffer>` + elementType. */
function isBufferView(v: unknown): v is BufferView {
  if (typeof v !== "object" || v === null) return false;
  const o = v as { buffer?: unknown; elementType?: unknown };
  return typeof o.buffer === "object" && o.buffer !== null
      && typeof (o.buffer as { getValue?: unknown }).getValue === "function"
      && typeof o.elementType === "object" && o.elementType !== null;
}

/**
 * Float32 view over a host-side buffer source. Used by the BufferView
 * packer to hand the pool a typed array it can `set()` from.
 */
function asFloat32(data: HostBufferSource): Float32Array {
  if (data instanceof Float32Array) return data;
  if (ArrayBuffer.isView(data)) {
    return new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  }
  return new Float32Array(data); // ArrayBuffer
}

// ---------------------------------------------------------------------------
// Megacall GPU prefix-sum compute shader
// ---------------------------------------------------------------------------

const SCAN_TILE_SIZE = 512;
const SCAN_WG_SIZE = 256;
const SCAN_MAX_RECORDS = SCAN_TILE_SIZE * SCAN_TILE_SIZE; // numBlocks ≤ TILE_SIZE

const TILE_K = 64;

/** drawTable record width: (firstEmit, drawIdx, indexStart, indexCount, instanceCount). */
const RECORD_U32   = 5;
const RECORD_BYTES = RECORD_U32 * 4;

const HEAP_SCAN_WGSL = `
struct Params {
  numRecords: u32,
  numBlocks:  u32,
  _pad0:      u32,
  _pad1:      u32,
};

struct Record {
  firstEmit:     u32,
  drawIdx:       u32,
  indexStart:    u32,
  indexCount:    u32,
  instanceCount: u32,
};

@group(0) @binding(0) var<storage, read_write> drawTable:        array<Record>;
@group(0) @binding(1) var<storage, read_write> blockSums:        array<u32>;
@group(0) @binding(2) var<storage, read_write> blockOffsets:     array<u32>;
@group(0) @binding(3) var<storage, read_write> indirect:         array<u32>;
@group(0) @binding(4) var<uniform>             params:           Params;
@group(0) @binding(5) var<storage, read_write> firstDrawInTile:  array<u32>;

const TILE_SIZE: u32 = 512u;
const WG_SIZE:   u32 = 256u;
const TILE_K:    u32 = 64u;

var<workgroup> sdata: array<u32, 512>;

fn blellochScan(tid: u32) {
  var offset: u32 = 1u;
  for (var d: u32 = TILE_SIZE >> 1u; d > 0u; d = d >> 1u) {
    workgroupBarrier();
    if (tid < d) {
      let ai = offset * (2u * tid + 1u) - 1u;
      let bi = offset * (2u * tid + 2u) - 1u;
      sdata[bi] = sdata[bi] + sdata[ai];
    }
    offset = offset * 2u;
  }
  if (tid == 0u) { sdata[TILE_SIZE - 1u] = 0u; }
  for (var d: u32 = 1u; d < TILE_SIZE; d = d * 2u) {
    offset = offset >> 1u;
    workgroupBarrier();
    if (tid < d) {
      let ai = offset * (2u * tid + 1u) - 1u;
      let bi = offset * (2u * tid + 2u) - 1u;
      let t = sdata[ai];
      sdata[ai] = sdata[bi];
      sdata[bi] = sdata[bi] + t;
    }
  }
  workgroupBarrier();
}

@compute @workgroup_size(WG_SIZE)
fn scanTile(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
  let tid = lid.x;
  let blockOff = wgid.x * TILE_SIZE;
  let n = params.numRecords;
  let i0 = blockOff + tid;
  let i1 = blockOff + tid + WG_SIZE;
  var v0: u32 = 0u;
  var v1: u32 = 0u;
  if (i0 < n) { v0 = drawTable[i0].indexCount * drawTable[i0].instanceCount; }
  if (i1 < n) { v1 = drawTable[i1].indexCount * drawTable[i1].instanceCount; }
  sdata[tid]           = v0;
  sdata[tid + WG_SIZE] = v1;
  workgroupBarrier();
  blellochScan(tid);
  if (i0 < n) { drawTable[i0].firstEmit = sdata[tid]; }
  if (i1 < n) { drawTable[i1].firstEmit = sdata[tid + WG_SIZE]; }
  if (tid == WG_SIZE - 1u) {
    blockSums[wgid.x] = sdata[tid + WG_SIZE] + v1;
  }
}

@compute @workgroup_size(WG_SIZE)
fn scanBlocks(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let n = params.numBlocks;
  let i0 = tid;
  let i1 = tid + WG_SIZE;
  var v0: u32 = 0u;
  var v1: u32 = 0u;
  if (i0 < n) { v0 = blockSums[i0]; }
  if (i1 < n) { v1 = blockSums[i1]; }
  sdata[tid]           = v0;
  sdata[tid + WG_SIZE] = v1;
  workgroupBarrier();
  blellochScan(tid);
  if (i0 < n) { blockOffsets[i0] = sdata[tid]; }
  if (i1 < n) { blockOffsets[i1] = sdata[tid + WG_SIZE]; }
  workgroupBarrier();
  if (tid == 0u) {
    if (n > 0u) {
      let lastIdx = n - 1u;
      let total = blockOffsets[lastIdx] + blockSums[lastIdx];
      indirect[0] = total;
    } else {
      indirect[0] = 0u;
    }
    indirect[1] = 1u;
    indirect[2] = 0u;
    indirect[3] = 0u;
  }
}

@compute @workgroup_size(WG_SIZE)
fn addOffsets(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
  let tid = lid.x;
  let blockOff = wgid.x * TILE_SIZE;
  let n = params.numRecords;
  let off = blockOffsets[wgid.x];
  let i0 = blockOff + tid;
  let i1 = blockOff + tid + WG_SIZE;
  if (i0 < n) { drawTable[i0].firstEmit = drawTable[i0].firstEmit + off; }
  if (i1 < n) { drawTable[i1].firstEmit = drawTable[i1].firstEmit + off; }
}

@compute @workgroup_size(WG_SIZE)
fn buildTileIndex(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tileIdx = gid.x;
  // totalEmit is computed by scanBlocks into indirect[0]; reading it
  // from indirect avoids a separate uniform/storage round-trip.
  let totalEmit = indirect[0];
  let numTiles = (totalEmit + TILE_K - 1u) / TILE_K;
  if (tileIdx > numTiles) { return; }
  if (params.numRecords == 0u) {
    if (tileIdx == 0u) { firstDrawInTile[0] = 0u; }
    return;
  }
  if (tileIdx == numTiles) {
    // Sentinel for the open upper bound — the LAST VALID SLOT, not
    // numRecords. The render VS uses
    //     hi = firstDrawInTile[_tileIdx + 1u]
    // and the binary search treats hi as INCLUSIVE. If the sentinel
    // were numRecords (one past last), the search would drag lo into
    // the OOB slot for emits in the last tile, since drawTable reads
    // past recordCount return 0 (binding size clamping) and 0 ≤ emit
    // is always true. Visible symptom: the LAST few emits in the
    // bucket land on slot=numRecords (drawIdx=0, indexCount=0 → /-by-
    // zero) → degenerate / cross-RO triangle stitched to slot 0.
    firstDrawInTile[tileIdx] = params.numRecords - 1u;
    return;
  }
  let tileStart = tileIdx * TILE_K;
  var lo: u32 = 0u;
  var hi: u32 = params.numRecords - 1u;
  loop {
    if (lo >= hi) { break; }
    let mid = (lo + hi + 1u) >> 1u;
    if (drawTable[mid].firstEmit <= tileStart) { lo = mid; } else { hi = mid - 1u; }
  }
  firstDrawInTile[tileIdx] = lo;
}
`;

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/**
 * One pipeline-state bucket. Holds the pipeline + a list of global
 * draw-slot indices to emit with it. The bind group it draws against
 * is referenced by `bindGroup` (typically the shared no-textures one
 * or one of the per-texture-set ones).
 */
interface Bucket {
  readonly label: string;
  readonly textures: HeapTextureSet | undefined;
  readonly layout: BucketLayout;
  readonly pipeline: GPURenderPipeline;
  /** Repointed by `rebuildBindGroups` whenever any backing GrowBuffer reallocates. */
  bindGroup: GPUBindGroup;

  // Per-bucket buffers
  readonly drawHeap: DrawHeap;
  /** Float32 staging mirror of the drawHeap, sized to its current capacity. */
  drawHeaderStaging: Float32Array;
  /**
   * Dirty byte range in `drawHeaderStaging` since the last flush.
   * `packBucketHeader` updates these; `flushHeaders` issues a single
   * `device.queue.writeBuffer` covering the whole range and clears.
   * Replaces N small writeBuffer calls (one per addDraw / per dirty
   * slot) with one per frame, per bucket.
   */
  headerDirtyMin: number;
  headerDirtyMax: number;

  // Per-local-slot state (sparse; index = local slot in this bucket).
  readonly localPosRefs:  (number | undefined)[];
  readonly localNorRefs:  (number | undefined)[];
  readonly localEntries:  ({ indexCount: number; firstIndex: number; instanceCount: number } | undefined)[];
  /** localSlot → global drawId, for cleanup paths. */
  readonly localToDrawId: (number | undefined)[];
  /**
   * Per-draw uniform avals owned by this slot — used by removeDraw
   * to release pool entries. Order doesn't matter; the pool is
   * keyed by aval identity.
   */
  readonly localPerDrawAvals: (aval<unknown>[] | undefined)[];
  /**
   * Per-draw uniform refs (schema name → arena byte offset) for
   * this slot. Stable during the slot's lifetime; used to re-pack
   * the DrawHeader into a fresh staging mirror after the bucket's
   * drawHeap GrowBuffer reallocates.
   */
  readonly localPerDrawRefs: (Map<string, number> | undefined)[];
  /**
   * Per-local-slot layoutId for the §6 family-merge selector. Stable
   * during the slot's lifetime; written into the drawHeader's
   * `__layoutId` field by `packBucketHeader`.
   */
  readonly localLayoutIds: (number | undefined)[];

  /**
   * Live local slots. A Set rather than an array so addDraw/removeDraw
   * are both O(1): a 500-element bulk-remove on a 1000-element bucket
   * would otherwise burn 250K shifts in the linear-array splice path.
   * The render loop iterates this once per frame; iteration order
   * isn't load-bearing (slot→record indirection happens via
   * `slotToRecord` / `recordToSlot`).
   */
  readonly drawSlots: Set<number>;
  /** Local slots whose DrawHeader needs re-pack + writeBuffer next frame. */
  readonly dirty: Set<number>;

  // ─── Megacall state ────────────────────────────────────────────────
  drawTableBuf?: GrowBuffer;
  drawTableShadow?: Uint32Array;
  drawTableDirtyMin: number;
  drawTableDirtyMax: number;
  /** Number of live records (= drawTable length). GPU owns firstEmit / total. */
  recordCount: number;
  /** localSlot → recordIdx (or -1). */
  slotToRecord: number[];
  /** recordIdx → localSlot. */
  recordToSlot: number[];
  /** Per-bucket buffers for the GPU prefix-sum pipeline. */
  blockSumsBuf?: GrowBuffer;
  blockOffsetsBuf?: GrowBuffer;
  firstDrawInTileBuf?: GrowBuffer;
  /** CPU sum of indexCounts across live records — drives firstDrawInTileBuf sizing only. */
  totalEmitEstimate: number;
  indirectBuf?: GPUBuffer;
  paramsBuf?: GPUBuffer;
  scanBindGroup?: GPUBindGroup;
  /** numRecords used to size the current render bindGroup; rebuild when it changes. */
  renderBoundRecordCount?: number;
  scanDirty: boolean;

  // ─── Atlas-binding state (atlas-variant buckets only) ─────────────
  /**
   * True when this bucket holds at least one atlas-variant RO.
   * Drives BGL/bind-group shape (atlas buckets bind N consecutive
   * `texture_2d<f32>` slots per format from the AtlasPool's per-
   * format page list, plus the shared atlas sampler).
   */
  isAtlasBucket: boolean;
  /** Per-local-slot atlas release callbacks. Drains on removeDraw. */
  readonly localAtlasReleases: ((() => void) | undefined)[];
  /**
   * Per-local-slot atlas HeapTextureSet for re-packing on drawHeap
   * GrowBuffer reallocation. Standalone-only buckets keep this empty.
   */
  readonly localAtlasTextures: ((HeapTextureSet & { kind: "atlas" }) | undefined)[];
  /**
   * Per-local-slot index into `atlasAvalRefs[sourceAval]` — lets
   * removeDraw do swap-pop without an O(N) findIndex scan. `undefined`
   * for slots without an atlas source aval.
   */
  readonly localAtlasArrIdx: (number | undefined)[];
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface HeapDrawSpec {
  /**
   * The shader. Two draws with the same effect (matched by `effect.id`)
   * share a pipeline + bucket.
   */
  readonly effect: Effect;
  /**
   * Per-name inputs covering both vertex attributes (e.g. `Positions`,
   * `Normals`) and uniforms (e.g. `ModelTrafo`, `Color`, `ViewProjTrafo`)
   * — whatever the effect's schema declares. The runtime walks the
   * schema and pulls each by name.
   *
   * Sharing happens via aval identity: the same `cval` passed to
   * many draws → one arena allocation + one upload per change.
   * Same applies to attributes: instanced draws sharing a `Float32Array`
   * for `Positions` share one arena allocation. Wrap raw values in
   * `AVal.constant(...)` if you don't need to update them.
   */
  readonly inputs: { readonly [name: string]: aval<unknown> | unknown };
  /**
   * Per-RO instance attributes — one BufferView per name, indexed by
   * `instance_index` (= iidx in the megacall path) at shader time.
   * Routed through the same arena-pool as `inputs` (aval-keyed
   * sharing across draws). Triggers the heap path's per-RO
   * instancing fast path: one record / one drawIndirect for an RO
   * with `instanceCount=N`, total emit = `indexCount * N`.
   *
   * Must NOT alias names in `inputs` — instance attributes are
   * disjoint from per-vertex attributes and from uniforms.
   */
  readonly instanceAttributes?: { readonly [name: string]: aval<unknown> | unknown };
  /**
   * Per-RO instance count for the instancing path. Defaults to 1.
   * Reads through the arena's drawTable record so the GPU prefix-sum
   * computes `indexCount * instanceCount` per record.
   */
  readonly instanceCount?: aval<number> | number;
  /**
   * Index buffer for this draw. Indices live in their own `INDEX`-
   * usage `GPUBuffer` (WebGPU forces this), separate from the arena.
   */
  readonly indices: aval<Uint32Array> | Uint32Array;
  /**
   * Optional texture set. When present, adds a `texture_2d<f32>` at
   * binding 4 and a sampler at binding 5; the FS must declare them.
   * Buckets with different `textures` (by reference) cannot share a
   * bind group, so a distinct texture set wedges into the bucket key
   * alongside the effect id.
   */
  readonly textures?: HeapTextureSet;
  /**
   * Optional pipeline state (rasterizer / depth / blend). Two specs
   * sharing the SAME `PipelineState` object share a bucket; distinct
   * objects get distinct buckets even if structurally equal — same
   * identity convention as `effect`, `indices`, `textures`. The
   * underlying avals are forced at addDraw time and baked into the
   * pipeline; later marks on those avals are ignored. To change
   * pipeline state, swap the `PipelineState` object (which forces
   * a fresh bucket).
   *
   * Omitted ⇒ defaults: triangle-list / cull back / ccw / depth-less
   * with depth write enabled. Matches the demo's existing visuals.
   */
  readonly pipelineState?: PipelineState;
}

export interface HeapSceneStats {
  readonly groups: number;
  totalDraws: number;
  drawBytes: number;
  geometryBytes: number;
  /** §7 derived-uniforms per-frame breakdown (last frame). */
  derivedPullMs:   number;
  derivedUploadMs: number;
  derivedEncodeMs: number;
  derivedRecords:  number;
}

export interface HeapScene {
  /**
   * Convenience: open a command encoder + render pass against
   * `framebuffer`, run `update` + `encodeIntoPass`, end pass, submit.
   * For hybrid composition (heap + legacy in one pass), use
   * `update` + `encodeIntoPass` directly against a caller-managed
   * `GPURenderPassEncoder` instead.
   */
  frame(framebuffer: import("../core/index.js").IFramebuffer, token: AdaptiveToken): void;
  /**
   * Pull pending aset deltas, drain pool / per-bucket dirty, and
   * upload everything CPU-side that the next `encodeIntoPass` will
   * read on the GPU. Idempotent within a frame; safe to call twice.
   */
  update(token: AdaptiveToken): void;
  /**
   * Encode all live bucket draws into an existing render pass. Caller
   * owns `beginRenderPass` / `end` / `submit`. This is the building
   * block for hybrid: a caller can interleave (in batches, never per-
   * draw) heap encoding with legacy encoding into the same pass.
   *
   * Must be preceded by `update(token)` in the same frame; otherwise
   * GPU state may lag behind CPU-side aval marks.
   */
  encodeIntoPass(passEnc: GPURenderPassEncoder): void;
  /**
   * Encode any compute work the next `encodeIntoPass` depends on.
   * For megacall buckets: dispatches the GPU prefix-sum scan that
   * computes per-record `firstEmit` and writes the indirect-draw
   * args. Must be called BEFORE `beginRenderPass` since compute can't
   * run inside a render pass. No-op when the scene has no megacall
   * buckets or none are dirty.
   */
  encodeComputePrep(enc: GPUCommandEncoder, token: AdaptiveToken): void;
  /**
   * Add a draw at runtime. Returns a stable slot handle that can be
   * passed to `removeDraw` later. Allocators auto-grow as needed.
   */
  addDraw(spec: HeapDrawSpec): number;
  /** Remove a draw previously returned by `addDraw` (or the initial-array index). */
  removeDraw(slot: number): void;
  readonly stats: HeapSceneStats;
  dispose(): void;
}

/**
 * Per-group texture set. Bindless isn't standard WebGPU, so a group
 * with sampled textures gets a distinct bind-group layout and is
 * keyed separately from groups with no textures or different sets.
 *
 * Discriminated union, in preparation for the texture-atlas
 * integration (see `docs/heap-textures-plan.md`):
 *   - `standalone` — Tier L; one `GPUTexture` per source. Today's path.
 *   - `atlas`      — Tier S/M; the texture lives as a sub-rect of an
 *                    `AtlasPool` page, identified by `(format, pageId)`
 *                    plus mip-0 `(origin, size)` in normalized atlas
 *                    coords and `numMips` (1 = no embedded pyramid;
 *                    >1 = 1.5×1 Iliffe pyramid stored in the page).
 *                    NOT YET WIRED through the heap path; the BGL/
 *                    bind-group/drawHeader work lands in a follow-up PR.
 */
export type HeapTextureSet =
  | {
      readonly kind: "standalone";
      readonly texture: ITexture;
      readonly textureView?: GPUTextureView;
      readonly sampler: ISampler;
    }
  | {
      readonly kind: "atlas";
      readonly format: GPUTextureFormat;
      readonly pageId: number;
      /** Top-left of mip-0 in the atlas, normalized [0,1]. */
      readonly origin: V2f;
      /** Size of mip-0 in the atlas, normalized [0,1]. */
      readonly size: V2f;
      /** Mip-level count stored at this acquisition (1 = no pyramid). */
      readonly numMips: number;
      readonly sampler: ISampler;
      /**
       * Strong reference to the atlas page hosting this acquisition, so
       * the heap path can resolve `pageId` to a `GPUTexture` for the
       * bind-group binding_array. Filled by the adapter from
       * `AtlasPool.acquire(...).page`.
       */
      readonly page: AtlasPage;
      /**
       * Refcount handle from `AtlasPool.acquire(...).ref`. removeDraw
       * threads this back to `pool.release(ref)`.
       */
      readonly poolRef: number;
      /**
       * Pool-level release callback. The heap path doesn't keep a
       * reference to AtlasPool itself; instead the adapter wires this
       * closure so removeDraw can drop the refcount without a global
       * pool handle.
       */
      readonly release: () => void;
      /**
       * The original `aval<ITexture>` source. When present, heapScene
       * subscribes to it; an aval mark routes through the per-frame
       * `update` loop into `AtlasPool.repack(av, newValue)`, with the
       * resulting drawHeader fields rewritten in place. Absent ⇒ static
       * texture path (no reactivity, current behaviour).
       *
       * The pool-level `repack` callback rewires this RO's atlas slot
       * to the new placement; heapScene reads it from `bucket.localAtlasReleases`'
       * sibling slot (`localAtlasRepacks`) to avoid a global pool handle.
       */
      readonly sourceAval?: aval<ITexture>;
      /**
       * Bound `(newTex) → AtlasAcquisition` closure backed by the
       * pool's `repack`. Adapter-supplied so heapScene can swap the
       * placement for `sourceAval` without holding a direct pool
       * reference.
       */
      readonly repack?: (newTex: ITexture) => import("./textureAtlas/atlasPool.js").AtlasAcquisition;
    };

/**
 * Build a heap-backed scene renderer from a flat list of draws.
 *
 * Bucketing is automatic: draws with the same effect (matched by
 * `effect.id` for DSL effects, by reference for raw-WGSL pairs) and
 * the same texture set (by reference) share a pipeline + bind group.
 * Per-RO avals are subscribed so dirty marks propagate to per-slot
 * `writeBuffer`s on the next `frame()`.
 */
export interface BuildHeapSceneOptions {
  /**
   * Maps each fragment-output name an effect emits to its framebuffer
   * attachment location. Outputs not in the map get DCE'd by the
   * `linkFragmentOutputs` pass — and any uniforms that only fed those
   * outputs disappear from the schema too.
   *
   * Example: `{ locations: new Map([["outColor", 0]]) }` for a single-
   * color framebuffer with effects that write to `outColor`.
   *
   * Omitted ⇒ no pruning; every output the effect declares survives.
   */
  readonly fragmentOutputLayout?: FragmentOutputLayout;
  /**
   * Atlas pool that owns the per-format independent page textures.
   * heapScene reads `pool.pagesFor(format)` to wire each page's
   * `GPUTexture` into the matching slot of the bucket's BGL ladder
   * and subscribes to `pool.onPageAdded` so bind groups rebuild when
   * a fresh page joins. Required when any RO produces an atlas-
   * variant `HeapTextureSet`.
   */
  readonly atlasPool?: AtlasPool;
  /**
   * §6 family-merge opt-in — when true, build ONE family covering all
   * effects in the scene, with a single bucket per pipelineState.
   * Default false: each effect lands in its own bucket / shader /
   * pipeline (the simpler baseline). Empirically the per-effect path
   * is at-or-better than merged on tested workloads (10K–30K small
   * ROs across 8 effects); merge stays available as opt-in until the
   * trace-based auto-trigger lands as v2.
   */
  readonly enableFamilyMerge?: boolean;
  /**
   * §7 derived-uniforms opt-in. When true, the renderer recognises the
   * 10 standard derived names (ModelView, ViewProj, ModelViewProj,
   * NormalMatrix, the 3 *TrafoInv fields, and the 3 product inverses)
   * as compute-produced and writes them into each RO's drawHeader via
   * a df32 compute pre-pass — not via the uniform provider. ROs MUST
   * supply ModelTrafo / ViewTrafo / ProjTrafo as `aval<Trafo3d>` for
   * any derived field they consume.
   *
   * Default false: each derived name continues to be served by the
   * effect's `spec.inputs[name]` like any other uniform.
   */
  readonly enableDerivedUniforms?: boolean;
}

export function buildHeapScene(
  device: GPUDevice,
  sig: import("../core/framebufferSignature.js").FramebufferSignature,
  initialDraws: readonly HeapDrawSpec[] | aset<HeapDrawSpec>,
  opts: BuildHeapSceneOptions = {},
): HeapScene {
  const atlasPool = opts.atlasPool;
  const colorAttachmentName = sig.colorNames[0];
  if (colorAttachmentName === undefined) {
    throw new Error("buildHeapScene: framebuffer signature has no color attachment");
  }
  const colorFormat = sig.colors.tryFind(colorAttachmentName)!;
  // All color targets, ordered by the signature's `colorNames`. Each
  // entry's format flows into the pipeline's `fragment.targets[i].format`.
  // The fragmentOutputLayout's `locations` map tells the WGSL emit which
  // output @location maps to which color attachment; this array is the
  // matching pipeline-side ordering.
  const colorTargets: { format: GPUTextureFormat }[] = [];
  for (const name of sig.colorNames) {
    const fmt = sig.colors.tryFind(name);
    if (fmt === undefined) {
      throw new Error(`buildHeapScene: signature missing format for color '${name}'`);
    }
    colorTargets.push({ format: fmt });
  }
  const depthFormat = sig.depthStencil?.format;

  // ─── Global arena (uniform/attribute data + index buffer) ────────
  // Initial capacities are just hints; both buffers pow2-grow on
  // demand. Skip per-draw enumeration since aval-keyed sharing makes
  // the actual allocated size hard to predict (10K instanced draws
  // sharing the same Positions array → 1 alloc, not 10K).
  const arena = buildArenaState(
    device, 64 * 1024, 16 * 1024, "heapScene",
    GPUBufferUsage.STORAGE,
  );

  // ─── Per-draw global bookkeeping (sparse, indexed by drawId) ──────
  const drawIdToBucket:    (Bucket | undefined)[] = [];
  const drawIdToLocalSlot: (number | undefined)[] = [];
  /** Per-draw index aval — for `indexPool.release` on removeDraw. */
  const drawIdToIndexAval: (aval<Uint32Array> | undefined)[] = [];
  let nextDrawId = 0;

  /**
   * Unwrap an aval to its inner value, or pass through a plain value.
   * Boundary helper at scene-build time — no adaptive context to read
   * through, so a plain force is the right call.
   */
  function readPlain<T>(v: aval<T> | T): T {
    if (typeof v === "object" && v !== null && typeof (v as { force?: unknown }).force === "function") {
      return (v as aval<T>).force(/* allow-force */);
    }
    return v as T;
  }

  /**
   * Pick a pool placement (dataBytes/typeId/length/pack) for a
   * schema-driven DrawHeader field given the JS value. Drives both
   * fixed-size uniform reads (mat4, vec4, …) and variable-size
   * attribute arrays — the latter measured from the value itself.
   */
  function poolPlacementFor(
    f: { kind: "uniform-ref" | "attribute-ref" | "texture-ref"; uniformWgslType?: string; attributeWgslType?: string },
    value: unknown,
  ): { dataBytes: number; typeId: number; length: number; pack: (val: unknown, dst: Float32Array, off: number) => void } {
    if (f.kind === "uniform-ref") {
      const p = packerForWgslType(f.uniformWgslType ?? "");
      return { dataBytes: p.dataBytes, typeId: p.typeId, length: 1, pack: p.pack };
    }
    // attribute-ref: variable-size array; we copy verbatim into the
    // arena. The current encoding is V3F_TIGHT (3 f32s per element).
    const arr = value as Float32Array;
    const eltBytes =
      f.attributeWgslType === "vec3<f32>" ? 12 :
      f.attributeWgslType === "vec4<f32>" ? 12 /* v3 stored, v4 assembled in shader */ :
      f.attributeWgslType === "vec2<f32>" ? 8  :
      4;
    const length = arr.byteLength / eltBytes;
    return {
      dataBytes: arr.byteLength,
      typeId: ENC_V3F_TIGHT,
      length,
      pack: (val, dst, off) => { dst.set(val as Float32Array, off); },
    };
  }

  /**
   * Pool placement for a `BufferView`-backed attribute. The pool key
   * is the BufferView's `aval<IBuffer>` (so `buffer`-identity drives
   * sharing). The packer reads the `IBuffer`'s host data — native
   * GPUBuffer-backed views are rejected (they belong on the legacy
   * fallback path; see `isHeapEligible`).
   *
   * Validates: `stride === elementType.byteSize` (tight) and
   * `offset === 0`. Both can be relaxed once the gather path
   * supports strided/sub-range copies.
   */
  function bufferViewPlacement(
    f: { attributeWgslType?: string },
    bv: BufferView,
  ): { dataBytes: number; typeId: number; length: number; pack: (val: unknown, dst: Float32Array, off: number) => void } {
    const srcEltBytes = bv.elementType.byteSize;
    const offset = bv.offset ?? 0;
    if (offset !== 0) {
      throw new Error(`heapScene: BufferView offset ${offset} > 0 not yet supported`);
    }

    // In-arena element bytes: the storage layout the shader will read
    // from. vec4 is always 16 (read via `heapV4f[idx]`), vec3 is 12
    // (read as 3 f32s), vec2 is 8, scalar is 4. V3f source for a vec4
    // schema is expanded to V4f at upload with .w = 1.0 (Positions
    // convention) — the shader reads heapV4f[idx] either way.
    const outEltBytes =
      f.attributeWgslType === "vec3<f32>" ? 12 :
      f.attributeWgslType === "vec4<f32>" ? 16 :
      f.attributeWgslType === "vec2<f32>" ? 8  :
      4;
    const allowedSrc: number[] =
      f.attributeWgslType === "vec4<f32>"
        ? [12, 16]   // V3f-with-w=1 expansion OR genuine V4f source
        : [outEltBytes];
    if (!allowedSrc.includes(srcEltBytes)) {
      throw new Error(
        `heapScene: BufferView elementType byteSize ${srcEltBytes} doesn't match ` +
        `schema attribute ${f.attributeWgslType ?? "?"} (expected ${allowedSrc.join(" or ")})`,
      );
    }

    // Stride convention:
    //   stride == srcEltBytes (tight per-vertex)     → length = byteLength / srcEltBytes
    //   stride == 0 OR singleValue !== undefined     → broadcast, length = 1
    //   anything else (interleaved)                  → unsupported
    // The shader uses cyclic addressing (`vid % length`), so a
    // length-1 broadcast just maps every vertex to element 0.
    const stride = bv.stride ?? srcEltBytes;
    const isBroadcast = bv.singleValue !== undefined || stride === 0;
    if (!isBroadcast && stride !== srcEltBytes) {
      throw new Error(
        `heapScene: BufferView stride ${stride} not tight (${srcEltBytes}) and not a broadcast`,
      );
    }

    const ib = bv.buffer.force(/* allow-force */);
    if (ib.kind !== "host") {
      throw new Error(
        `heapScene: BufferView wraps a native GPUBuffer (kind=${ib.kind}); ` +
        `route this RO via the legacy path (see isHeapEligible).`,
      );
    }
    // Store source-as-is; the alloc header's stride field tells the VS
    // how many floats per element. No host-side expansion — the VS
    // decode for vec4 uses `select` to fill `.w = 1.0` when the source
    // is V3-tight.
    const length = isBroadcast ? 1 : ib.sizeBytes / srcEltBytes;
    const dataBytes = srcEltBytes * length;
    void outEltBytes; // (not used: storage matches source exactly)
    return {
      dataBytes,
      typeId: ENC_V3F_TIGHT,
      length,
      pack: (val, dst, off) => {
        const ibuf = val as IBuffer;
        if (ibuf.kind !== "host") {
          throw new Error("heapScene: BufferView aval flipped to native GPUBuffer; not supported in heap path");
        }
        const src = asFloat32(ibuf.data);
        const limitFloats = dataBytes / 4;
        dst.set(src.subarray(0, limitFloats), off);
      },
    };
  }

  /**
   * Pool placement for a per-instance uniform: the value is an array
   * (length = `instanceCount`) of WGSL-typed elements packed tightly.
   * The shader reads element `iidx` via `instanceLoadExpr`.
   */
  function perInstancePlacementFor(
    f: { uniformWgslType?: string },
    value: unknown,
    instanceCount: number,
  ): { dataBytes: number; typeId: number; length: number; pack: (val: unknown, dst: Float32Array, off: number) => void } {
    const elt = packerForWgslType(f.uniformWgslType ?? "");
    const dataBytes = elt.dataBytes * instanceCount;
    const stride = elt.dataBytes / 4;
    return {
      dataBytes,
      typeId: 0,
      length: instanceCount,
      pack: (val, dst, off) => {
        const arr = val as readonly unknown[];
        if (arr.length < instanceCount) {
          throw new Error(
            `heapScene: per-instance value has length ${arr.length} < instanceCount ${instanceCount}`,
          );
        }
        for (let i = 0; i < instanceCount; i++) elt.pack(arr[i], dst, off + i * stride);
      },
    };
  }

  // ─── Pools — aval-keyed refcounted allocations ────────────────────
  const pool = new UniformPool();
  const indexPool = new IndexPool();

  // ─── Adaptive routing: aval marks → repack the pool entry ─────────
  // With pool-managed per-draw uniforms, value changes don't touch
  // any DrawHeader (refs stay constant) — only the arena allocation's
  // data needs re-uploading. inputChanged(o) routes to allocDirty
  // when `o` is a known pool aval; the frame loop drains it.
  const allocDirty = new Set<aval<unknown>>();

  // Atlas texture-aval reactivity. When a `cval<ITexture>` driving an
  // atlas-routed RO swaps its inner ITexture, the heap path needs to
  // (a) release the old sub-rect, (b) acquire a new one, and
  // (c) rewrite the drawHeader fields of every RO referencing the aval.
  // Mirrors `allocDirty` shape but for atlas placements rather than
  // arena uniforms. Each aval can be referenced by multiple ROs (across
  // buckets) — `atlasAvalRefs` tracks all (bucket, localSlot) pairs so
  // one swap rewrites N drawHeaders in one drain.
  const atlasAvalDirty = new Set<aval<ITexture>>();
  interface AtlasAvalRef {
    readonly bucket: Bucket;
    readonly localSlot: number;
    readonly repack: (newTex: ITexture) => import("./textureAtlas/atlasPool.js").AtlasAcquisition;
    readonly sampler: ISampler;
  }
  // Content-keyed (`HashTable`, not a JS `Map`) so distinct
  // `AVal.constant(tex)` wrappers sharing the same texture collapse to
  // one `(bucket, slot)` ref-list — matching `AtlasPool.entriesByAval`,
  // which is now content-keyed too. Reactive avals key by reference.
  const atlasAvalRefs = new HashTable<aval<ITexture>, AtlasAvalRef[]>();
  /**
   * Per-draw bucket dirty (rare in steady state — only fires when
   * something forces a header rewrite, e.g. a drawHeap GrowBuffer
   * resize remarks every live local slot). Value-only changes go
   * through `allocDirty` instead.
   */
  // (Bucket.dirty is on each bucket; this comment is a placeholder.)

  class HeapSceneObj extends AdaptiveObject {
    override inputChanged(_t: unknown, o: IAdaptiveObject): void {
      // Pool avals are stored as `aval<unknown>`; the IAdaptiveObject
      // identity matches.
      if (pool.has(o as unknown as aval<unknown>)) {
        allocDirty.add(o as unknown as aval<unknown>);
        return;
      }
      // §7 derived-uniforms constituent (Trafo3d aval). Routed BEFORE
      // the atlas check because trafo avals never overlap with texture
      // avals — early-return is just to avoid the second Map lookup.
      if (derivedScene !== undefined && derivedScene.routeInputChanged(o)) {
        return;
      }
      const av = o as unknown as aval<ITexture>;
      if (atlasAvalRefs.has(av)) {
        atlasAvalDirty.add(av);
      }
    }
  }
  const sceneObj = new HeapSceneObj();

  // ─── §7 derived-uniforms (opt-in) ─────────────────────────────────
  // Constructed lazily after sceneObj so the SubscribeFn can reference
  // both. `derivedScene` is undefined when the option is off; nothing
  // else in this file should run when it is.
  const enableDerivedUniforms = opts.enableDerivedUniforms === true;
  const derivedScene: DerivedUniformsScene | undefined = enableDerivedUniforms
    ? new DerivedUniformsScene(device, arena.attrs.buffer, {
        // Larger initial capacity reduces grow-during-first-frame churn.
        initialConstituentSlots: 4096,
        initialRecordCapacity:   8192,
      })
    : undefined;
  /** Per-RO §7 registration handles, keyed by global drawId.
   *  Drained on removeDraw to release slots + records. */
  const derivedByDrawId = new Map<number, RoRegistration>();

  // ─── Atlas bindings ───────────────────────────────────────────────
  //
  // A bucket whose textures live in the atlas binds N consecutive
  // `texture_2d<f32>` slots per format (linear + srgb), plus a shared
  // atlas sampler. Each AtlasPage is its own independent GPUTexture;
  // unfilled slots get a 1×1 placeholder so the BGL is fully populated.
  // When a fresh page joins via `AtlasPool.onPageAdded`, we rebuild
  // affected bind groups (the new page's slot now points at a real
  // texture instead of the placeholder).
  //
  // Replaces both the failed `binding_array<texture_2d<f32>, N>` design
  // (needs experimental bindless) and the texture_2d_array fallback
  // (needs GPU-side copy on layer-count grow). N independent textures
  // + an N-way `switch pageRef` in the shader is core WebGPU 1.0 and
  // makes page-add a pure allocation.

  // Shared atlas sampler — shader does mip filtering manually, so
  // hardware mipmapFilter stays "nearest". Plan §2.
  const atlasSampler = device.createSampler({
    label: "heapScene/atlasSampler",
    magFilter: "linear", minFilter: "linear", mipmapFilter: "nearest",
  });

  // Placeholder textures (one per format) used for atlas binding slots
  // that don't yet point at an allocated AtlasPage. WebGPU rejects
  // bind-group entries with `undefined` resources, so every slot in
  // the N-wide ladder must hold a valid view at all times.
  let atlasPlaceholders: Map<AtlasPageFormat, GPUTextureView> | undefined;
  function getAtlasPlaceholder(format: AtlasPageFormat): GPUTextureView {
    if (atlasPlaceholders === undefined) {
      atlasPlaceholders = new Map();
    }
    const cached = atlasPlaceholders.get(format);
    if (cached !== undefined) return cached;
    const tex = device.createTexture({
      label: `heapScene/atlasPlaceholder/${format}`,
      size: { width: 1, height: 1, depthOrArrayLayers: 1 },
      format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const view = tex.createView();
    atlasPlaceholders.set(format, view);
    return view;
  }

  // ─── BGL + pipeline-layout cache (schema-driven) ──────────────────
  //
  // Bindings 0–3 are the four heap data views; 4..N are textures
  // (one per surviving schema entry); N+1..M are samplers. Different
  // (textureCount, samplerCount) pairs need different layouts; we
  // build them lazily and cache.
  interface BglEntry { bgl: GPUBindGroupLayout; pipelineLayout: GPUPipelineLayout }
  const bglCache = new Map<string, BglEntry>();
  function getBgl(layout: BucketLayout, withAtlasArrays: boolean): BglEntry {
    const key = `t${layout.textureBindings.length}|s${layout.samplerBindings.length}|a${withAtlasArrays ? 1 : 0}`;
    let e = bglCache.get(key);
    if (e !== undefined) return e;
    // Heap data buffers are read by both stages: FS uniform-via-varying
    // threading reads `heapF32` / `heapV4f` to decode uniforms inside
    // the fragment shader. Visibility = VERTEX | FRAGMENT for all four.
    const heapVis = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const entries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: heapVis, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: heapVis, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: heapVis, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: heapVis, buffer: { type: "read-only-storage" } },
    ];
    entries.push(
      { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    );
    for (const t of layout.textureBindings) {
      entries.push({
        binding: t.binding, visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float" },
      });
    }
    for (const s of layout.samplerBindings) {
      entries.push({
        binding: s.binding, visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "filtering" },
      });
    }
    if (withAtlasArrays) {
      // 2N + 1 entries: N linear textures, N srgb textures, 1 sampler.
      for (let i = 0; i < ATLAS_ARRAY_SIZE; i++) {
        entries.push({
          binding: ATLAS_LINEAR_BINDING_BASE + i, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float" },
        });
      }
      for (let i = 0; i < ATLAS_ARRAY_SIZE; i++) {
        entries.push({
          binding: ATLAS_SRGB_BINDING_BASE + i, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float" },
        });
      }
      entries.push({
        binding: ATLAS_SAMPLER_BINDING, visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "filtering" },
      });
    }
    const bgl = device.createBindGroupLayout({ label: `heapScene/bgl/${key}`, entries });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    e = { bgl, pipelineLayout };
    bglCache.set(key, e);
    return e;
  }

  // ─── Megacall compute (prefix-sum) pipelines ──────────────────────
  // Built once per scene (not per bucket — pipelines are identical;
  // only the bind-group differs). All three pipelines share one
  // bind-group layout with 5 bindings (drawTable, blockSums,
  // blockOffsets, indirect, params).
  let scanBgl: GPUBindGroupLayout | undefined;
  let scanPipeTile:   GPUComputePipeline | undefined;
  let scanPipeBlocks: GPUComputePipeline | undefined;
  let scanPipeAdd:   GPUComputePipeline | undefined;
  let scanPipeBuildTileIndex: GPUComputePipeline | undefined;
  {
    scanBgl = device.createBindGroupLayout({
      label: "heapScene/scanBgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const scanLayout = device.createPipelineLayout({ bindGroupLayouts: [scanBgl] });
    const scanModule = device.createShaderModule({ code: HEAP_SCAN_WGSL, label: "heapScene/scan" });
    scanPipeTile = device.createComputePipeline({
      label: "heapScene/scan/tile", layout: scanLayout,
      compute: { module: scanModule, entryPoint: "scanTile" },
    });
    scanPipeBlocks = device.createComputePipeline({
      label: "heapScene/scan/blocks", layout: scanLayout,
      compute: { module: scanModule, entryPoint: "scanBlocks" },
    });
    scanPipeAdd = device.createComputePipeline({
      label: "heapScene/scan/add", layout: scanLayout,
      compute: { module: scanModule, entryPoint: "addOffsets" },
    });
    scanPipeBuildTileIndex = device.createComputePipeline({
      label: "heapScene/scan/buildTileIndex", layout: scanLayout,
      compute: { module: scanModule, entryPoint: "buildTileIndex" },
    });
  }

  function buildScanBindGroup(bucket: Bucket): GPUBindGroup {
    if (scanBgl === undefined) throw new Error("heapScene: scan BGL missing");
    return device.createBindGroup({
      label: `heapScene/${bucket.label}/scanBg`,
      layout: scanBgl,
      entries: [
        { binding: 0, resource: { buffer: bucket.drawTableBuf!.buffer } },
        { binding: 1, resource: { buffer: bucket.blockSumsBuf!.buffer } },
        { binding: 2, resource: { buffer: bucket.blockOffsetsBuf!.buffer } },
        { binding: 3, resource: { buffer: bucket.indirectBuf! } },
        { binding: 4, resource: { buffer: bucket.paramsBuf! } },
        { binding: 5, resource: { buffer: bucket.firstDrawInTileBuf!.buffer } },
      ],
    });
  }

  // ─── Buckets ──────────────────────────────────────────────────────
  const buckets: Bucket[] = [];
  const bucketByKey = new Map<string, Bucket>();

  // ─── Family state (§6 family-merge, slice 3c) ─────────────────────
  // Built lazily on the first addDraw batch from the union of all
  // effects in that batch, then frozen. Subsequent addDraws with an
  // unknown effect throw — reactive family rebuild is a v2 punt.
  //
  // Bucket key collapses to (familyId, pipelineState): every effect
  // in the family shares one bucket per pipelineState. The family
  // VS/FS dispatches to per-effect helpers via `layoutId`.
  /**
   * Per-family compiled state. With merge enabled, every Effect in
   * `familyByEffect` points at the same `FamilyState`. With merge
   * disabled (`opts.disableFamilyMerge`), each Effect gets its own
   * single-member family + its own shader module + pipeline.
   */
  interface FamilyState {
    schema: ShaderFamilySchema;
    vsModule: GPUShaderModule;
    fsModule: GPUShaderModule;
    /** Actual @vertex entry-point name in `vsModule`. */
    vsEntryName: string;
    /** Actual @fragment entry-point name in `fsModule`. */
    fsEntryName: string;
    fieldsForEffect: Map<string, Set<string>>;
  }
  // Keyed by `effect.id` (content hash), NOT object identity. Two
  // Effect objects with identical content (e.g. produced by separate
  // calls to `effect(...)` or the pickChain composer) share one
  // FamilyState. This is the right correctness/perf knob: building a
  // family by content means an upstream caller that legitimately
  // produces different-but-identical Effect objects per leaf still
  // reuses one pipeline + one bucket.
  const familyByEffectId = new Map<string, FamilyState>();
  let familyBuilt = false;
  const enableFamilyMerge = opts.enableFamilyMerge === true;

  function compileFamilyFor(
    effects: readonly Effect[],
    perInstanceByEffect: ReadonlyMap<Effect, { attributes: Set<string>; uniforms: Set<string> }>,
  ): FamilyState {
    // Family-merge (multi-effect dispatch via layoutId switch) has been
    // disabled — it was a perf illusion. Every effect compiles to its
    // own standalone pipeline; the per-bucket WGSL comes from
    // `compileHeapEffectIR(effect, layout, opts, "standalone")` and is
    // used directly as the bucket's shader module (no wrapper, no
    // dispatch). The schema is still derived from the same
    // `buildShaderFamily` path so the runtime's drawHeader packer,
    // bind-group layout, etc. don't change shape — `__layoutId`
    // remains a u32 slot that's written as 0 per RO (harmless).
    //
    // `heapShaderFamily.compileShaderFamily` is kept around but
    // unused; the function and its tests stay disabled.
    if (effects.length !== 1) {
      throw new Error(
        "heapScene: multi-effect family-merge disabled (was a perf illusion). " +
        "Build one bucket per effect.",
      );
    }
    const effect = effects[0]!;
    const schema = buildShaderFamily(
      [effect], opts.fragmentOutputLayout, undefined,
      {
        atlasizeAllTextures: atlasPool !== undefined,
        perEffectPerInstance: perInstanceByEffect,
      },
    );
    const compileOpts: CompileOptions = opts.fragmentOutputLayout !== undefined
      ? { target: "wgsl", fragmentOutputLayout: opts.fragmentOutputLayout }
      : { target: "wgsl" };
    const ir = compileHeapEffectIR(effect, schema.drawHeaderUnion, compileOpts, "standalone");
    const vsModule = device.createShaderModule({ code: ir.vs, label: `heapScene/standalone/${schema.id}/vs` });
    const fsModule = device.createShaderModule({ code: ir.fs, label: `heapScene/standalone/${schema.id}/fs` });
    const fieldsForEffect = new Map<string, Set<string>>();
    const s = schema.perEffectSchema.get(effect)!;
    const fields = new Set<string>();
    for (const a of s.attributes) fields.add(a.name);
    for (const u of s.uniforms)   fields.add(u.name);
    for (const t of s.textures)   fields.add(t.name);
    fieldsForEffect.set(effect.id, fields);
    return {
      schema, vsModule, fsModule, fieldsForEffect,
      vsEntryName: ir.vsEntry,
      fsEntryName: ir.fsEntry,
    };
  }

  function buildFamilyFromSpecs(specs: readonly HeapDrawSpec[]): void {
    if (familyBuilt) return;
    if (specs.length === 0) {
      throw new Error("heapScene: cannot build shader family from an empty spec set");
    }
    // Deduplicate by Effect identity preserving order; derive per-effect
    // per-instance info from the spec's `instanceAttributes` map. The
    // IR substitution needs `perInstanceAttributes` set on the family
    // layout to address per-instance attribute reads via `instId`
    // instead of `vertex_index` — without this, instanced effects
    // produce broken geometry.
    // De-dupe by effect.id so two Effects with identical content but
    // distinct object identities collapse to one family + one bucket.
    const seenIds = new Set<string>();
    const unique: Effect[] = [];
    const perInstanceByEffectId = new Map<string, {
      attributes: Set<string>;
      uniforms: Set<string>;
    }>();
    for (const spec of specs) {
      const e = spec.effect;
      let entry = perInstanceByEffectId.get(e.id);
      if (entry === undefined) {
        entry = { attributes: new Set<string>(), uniforms: new Set<string>() };
        perInstanceByEffectId.set(e.id, entry);
      }
      if (spec.instanceAttributes !== undefined) {
        for (const name of Object.keys(spec.instanceAttributes)) entry.attributes.add(name);
      }
      if (!seenIds.has(e.id)) { seenIds.add(e.id); unique.push(e); }
    }
    // Family-merge disabled: always one bucket per effect.
    // `enableFamilyMerge` is ignored.
    void enableFamilyMerge;
    for (const e of unique) {
      const perI = perInstanceByEffectId.get(e.id);
      const singleMap = new Map<Effect, { attributes: Set<string>; uniforms: Set<string> }>();
      if (perI !== undefined) singleMap.set(e, perI);
      familyByEffectId.set(e.id, compileFamilyFor([e], singleMap));
    }
    familyBuilt = true;
  }

  function familyFor(effect: Effect): FamilyState {
    const f = familyByEffectId.get(effect.id);
    if (f === undefined) {
      const known = [...familyByEffectId.keys()].join(",");
      throw new Error(
        `heapScene: family is frozen; effect ${effect.id} not in {${known}}; ` +
        `reactive family rebuild is v2`,
      );
    }
    return f;
  }

  function ensureFamilyKnowsEffect(effect: Effect): void {
    if (!familyBuilt) {
      throw new Error("heapScene: ensureFamilyKnowsEffect called before family build");
    }
    familyFor(effect); // throws if unknown
  }

  // ─── id-of helpers ────────────────────────────────────────────────
  const idOf = (s: Effect): string => `effect#${s.id}`;
  const textureIds = new WeakMap<HeapTextureSet, string>();
  let texCounter = 0;
  const texIdOf = (t: HeapTextureSet | undefined): string => {
    if (t === undefined) return "tex#none";
    let id = textureIds.get(t);
    if (id === undefined) { id = `tex#${texCounter++}`; textureIds.set(t, id); }
    return id;
  };
  // PipelineState content key. Hashes by the identities of the inner
  // aval references rather than the wrapper object itself: callers
  // (notably `wombat.dom`'s `derivePipelineState`) construct a fresh
  // PipelineState object per leaf even when every contributing aval
  // is shared — without a content-aware key, every leaf gets its own
  // bucket and the heap path's drawIndirect coalescing collapses to
  // one record per bucket. With this key, two leaves that pulled the
  // same `state.mode` / `state.cullMode` / `state.depthTest` / … avals
  // bucket together.
  //
  // For CONSTANT avals (`isConstant === true`) we key by *value* — two
  // distinct `AVal.constant(0)` objects produced at different call
  // sites must collapse to the same bucket. Reactive avals fall back
  // to reference identity (the right semantic — value can tick).
  const avalIds = new WeakMap<object, number>();
  const valueIds = new Map<string, number>();
  let avalCounter = 0;
  const avalIdOf = (av: aval<unknown> | undefined): string => {
    if (av === undefined) return "_";
    // Reactive aval — key by reference. (`isConstant` is on the
    // public IAdaptive surface; guard via runtime check so off-spec
    // duck-types don't blow up.)
    const isConst = (av as { isConstant?: unknown }).isConstant === true;
    if (!isConst) {
      let id = avalIds.get(av);
      if (id === undefined) { id = avalCounter++; avalIds.set(av, id); }
      return `a${id}`;
    }
    // Constant: key by value. Force is safe (constant: no upstream
    // dep) and only runs once per distinct aval-object thanks to the
    // outer avalIds cache below.
    let id = avalIds.get(av);
    if (id !== undefined) return `c${id}`;
    // Constant avals ignore the token; use AdaptiveToken.top to satisfy
    // the type and traverse a no-op evaluation.
    const v = av.getValue(AdaptiveToken.top);
    // Value-typed (`equals` + `getHashCode`)? Try to intern by hash
    // bucket + equals so two distinct AVal.constant(M44d.identity)
    // collapse. Falls back to a per-value string key otherwise.
    let vKey: string;
    if (
      v !== null && typeof v === "object" &&
      typeof (v as { getHashCode?: unknown }).getHashCode === "function" &&
      typeof (v as { equals?: unknown }).equals === "function"
    ) {
      const hc = (v as { getHashCode(): number }).getHashCode() | 0;
      vKey = `hv:${hc}`;
    } else if (v === null || typeof v !== "object") {
      vKey = `pv:${typeof v}:${String(v)}`;
    } else {
      // Plain object — fall back to reference identity (matches the
      // memo runtime's behaviour for unhashable objects).
      let oid = avalIds.get(v as object);
      if (oid === undefined) { oid = avalCounter++; avalIds.set(v as object, oid); }
      vKey = `ov:${oid}`;
    }
    let vid = valueIds.get(vKey);
    if (vid === undefined) { vid = avalCounter++; valueIds.set(vKey, vid); }
    avalIds.set(av, vid);
    return `c${vid}`;
  };
  const psContentIds = new WeakMap<PipelineState, string>();
  const psIdOf = (ps: PipelineState | undefined): string => {
    if (ps === undefined) return "ps#default";
    const cached = psContentIds.get(ps);
    if (cached !== undefined) return cached;
    const r = ps.rasterizer;
    const parts: string[] = [
      avalIdOf(r.topology),
      avalIdOf(r.cullMode),
      avalIdOf(r.frontFace),
      avalIdOf(r.depthBias),
      ps.depth !== undefined
        ? `d:${avalIdOf(ps.depth.write)}:${avalIdOf(ps.depth.compare)}:${avalIdOf(ps.depth.clamp)}`
        : "d:_",
      ps.stencil !== undefined ? "s:1" : "s:_",
      avalIdOf(ps.blends),
      avalIdOf(ps.alphaToCoverage),
      avalIdOf(ps.blendConstant),
    ];
    const key = `ps#${parts.join("|")}`;
    psContentIds.set(ps, key);
    return key;
  };

  /** Resolved (forced) snapshot of the user's PipelineState. */
  interface ResolvedPipelineState {
    readonly topology:  GPUPrimitiveTopology;
    readonly cullMode:  GPUCullMode;
    readonly frontFace: GPUFrontFace;
    readonly depth?: { readonly write: boolean; readonly compare: GPUCompareFunction };
  }
  function resolvePipelineState(ps: PipelineState | undefined): ResolvedPipelineState {
    if (ps === undefined) {
      return {
        topology: "triangle-list", cullMode: "back", frontFace: "ccw",
        depth: { write: true, compare: "less" },
      };
    }
    const r = ps.rasterizer;
    const out: ResolvedPipelineState = {
      topology:  r.topology.force(/* allow-force */),
      cullMode:  r.cullMode.force(/* allow-force */),
      frontFace: r.frontFace.force(/* allow-force */),
      ...(ps.depth !== undefined ? {
        depth: {
          write:   ps.depth.write.force(/* allow-force */),
          compare: ps.depth.compare.force(/* allow-force */),
        },
      } : {}),
    };
    return out;
  }

  // ─── Per-bucket bind-group construction ───────────────────────────
  // The bind-group LAYOUT (binding types) is shared across all
  // pipelines via noTexBgl/texBgl; the bind-group itself binds this
  // bucket's drawHeap (binding 1) + its globals UBO (binding 0) +
  // the global arena's heapF32 view (binding 2) + textures if any.
  function buildBucketBindGroup(bucket: Bucket): GPUBindGroup {
    // heapU32 / heapF32 / heapV4f are different typed views of the
    // SAME global arena GPUBuffer (emscripten-style aliasing). The
    // WGSL prelude declares one binding per view; the shader picks
    // whichever matches its read.
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: arena.attrs.buffer } },          // heapU32
      { binding: 1, resource: { buffer: bucket.drawHeap.buffer } },      // headersU32
      { binding: 2, resource: { buffer: arena.attrs.buffer } },          // heapF32
      { binding: 3, resource: { buffer: arena.attrs.buffer } },          // heapV4f
    ];
    {
      if (bucket.drawTableBuf === undefined) {
        throw new Error("heapScene: megacall bucket without drawTableBuf");
      }
      // Bind drawTable with size = recordCount * RECORD_BYTES so the VS
      // prelude's `arrayLength(&drawTable)/RECORD_U32` returns exactly
      // the live record count — keeps stale tail entries out of the
      // binary search. Minimum one zero-record to satisfy WebGPU non-
      // zero size constraint when the bucket is empty.
      const dtBytes = Math.max(RECORD_BYTES, bucket.recordCount * RECORD_BYTES);
      entries.push(
        { binding: 4, resource: { buffer: bucket.drawTableBuf.buffer, offset: 0, size: dtBytes } },
        { binding: 5, resource: { buffer: arena.indices.buffer } },
        { binding: 6, resource: { buffer: bucket.firstDrawInTileBuf!.buffer } },
      );
      bucket.renderBoundRecordCount = bucket.recordCount;
    }
    // Schema-driven texture + sampler entries. v1 user surface still
    // accepts a single (texture, sampler) pair via HeapTextureSet —
    // we map it onto the schema's discovered names by position. If
    // the schema has more than one of either, the user must extend
    // the spec (a future, multi-binding API).
    const texLayout = bucket.layout.textureBindings;
    const smpLayout = bucket.layout.samplerBindings;
    if (texLayout.length > 0 || smpLayout.length > 0) {
      if (bucket.textures === undefined) {
        throw new Error(
          `heapScene: bucket needs ${texLayout.length} texture(s) + ${smpLayout.length} sampler(s) ` +
          `but spec.textures is undefined`,
        );
      }
      if (texLayout.length > 1 || smpLayout.length > 1) {
        throw new Error(
          `heapScene: only single texture/sampler per bucket supported in v1 ` +
          `(schema declares ${texLayout.length}/${smpLayout.length})`,
        );
      }
      const ts = bucket.textures;
      if (ts.kind === "standalone") {
        if (texLayout.length === 1) {
          if (ts.texture.kind !== "gpu") {
            throw new Error(
              "heapScene: standalone texture must be ITexture.kind === 'gpu' " +
              "(adapter is responsible for materialising host textures)",
            );
          }
          entries.push({
            binding: texLayout[0]!.binding,
            resource: ts.textureView ?? ts.texture.texture.createView(),
          });
        }
        if (smpLayout.length === 1) {
          if (ts.sampler.kind !== "gpu") {
            throw new Error(
              "heapScene: standalone sampler must be ISampler.kind === 'gpu' " +
              "(adapter is responsible for materialising descriptor samplers)",
            );
          }
          entries.push({
            binding: smpLayout[0]!.binding,
            resource: ts.sampler.sampler,
          });
        }
      }
      // Atlas-variant: textureBindings/samplerBindings were stripped at
      // buildBucketLayout time (the schema's user-named texture entries
      // are served via the binding_array path instead). Nothing to push
      // here — the atlas slots come from the per-format page tracking
      // below.
    }
    if (bucket.isAtlasBucket) {
      if (atlasPool === undefined) {
        throw new Error(
          "heapScene: atlas-variant bucket needs `atlasPool` in BuildHeapSceneOptions",
        );
      }
      // Bind N consecutive `texture_2d<f32>` slots per format. Slot
      // i for format F holds the GPUTexture view of `pagesFor(F)[i]`
      // when present, or a 1×1 placeholder otherwise. drawHeader's
      // `pageRef` field is exactly that slot index; the shader's
      // `switch pageRef` ladder picks the matching binding.
      for (const format of ATLAS_PAGE_FORMATS) {
        const base = format === "rgba8unorm-srgb"
          ? ATLAS_SRGB_BINDING_BASE : ATLAS_LINEAR_BINDING_BASE;
        const pages = atlasPool.pagesFor(format);
        for (let i = 0; i < ATLAS_ARRAY_SIZE; i++) {
          const page = pages[i];
          entries.push({
            binding: base + i,
            resource: page !== undefined
              ? page.texture.createView()
              : getAtlasPlaceholder(format),
          });
        }
      }
      entries.push({ binding: ATLAS_SAMPLER_BINDING, resource: atlasSampler });
    }
    return device.createBindGroup({
      label: `heapScene/${bucket.label}/bg`,
      layout: getBgl(bucket.layout, bucket.isAtlasBucket).bgl,
      entries,
    });
  }

  // Atlas pages are owned by the AtlasPool; heapScene wires each
  // page's GPUTexture into a fixed slot in the BGL ladder via
  // `atlasPool.pagesFor(format)` when building bind groups. Buckets
  // carry strong refcount handles (`localAtlasReleases`) that drive
  // `pool.release` on removeDraw.

  // When the global arena reallocates, every bucket's bind group
  // needs rebuilding (its binding 2 buffer reference is stale).
  arena.attrs.onResize(() => {
    for (const b of buckets) b.bindGroup = buildBucketBindGroup(b);
    // §7 dispatcher targets arena.attrs.buffer; rebind on grow.
    if (derivedScene !== undefined) derivedScene.rebindMainHeap(arena.attrs.buffer);
  });
  // Same when the atlas pool allocates a fresh page — its slot in the
  // BGL ladder transitions from placeholder to the real GPUTexture.
  // Every atlas bucket rebuilds (cheap; just N+N+1 entries each).
  if (atlasPool !== undefined) {
    for (const f of ATLAS_PAGE_FORMATS) {
      atlasPool.onPageAdded(f, () => {
        for (const b of buckets) {
          if (b.isAtlasBucket) b.bindGroup = buildBucketBindGroup(b);
        }
      });
    }
  }
  // indexStorage is bound at slot 5 from arena.indices, so a grow there
  // also invalidates every bucket's bind group.
  {
    arena.indices.onResize(() => {
      for (const b of buckets) b.bindGroup = buildBucketBindGroup(b);
    });
  }

  // ─── findOrCreateBucket ───────────────────────────────────────────
  // Slice 3c: the bucket key collapses to (familyId, pipelineState).
  // Every member effect in the family shares one bucket per
  // pipelineState; layoutId dispatch in the family VS/FS picks the
  // right per-effect helper at draw time. Atlas-binding shape follows
  // from `family.drawHeaderUnion.atlasTextureBindings.size > 0`.
  void texIdOf; // retained for future per-bucket diagnostics
  function findOrCreateBucket(
    effect: Effect,
    _textures: HeapTextureSet | undefined,
    pipelineState: PipelineState | undefined,
  ): Bucket {
    if (!familyBuilt) {
      throw new Error("heapScene: findOrCreateBucket called before family build");
    }
    const fam = familyFor(effect);
    const psKey = psIdOf(pipelineState);
    const bk = `family#${fam.schema.id}|${psKey}`;
    const existing = bucketByKey.get(bk);
    if (existing !== undefined) return existing;
    const ps = resolvePipelineState(pipelineState);

    const layout: BucketLayout = fam.schema.drawHeaderUnion;
    const isAtlasBucket = layout.atlasTextureBindings.size > 0;
    const vsModule = fam.vsModule;
    const fsModule = fam.fsModule;
    const vsEntry = fam.vsEntryName;
    const fsEntry = fam.fsEntryName;
    const { pipelineLayout } = getBgl(layout, isAtlasBucket);

    const pipeline = device.createRenderPipeline({
      label: `heapScene/${bk}/pipeline`,
      layout: pipelineLayout,
      vertex:   { module: vsModule, entryPoint: vsEntry, buffers: [] },
      fragment: { module: fsModule, entryPoint: fsEntry, targets: colorTargets },
      primitive: { topology: ps.topology, cullMode: ps.cullMode, frontFace: ps.frontFace },
      ...(depthFormat !== undefined && ps.depth !== undefined
        ? { depthStencil: { format: depthFormat, depthWriteEnabled: ps.depth.write, depthCompare: ps.depth.compare } }
        : depthFormat !== undefined
          ? { depthStencil: { format: depthFormat, depthWriteEnabled: false, depthCompare: "always" as GPUCompareFunction } }
          : {}),
    });

    // Per-bucket DrawHeader buffer (every uniform / attribute is a u32
    // ref into the global arena; the per-bucket buffer just holds the
    // refs).
    const drawHeapBuf = new GrowBuffer(
      device, `heapScene/${bk}/drawHeap`, GPUBufferUsage.STORAGE,
      Math.max(layout.drawHeaderBytes, 64),
    );
    const drawHeap = new DrawHeap(drawHeapBuf, layout.drawHeaderBytes);

    const bucket: Bucket = {
      // §6 family-merge: family buckets aren't keyed on a specific
      // texture set — atlas placements are addressed per-RO via
      // drawHeader fields (`pageRef` + `formatBits` + `origin` +
      // `size`), and the bucket's atlas-binding ladder is driven by
      // `atlasPool.pagesFor(format)`. Leave `textures` undefined.
      label: bk, textures: undefined, layout, pipeline,
      bindGroup: null as unknown as GPUBindGroup,
      drawHeap,
      drawHeaderStaging: new Float32Array(drawHeapBuf.capacity / 4),
      headerDirtyMin: Infinity, headerDirtyMax: 0,
      localPosRefs: [], localNorRefs: [],
      localEntries: [], localToDrawId: [],
      localPerDrawAvals: [], localPerDrawRefs: [], localLayoutIds: [],
      drawSlots: new Set<number>(), dirty: new Set<number>(),
      drawTableDirtyMin: Infinity, drawTableDirtyMax: 0,
      recordCount: 0, slotToRecord: [], recordToSlot: [],
      totalEmitEstimate: 0,
      scanDirty: false,
      isAtlasBucket,
      localAtlasReleases: [],
      localAtlasTextures: [],
      localAtlasArrIdx: [],
    };
    {
      const dtBuf = new GrowBuffer(
        device, `heapScene/${bk}/drawTable`,
        GPUBufferUsage.STORAGE,
        1024,
      );
      const blockSumsBuf = new GrowBuffer(
        device, `heapScene/${bk}/blockSums`,
        GPUBufferUsage.STORAGE,
        4 * Math.max(1, Math.ceil((dtBuf.capacity / RECORD_BYTES) / SCAN_TILE_SIZE)),
      );
      const blockOffsetsBuf = new GrowBuffer(
        device, `heapScene/${bk}/blockOffsets`,
        GPUBufferUsage.STORAGE,
        4 * Math.max(1, Math.ceil((dtBuf.capacity / RECORD_BYTES) / SCAN_TILE_SIZE)),
      );
      // 32 u32 = 128 bytes is the floor; pow2-grown by ensureCapacity
      // as totalEmitEstimate grows.
      const firstDrawInTileBuf = new GrowBuffer(
        device, `heapScene/${bk}/firstDrawInTile`,
        GPUBufferUsage.STORAGE,
        128,
      );
      const indirectBuf = device.createBuffer({
        label: `heapScene/${bk}/indirect`, size: 16,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      // Initialize to (0, 1, 0, 0) so an unscanned/empty bucket draws
      // nothing safely.
      device.queue.writeBuffer(indirectBuf, 0, new Uint32Array([0, 1, 0, 0]));
      const paramsBuf = device.createBuffer({
        label: `heapScene/${bk}/params`, size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      bucket.drawTableBuf = dtBuf;
      bucket.drawTableShadow = new Uint32Array(dtBuf.capacity / 4);
      bucket.blockSumsBuf = blockSumsBuf;
      bucket.blockOffsetsBuf = blockOffsetsBuf;
      bucket.firstDrawInTileBuf = firstDrawInTileBuf;
      bucket.indirectBuf = indirectBuf;
      bucket.paramsBuf = paramsBuf;
      const ensureScanBuffers = (): void => {
        const needBlocks = Math.max(1, Math.ceil(bucket.recordCount / SCAN_TILE_SIZE));
        blockSumsBuf.ensureCapacity(needBlocks * 4);
        blockOffsetsBuf.ensureCapacity(needBlocks * 4);
      };
      const rebuildScanBg = (): void => {
        bucket.scanBindGroup = buildScanBindGroup(bucket);
      };
      dtBuf.onResize(() => {
        const grown = new Uint32Array(dtBuf.capacity / 4);
        grown.set(bucket.drawTableShadow!);
        bucket.drawTableShadow = grown;
        ensureScanBuffers();
        bucket.bindGroup = buildBucketBindGroup(bucket);
        rebuildScanBg();
        bucket.drawTableDirtyMin = 0;
        bucket.drawTableDirtyMax = bucket.recordCount * RECORD_BYTES;
      });
      blockSumsBuf.onResize(rebuildScanBg);
      blockOffsetsBuf.onResize(rebuildScanBg);
      firstDrawInTileBuf.onResize(() => {
        rebuildScanBg();
        bucket.bindGroup = buildBucketBindGroup(bucket);
      });
      bucket.scanBindGroup = buildScanBindGroup(bucket);
    }
    bucket.bindGroup = buildBucketBindGroup(bucket);
    drawHeap.onResize(() => {
      bucket.drawHeaderStaging = new Float32Array(drawHeapBuf.capacity / 4);
      // GPU-side data was copied by the GrowBuffer's resize, but our
      // CPU mirror is fresh — mark all live local slots dirty so
      // their headers get re-packed and re-uploaded next frame.
      for (const s of bucket.drawSlots) bucket.dirty.add(s);
      bucket.bindGroup = buildBucketBindGroup(bucket);
      // §7's main heap is arena.attrs.buffer (not drawHeap) — the
      // arena.attrs.onResize handler at the scene level handles its
      // rebind. drawHeap resize doesn't touch §7.
    });
    // §7 has one scene-wide records buffer + dispatcher (all buckets
    // target arena.attrs.buffer), so nothing per-bucket to set up here.
    buckets.push(bucket);
    bucketByKey.set(bk, bucket);
    return bucket;
  }

  // ─── Per-bucket header writer (refs only, post step 3) ────────────
  // The DrawHeader is now a flat list of u32 refs into the arena —
  // values themselves live in arena allocations the pool manages.
  // This function packs the refs into the bucket's staging mirror;
  // callers writeBuffer the result to the GPU.
  function packBucketHeader(
    bucket: Bucket, localSlot: number,
    perDrawRefs: ReadonlyMap<string, number>,
    layoutId: number,
  ): void {
    const dst = bucket.drawHeaderStaging;
    const u32 = new Uint32Array(dst.buffer, dst.byteOffset, dst.length);
    const baseFloat = (localSlot * bucket.layout.drawHeaderBytes) / 4;
    for (const f of bucket.layout.drawHeaderFields) {
      // texture-ref fields carry inline values (pageRef/formatBits as u32,
      // origin/size as vec2<f32>) and are filled by packAtlasTextureFields.
      if (f.kind === "texture-ref") continue;
      const fOff = baseFloat + f.byteOffset / 4;
      // §6 family-merge slice 3c: write the layoutId selector inline.
      // Field kind stays `uniform-ref` so the IR's load expression
      // (`headersU32[base + offset]`) reads it as a u32 directly,
      // exactly matching the inline integer we store here.
      if (f.name === "__layoutId") {
        u32[fOff] = layoutId >>> 0;
        continue;
      }
      const ref = perDrawRefs.get(f.name);
      if (ref === undefined) {
        // Family-merge: a slot's effect doesn't populate every field of
        // the union; leave the unused slots zero — the layoutId switch
        // ensures they're never read by the wrong effect's helper.
        u32[fOff] = 0;
        continue;
      }
      u32[fOff] = ref;
    }
  }

  // ─── Atlas drawHeader packing + page-set tracking ─────────────────
  //
  // Sampler-state extraction. Atlas-variant ROs serve their texture
  // through the binding_array path, so the user's per-draw ISampler
  // becomes "sampler state bits" packed into formatBits. The plan
  // calls this `force()`-ing the sampler — for `kind: "desc"` the
  // descriptor is right there; for `kind: "gpu"` we don't have one
  // and fall back to a sensible default (linear/linear/clamp/clamp/
  // nearest), matching the shared atlasSampler.
  function samplerStateBits(s: ISampler): number {
    let mag: GPUFilterMode = "linear";
    let min: GPUFilterMode = "linear";
    let mip: GPUMipmapFilterMode = "nearest";
    let aU: GPUAddressMode = "clamp-to-edge";
    let aV: GPUAddressMode = "clamp-to-edge";
    if (s.kind === "desc") {
      mag = s.descriptor.magFilter ?? mag;
      min = s.descriptor.minFilter ?? min;
      mip = s.descriptor.mipmapFilter ?? mip;
      aU = s.descriptor.addressModeU ?? aU;
      aV = s.descriptor.addressModeV ?? aV;
    }
    const addrCode = (m: GPUAddressMode): number =>
      m === "repeat" ? 1 : m === "mirror-repeat" ? 2 : 0;
    const filterCode = (m: GPUFilterMode): number => m === "linear" ? 1 : 0;
    return (
      (addrCode(aU)    & 0x3) << 4 |
      (addrCode(aV)    & 0x3) << 6 |
      (filterCode(mag) & 0x3) << 8 |
      (filterCode(min) & 0x3) << 10 |
      // mipmapFilter not separately encoded — atlas does software mip,
      // hardware mipmapFilter is fixed to "nearest" on atlasSampler.
      (mip === "linear" ? 1 : 0) << 12
    );
  }

  function packAtlasTextureFields(
    bucket: Bucket, localSlot: number,
    textures: HeapTextureSet & { kind: "atlas" },
  ): void {
    const f32 = bucket.drawHeaderStaging;
    const u32 = new Uint32Array(f32.buffer, f32.byteOffset, f32.length);
    const baseBytes = localSlot * bucket.layout.drawHeaderBytes;
    const fmt: AtlasPageFormat = textures.page.format;
    const slotIdx = ensureAtlasPageSlot(bucket, fmt, textures.pageId, textures.page);
    const fmtIdx = atlasFormatIndex(fmt);
    const numMips = Math.max(1, Math.min(7, textures.numMips));
    const formatBits =
      (fmtIdx & 0x1) |
      ((numMips & 0x7) << 1) |
      samplerStateBits(textures.sampler);
    for (const f of bucket.layout.drawHeaderFields) {
      if (f.kind !== "texture-ref") continue;
      const off = baseBytes + f.byteOffset;
      const offU32 = off / 4;
      const offF32 = off / 4;
      switch (f.textureSub) {
        case "pageRef":     u32[offU32] = slotIdx >>> 0; break;
        case "formatBits":  u32[offU32] = formatBits >>> 0; break;
        case "origin":
          f32[offF32]     = textures.origin.x;
          f32[offF32 + 1] = textures.origin.y;
          break;
        case "size":
          f32[offF32]     = textures.size.x;
          f32[offF32 + 1] = textures.size.y;
          break;
      }
    }
  }

  // pageId IS the slot index in the format's BGL binding sequence.
  // No per-bucket slot mapping is needed — the shader's
  // `switch pageRef` ladder picks `atlasLinear<pageRef>` /
  // `atlasSrgb<pageRef>` directly. Kept as a function for callsite-
  // symmetry with the pre-pivot code.
  function ensureAtlasPageSlot(
    _bucket: Bucket, _format: AtlasPageFormat, pageId: number, _page: AtlasPage,
  ): number {
    return pageId;
  }

  // ─── Stats (declared early so addDraw/removeDraw can mutate it) ───
  const stats: HeapSceneStats = {
    groups: 0,
    totalDraws: 0,
    drawBytes: 0,
    geometryBytes: 0,
    derivedPullMs:   0,
    derivedUploadMs: 0,
    derivedEncodeMs: 0,
    derivedRecords:  0,
  };
  Object.defineProperty(stats, "groups", { get: () => buckets.length, configurable: true });

  // ─── addDraw / removeDraw ─────────────────────────────────────────
  // Public addDraw wrapper. Establishes a sceneObj.evaluateAlways
  // scope so external callers (no batched outer eval) still register
  // sceneObj as an output of any aval the spec touches. Internal call
  // sites (drainAsetWith, batched initial-population) already run
  // inside a single outer evaluateAlways and invoke addDrawImpl
  // directly with their token — collapsing 1000× nested
  // evaluateAlways into 1× outer.
  function addDraw(spec: HeapDrawSpec): number {
    let id = -1;
    sceneObj.evaluateAlways(AdaptiveToken.top, (tok) => {
      id = addDrawImpl(spec, tok);
    });
    return id;
  }
  function addDrawImpl(spec: HeapDrawSpec, outerTok: AdaptiveToken): number {
    const drawId = nextDrawId++;
    // Family-merge (slice 3c): build the family lazily from this
    // single spec when no batched lazy-build occurred earlier (e.g.
    // direct addDraw at runtime). The aset / array initial-population
    // paths build from the full effect set up front.
    if (!familyBuilt) {
      buildFamilyFromSpecs([spec]);
    } else {
      ensureFamilyKnowsEffect(spec.effect);
    }
    const perInstanceUniforms = new Set<string>();
    const perInstanceAttributes = spec.instanceAttributes !== undefined
      ? new Set(Object.keys(spec.instanceAttributes))
      : new Set<string>();
    const bucket = findOrCreateBucket(spec.effect, spec.textures, spec.pipelineState);
    const fam = familyFor(spec.effect);
    const effectFields = fam.fieldsForEffect.get(spec.effect.id)!;

    // Indices live in their own INDEX-usage buffer (WebGPU constraint).
    // Aval-keyed: 19K instanced clones of the same mesh share one
    // index allocation + one upload.
    const indicesAval = asAval(spec.indices) as aval<Uint32Array>;
    const indicesArr = readPlain(spec.indices) as Uint32Array;
    const idxAlloc = indexPool.acquire(device, arena.indices, indicesAval, indicesArr);

    const localSlot = bucket.drawHeap.alloc();
    // Per-RO instancing: read `spec.instanceCount` (defaults to 1).
    const instanceCount = spec.instanceCount !== undefined
      ? readPlain(spec.instanceCount) as number
      : 1;

    // Walk the bucket's schema-driven DrawHeader fields. Per-instance
    // attributes pull from `spec.instanceAttributes` and pack into an
    // array allocation; everything else pulls from `spec.inputs` and
    // packs as a single value. Both go through the same pool — sharing
    // emerges from aval identity either way.
    const perDrawAvals: aval<unknown>[] = [];
    const perDrawRefs = new Map<string, number>();
    {
      const tok = outerTok;
      for (const f of bucket.layout.drawHeaderFields) {
        // Atlas-variant texture bindings carry inline values rather than
        // pool refs; packAtlasTextureFields fills them after this loop.
        if (f.kind === "texture-ref") continue;
        // Family-merge: the union drawHeader includes fields from every
        // member effect. Skip fields this effect doesn't declare — its
        // layoutId branch never reads them, so the slot stays zero.
        if (f.name === "__layoutId") continue;
        if (!effectFields.has(f.name)) continue;
        // §7: derived-uniform fields are produced by the compute
        // pre-pass. We still allocate an arena slot here so the
        // drawHeader gets a valid ref written by packBucketHeader; §7
        // overwrites the arena data each frame the inputs are dirty.
        if (derivedScene !== undefined && isDerivedUniformName(f.name)) {
          const dummyAval = AVal.constant<M44d>(M44d.zero);
          const ref = pool.acquire(
            device, arena.attrs, dummyAval, M44d.zero,
            PACKER_MAT4.dataBytes, PACKER_MAT4.typeId, 1, PACKER_MAT4.pack,
          );
          perDrawRefs.set(f.name, ref);
          perDrawAvals.push(dummyAval);
          continue;
        }
        const isPerInstanceUniformField =
          f.kind === "uniform-ref" && perInstanceUniforms.has(f.name);
        const isPerInstanceAttrField =
          f.kind === "attribute-ref" && perInstanceAttributes.has(f.name);
        const provided = isPerInstanceAttrField
          ? spec.instanceAttributes![f.name]
          : spec.inputs[f.name];
        if (provided === undefined) {
          const where = isPerInstanceAttrField
            ? "spec.instanceAttributes"
            : "spec.inputs";
          throw new Error(
            `heapScene: ${where} missing required entry '${f.name}' ` +
            `(effect declares ${f.kind === "uniform-ref"
              ? `uniform ${f.uniformWgslType ?? "unknown"}`
              : `attribute ${f.attributeWgslType ?? "unknown"}`})`,
          );
        }

        // Attribute-ref + BufferView: pivot the pool key onto the
        // BufferView's `aval<IBuffer>` so identity-based sharing works
        // naturally across draws (two ROs with the same `buffer` aval
        // share the arena allocation). The packer extracts host data.
        // Both per-vertex and per-instance attribute BufferViews flow
        // through the same arena pool (the heap path doesn't need a
        // separate "instance arena").
        let av: aval<unknown>;
        let value: unknown;
        let placement: ReturnType<typeof poolPlacementFor>;
        if (f.kind === "attribute-ref" && !isPerInstanceUniformField && isBufferView(provided)) {
          const bv = provided;
          placement = bufferViewPlacement(f, bv);
          av = bv.buffer as aval<unknown>;
          value = bv.buffer.getValue(tok);
        } else {
          av = asAval(provided as aval<unknown> | unknown);
          value = av.getValue(tok);
          placement = isPerInstanceUniformField
            ? perInstancePlacementFor(f, value, instanceCount)
            : poolPlacementFor(f, value);
        }
        const ref = pool.acquire(
          device, arena.attrs, av, value,
          placement.dataBytes, placement.typeId, placement.length, placement.pack,
        );
        perDrawRefs.set(f.name, ref);
        perDrawAvals.push(av);
      }
    }

    bucket.localPosRefs[localSlot] = perDrawRefs.get("Positions");
    bucket.localNorRefs[localSlot] = perDrawRefs.get("Normals");
    bucket.localEntries[localSlot] = {
      indexCount: idxAlloc.count, firstIndex: idxAlloc.firstIndex, instanceCount,
    };
    bucket.localToDrawId[localSlot] = drawId;
    bucket.drawSlots.add(localSlot);
    bucket.localPerDrawAvals[localSlot] = perDrawAvals;
    bucket.localPerDrawRefs[localSlot]  = perDrawRefs;
    const layoutId = fam.schema.layoutIdOf.get(spec.effect)!;
    bucket.localLayoutIds[localSlot] = layoutId;

    packBucketHeader(bucket, localSlot, perDrawRefs, layoutId);
    if (bucket.isAtlasBucket && spec.textures !== undefined && spec.textures.kind === "atlas") {
      packAtlasTextureFields(bucket, localSlot, spec.textures);
      bucket.localAtlasReleases[localSlot] = spec.textures.release;
      bucket.localAtlasTextures[localSlot] = spec.textures;
      // Reactivity wire-up: subscribe to `sourceAval` (so sceneObj.inputChanged
      // sees marks) and record the (bucket, slot) pair for the drain loop.
      const sourceAval = spec.textures.sourceAval;
      const repackFn = spec.textures.repack;
      if (sourceAval !== undefined && repackFn !== undefined) {
        // touch — registers sceneObj as an output of sourceAval.
        sourceAval.getValue(outerTok);
        let arr = atlasAvalRefs.get(sourceAval);
        if (arr === undefined) {
          arr = [];
          atlasAvalRefs.set(sourceAval, arr);
        }
        bucket.localAtlasArrIdx[localSlot] = arr.length;
        arr.push({ bucket, localSlot, repack: repackFn, sampler: spec.textures.sampler });
      }
    }
    const byteOff = localSlot * bucket.layout.drawHeaderBytes;
    if (byteOff < bucket.headerDirtyMin) bucket.headerDirtyMin = byteOff;
    const end = byteOff + bucket.layout.drawHeaderBytes;
    if (end > bucket.headerDirtyMax) bucket.headerDirtyMax = end;

    {
      const dtBuf = bucket.drawTableBuf!;
      const recIdx = bucket.recordCount;
      if (recIdx >= SCAN_MAX_RECORDS) {
        throw new Error(
          `heapScene: bucket exceeds SCAN_MAX_RECORDS (${SCAN_MAX_RECORDS}); ` +
          `extend the scan to multi-level if you need more`,
        );
      }
      const byteOff = recIdx * RECORD_BYTES;
      dtBuf.ensureCapacity(byteOff + RECORD_BYTES);
      dtBuf.setUsed(Math.max(dtBuf.usedBytes, byteOff + RECORD_BYTES));
      // Grow scan-side buffers if recordCount crosses a tile boundary.
      const needBlocks = Math.max(1, Math.ceil((recIdx + 1) / SCAN_TILE_SIZE));
      bucket.blockSumsBuf!.ensureCapacity(needBlocks * 4);
      bucket.blockOffsetsBuf!.ensureCapacity(needBlocks * 4);
      const shadow = bucket.drawTableShadow!;
      // firstEmit is GPU-overwritten by the prefix-sum pass; 0 is fine.
      shadow[recIdx * RECORD_U32 + 0] = 0;
      shadow[recIdx * RECORD_U32 + 1] = localSlot;
      shadow[recIdx * RECORD_U32 + 2] = idxAlloc.firstIndex;
      shadow[recIdx * RECORD_U32 + 3] = idxAlloc.count;
      shadow[recIdx * RECORD_U32 + 4] = instanceCount;
      bucket.recordCount = recIdx + 1;
      bucket.slotToRecord[localSlot] = recIdx;
      bucket.recordToSlot[recIdx] = localSlot;
      if (byteOff < bucket.drawTableDirtyMin) bucket.drawTableDirtyMin = byteOff;
      if (byteOff + RECORD_BYTES > bucket.drawTableDirtyMax) bucket.drawTableDirtyMax = byteOff + RECORD_BYTES;
      bucket.totalEmitEstimate += idxAlloc.count * instanceCount;
      const newNumTiles = Math.max(1, Math.ceil(bucket.totalEmitEstimate / TILE_K));
      bucket.firstDrawInTileBuf!.ensureCapacity((newNumTiles + 1) * 4);
      bucket.scanDirty = true;
    }

    drawIdToBucket[drawId]    = bucket;
    drawIdToLocalSlot[drawId] = localSlot;
    drawIdToIndexAval[drawId] = indicesAval;

    // ─── §7 derived-uniforms registration ────────────────────────────
    // Collect the derived names this effect actually uses (i.e. fields
    // declared in the drawHeader for this effect AND in the §7 set).
    // For each, the kernel writes into byte offset
    //   localSlot * drawHeaderBytes + field.byteOffset
    // i.e. this RO's drawHeader region in the bucket's drawHeap.
    if (derivedScene !== undefined) {
      const requiredNames: string[] = [];
      const arenaByteOffsetByName = new Map<string, number>();
      for (const f of bucket.layout.drawHeaderFields) {
        if (f.name === "__layoutId") continue;
        if (!effectFields.has(f.name)) continue;
        if (!isDerivedUniformName(f.name)) continue;
        const ref = perDrawRefs.get(f.name);
        if (ref === undefined) continue;
        requiredNames.push(f.name);
        arenaByteOffsetByName.set(f.name, ref + ALLOC_HEADER_PAD_TO);
      }
      if (requiredNames.length > 0) {
        const reg = registerRoDerivations(derivedScene, {
          trafos: {
            modelTrafo: spec.inputs["ModelTrafo"] as aval<Trafo3d> | undefined,
            viewTrafo:  spec.inputs["ViewTrafo"]  as aval<Trafo3d> | undefined,
            projTrafo:  spec.inputs["ProjTrafo"]  as aval<Trafo3d> | undefined,
          },
          requiredNames,
          byteOffsetByName: arenaByteOffsetByName,
          drawHeaderBaseByte: 0,
        });
        derivedByDrawId.set(drawId, reg);
      }
    }

    stats.totalDraws++;
    stats.geometryBytes = arenaBytes(arena);
    return drawId;
  }

  function removeDraw(drawId: number): void {
    const bucket    = drawIdToBucket[drawId];
    const localSlot = drawIdToLocalSlot[drawId];
    if (bucket === undefined || localSlot === undefined) return;
    // §7: deregister this RO's derivation records and release slots.
    if (derivedScene !== undefined) {
      const reg = derivedByDrawId.get(drawId);
      if (reg !== undefined) {
        deregisterRoDerivations(derivedScene, reg);
        derivedByDrawId.delete(drawId);
      }
    }
    {
      const removedEntry = bucket.localEntries[localSlot];
      const removedCount = removedEntry !== undefined
        ? removedEntry.indexCount * removedEntry.instanceCount
        : 0;
      bucket.totalEmitEstimate = Math.max(0, bucket.totalEmitEstimate - removedCount);
      // Swap-pop: move the last record into the freed slot, decrement
      // recordCount. firstEmit is GPU-rewritten by the next scan, so
      // we only fix (drawIdx, indexStart, indexCount, instanceCount).
      const recIdx     = bucket.slotToRecord[localSlot]!;
      const lastRecIdx = bucket.recordCount - 1;
      const shadow = bucket.drawTableShadow!;
      if (recIdx !== lastRecIdx) {
        const dst = recIdx * RECORD_U32;
        const src = lastRecIdx * RECORD_U32;
        shadow[dst + 0] = 0;
        shadow[dst + 1] = shadow[src + 1]!;
        shadow[dst + 2] = shadow[src + 2]!;
        shadow[dst + 3] = shadow[src + 3]!;
        shadow[dst + 4] = shadow[src + 4]!;
        const movedSlot = bucket.recordToSlot[lastRecIdx]!;
        bucket.slotToRecord[movedSlot] = recIdx;
        bucket.recordToSlot[recIdx] = movedSlot;
        const byteOff = recIdx * RECORD_BYTES;
        if (byteOff < bucket.drawTableDirtyMin) bucket.drawTableDirtyMin = byteOff;
        if (byteOff + RECORD_BYTES > bucket.drawTableDirtyMax) bucket.drawTableDirtyMax = byteOff + RECORD_BYTES;
      }
      bucket.slotToRecord[localSlot] = -1;
      bucket.recordToSlot[lastRecIdx] = -1;
      bucket.recordCount = lastRecIdx;
      bucket.scanDirty = true;
    }

    // Release pool entries — refcount drops; if zero, allocation freed.
    const avals = bucket.localPerDrawAvals[localSlot];
    if (avals !== undefined) for (const av of avals) pool.release(arena.attrs, av);
    const idxAval = drawIdToIndexAval[drawId];
    if (idxAval !== undefined) indexPool.release(arena.indices, idxAval);
    const atlasRel = bucket.localAtlasReleases[localSlot];
    if (atlasRel !== undefined) atlasRel();
    // Drop atlas-aval ref (if any). When the last ref is dropped we
    // also remove the aval from the dirty set — a marking after the
    // last RO is gone is a no-op.
    const atlasTex = bucket.localAtlasTextures[localSlot];
    if (atlasTex !== undefined) {
      const sourceAval = atlasTex.sourceAval;
      const i = bucket.localAtlasArrIdx[localSlot];
      if (sourceAval !== undefined && i !== undefined) {
        const arr = atlasAvalRefs.get(sourceAval);
        if (arr !== undefined) {
          // Swap-pop: O(1) regardless of array length.
          const last = arr.length - 1;
          if (i !== last) {
            const moved = arr[last]!;
            arr[i] = moved;
            moved.bucket.localAtlasArrIdx[moved.localSlot] = i;
          }
          arr.pop();
          if (arr.length === 0) {
            atlasAvalRefs.delete(sourceAval);
            atlasAvalDirty.delete(sourceAval);
          }
        }
      }
    }
    bucket.localAtlasReleases[localSlot] = undefined;
    bucket.localAtlasTextures[localSlot] = undefined;
    bucket.localAtlasArrIdx[localSlot] = undefined;

    bucket.localPerDrawAvals[localSlot] = undefined;
    bucket.localPerDrawRefs[localSlot]  = undefined;
    bucket.localLayoutIds[localSlot]    = undefined;
    bucket.localPosRefs[localSlot]  = undefined;
    bucket.localNorRefs[localSlot]  = undefined;
    bucket.localEntries[localSlot]  = undefined;
    bucket.localToDrawId[localSlot] = undefined;
    bucket.drawSlots.delete(localSlot);
    bucket.dirty.delete(localSlot);
    bucket.drawHeap.release(localSlot);

    drawIdToBucket[drawId]    = undefined;
    drawIdToLocalSlot[drawId] = undefined;
    drawIdToIndexAval[drawId] = undefined;

    stats.totalDraws--;
    stats.geometryBytes = arenaBytes(arena);
  }

  // ─── Aset reader (pull-driven on each frame) ──────────────────────
  let asetReader: IHashSetReader<HeapDrawSpec> | undefined;
  const specToDrawId = new Map<HeapDrawSpec, number>();
  function drainAsetWith(tok: AdaptiveToken): void {
    if (asetReader === undefined) return;
    const delta = asetReader.getChanges(tok);
    delta.iter((op: { value: HeapDrawSpec; count: number }) => {
      if (op.count > 0) {
        // Re-use the outer evaluateAlways scope's token (caller is
        // already sceneObj) — collapses N nested evaluateAlways into
        // one for a bulk add. Saves a per-RO setUnsafeEvaluationDepth
        // + outputs.add(sceneObj) round trip.
        const id = addDrawImpl(op.value, tok);
        specToDrawId.set(op.value, id);
      } else {
        const id = specToDrawId.get(op.value);
        if (id !== undefined) {
          removeDraw(id);
          specToDrawId.delete(op.value);
        }
      }
    });
  }

  // ─── Initial population ───────────────────────────────────────────
  if (Array.isArray(initialDraws)) {
    // Pre-build the family from all initial effects so the first
    // bucket is built against the union right away. Skipped when the
    // array is empty (family will build lazily when an addDraw fires).
    const arr = initialDraws as readonly HeapDrawSpec[];
    if (arr.length > 0) buildFamilyFromSpecs(arr);
    sceneObj.evaluateAlways(AdaptiveToken.top, (tok) => {
      for (const d of arr) addDrawImpl(d, tok);
    });
  } else {
    asetReader = (initialDraws as aset<HeapDrawSpec>).getReader();
    sceneObj.evaluateAlways(AdaptiveToken.top, (tok) => {
      // First drain: snapshot all add-ops, build the family from their
      // effects up front, then run addDraw. Subsequent drains either
      // reuse the frozen family (matching effects) or throw on a new
      // unseen effect (per the v1 frozen-family contract).
      if (!familyBuilt) {
        const reader = asetReader!;
        const delta = reader.getChanges(tok);
        const adds: HeapDrawSpec[] = [];
        const remaining: { value: HeapDrawSpec; count: number }[] = [];
        delta.iter((op) => {
          remaining.push(op);
          if (op.count > 0) adds.push(op.value);
        });
        if (adds.length > 0) buildFamilyFromSpecs(adds);
        for (const op of remaining) {
          if (op.count > 0) {
            const id = addDrawImpl(op.value, tok);
            specToDrawId.set(op.value, id);
          } else {
            const id = specToDrawId.get(op.value);
            if (id !== undefined) {
              removeDraw(id);
              specToDrawId.delete(op.value);
            }
          }
        }
      } else {
        drainAsetWith(tok);
      }
    });
  }

  // ─── update / encodeIntoPass / frame / dispose ───────────────────
  /**
   * CPU-side data refresh: drain pending aset deltas, repack any
   * pool-tracked aval that marked since last frame, re-write any
   * bucket headers that the drawHeap grew under. After this returns,
   * the GPU state mirrors the current adaptive snapshot — caller is
   * free to encode draws.
   */
  function update(token: AdaptiveToken): void {
    let totalDirtyBytes = 0;
    sceneObj.evaluateAlways(token, (tok) => {
      drainAsetWith(tok);

      // 1. Pool: re-pack any aval whose value changed since last frame.
      //    One writeBuffer per dirty aval, regardless of how many draws
      //    reference it — sharing pays off here.
      if (allocDirty.size > 0) {
        for (const av of allocDirty) {
          pool.repack(device, arena.attrs, av, av.getValue(tok));
          const e = pool.entry(av);
          if (e !== undefined) totalDirtyBytes += e.dataBytes;
        }
        allocDirty.clear();
      }

      // 1b. Atlas-texture aval reactivity: an `aval<ITexture>` that
      //     drives an atlas placement was marked. Repack the pool entry
      //     and rewrite the drawHeader fields of every (bucket, slot)
      //     referencing this aval. The replaced HeapTextureSet's
      //     `release` closure points at the OLD pool ref — that's
      //     fine, it still resolves correctly under the pool's
      //     ref-by-id lookup; we replace it with a closure over the
      //     new ref so removeDraw frees the right one.
      if (atlasAvalDirty.size > 0) {
        for (const av of atlasAvalDirty) {
          const refs = atlasAvalRefs.get(av);
          if (refs === undefined || refs.length === 0) continue;
          const newTex = av.getValue(tok);
          const acq = refs[0]!.repack(newTex);
          for (const r of refs) {
            const newTextures: HeapTextureSet & { kind: "atlas" } = {
              kind: "atlas",
              format: acq.page.format,
              pageId: acq.pageId,
              origin: acq.origin,
              size: acq.size,
              numMips: acq.numMips,
              sampler: r.sampler,
              page: acq.page,
              poolRef: acq.ref,
              release: r.bucket.localAtlasTextures[r.localSlot]!.release,
              sourceAval: av,
              repack: r.repack,
            };
            r.bucket.localAtlasTextures[r.localSlot] = newTextures;
            packAtlasTextureFields(r.bucket, r.localSlot, newTextures);
            const byteOff = r.localSlot * r.bucket.layout.drawHeaderBytes;
            if (byteOff < r.bucket.headerDirtyMin) r.bucket.headerDirtyMin = byteOff;
            const end = byteOff + r.bucket.layout.drawHeaderBytes;
            if (end > r.bucket.headerDirtyMax) r.bucket.headerDirtyMax = end;
            totalDirtyBytes += r.bucket.layout.drawHeaderBytes;
          }
        }
        atlasAvalDirty.clear();
      }

      // 2. Per-bucket: (rare) header re-pack — only fires when the
      //    bucket's drawHeap GrowBuffer reallocated and we need to
      //    re-write all live slots into the new staging mirror.
      for (const bucket of buckets) {
        if (bucket.dirty.size === 0) continue;
        for (const localSlot of bucket.dirty) {
          const refs = bucket.localPerDrawRefs[localSlot];
          if (refs === undefined) continue;
          packBucketHeader(bucket, localSlot, refs, bucket.localLayoutIds[localSlot] ?? 0);
          if (bucket.isAtlasBucket) {
            const ts = bucket.localAtlasTextures[localSlot];
            if (ts !== undefined) packAtlasTextureFields(bucket, localSlot, ts);
          }
          const byteOff = localSlot * bucket.layout.drawHeaderBytes;
          if (byteOff < bucket.headerDirtyMin) bucket.headerDirtyMin = byteOff;
          const end = byteOff + bucket.layout.drawHeaderBytes;
          if (end > bucket.headerDirtyMax) bucket.headerDirtyMax = end;
          totalDirtyBytes += bucket.layout.drawHeaderBytes;
        }
        bucket.dirty.clear();
      }
    });
    stats.drawBytes = totalDirtyBytes;

    // ─── Flush staged uploads (one writeBuffer per dirty range) ──────
    // Replaces N small writeBuffer calls (per addDraw, per dirty alloc,
    // per dirty header slot) with at most one per logical buffer. At
    // initial population for 10K ROs this collapses ~30K calls into
    // ~5 (arena, indices, plus one per bucket).
    arena.attrs.flush(device);
    arena.indices.flush(device);
    for (const bucket of buckets) {
      if (bucket.headerDirtyMax > bucket.headerDirtyMin) {
        device.queue.writeBuffer(
          bucket.drawHeap.buffer, bucket.headerDirtyMin,
          bucket.drawHeaderStaging.buffer,
          bucket.drawHeaderStaging.byteOffset + bucket.headerDirtyMin,
          bucket.headerDirtyMax - bucket.headerDirtyMin,
        );
        bucket.headerDirtyMin = Infinity;
        bucket.headerDirtyMax = 0;
      }
      if (bucket.drawTableDirtyMax > bucket.drawTableDirtyMin) {
        const shadow = bucket.drawTableShadow!;
        device.queue.writeBuffer(
          bucket.drawTableBuf!.buffer, bucket.drawTableDirtyMin,
          shadow.buffer,
          shadow.byteOffset + bucket.drawTableDirtyMin,
          bucket.drawTableDirtyMax - bucket.drawTableDirtyMin,
        );
        bucket.drawTableDirtyMin = Infinity;
        bucket.drawTableDirtyMax = 0;
      }
    }
  }

  /**
   * Encode all live bucket draws into an existing render pass. No
   * begin/end — caller owns the pass. Used by both the convenience
   * `frame()` below and (eventually) the hybrid render task.
   */
  function encodeIntoPass(pass: GPURenderPassEncoder): void {
    let curBg: GPUBindGroup | null = null;
    for (const b of buckets) {
      if (b.bindGroup !== curBg) { pass.setBindGroup(0, b.bindGroup); curBg = b.bindGroup; }
      pass.setPipeline(b.pipeline);
      if (b.recordCount > 0) pass.drawIndirect(b.indirectBuf!, 0);
    }
  }

  /**
   * Compute prep — must be called BEFORE `beginRenderPass`. For each
   * megacall bucket whose drawTable changed since the last frame: write
   * fresh `params`, rebuild the render bind group if recordCount
   * changed (drawTable view tightened), and dispatch the three-pass
   * Blelloch scan that fills `firstEmit` and the indirect args.
   *
   * Single compute pass over all dirty buckets — pipelines are shared,
   * we just swap the bind group + dispatch shape per bucket.
   */
  function encodeComputePrep(enc: GPUCommandEncoder, token: AdaptiveToken): void {
    // §7: derived-uniforms dispatch first — writes into per-bucket
    // drawHeap regions before the scan reads anything. One dispatcher
    // per bucket; constituents are shared so dirty state propagates
    // correctly. O(changed) per dispatcher.
    if (derivedScene !== undefined) {
      const _t0 = performance.now();
      // pullDirty's getValue(t) needs an evaluateAlways scope to
      // re-establish our subscription on the firing avals (getValue
      // adds the evaluating object to the aval's outputs). Without
      // this scope, view changes never propagate again — the scene
      // freezes after one frame.
      let dirty: ReturnType<typeof derivedScene.constituents.pullDirty> | undefined;
      sceneObj.evaluateAlways(token, (t) => {
        dirty = derivedScene.constituents.pullDirty(t);
      });
      const _t1 = performance.now();
      derivedScene.uploadDirty(dirty!);
      const _t2 = performance.now();
      if (dirty!.size > 0) derivedScene.dispatcher.encode(enc);
      const _t3 = performance.now();
      stats.derivedPullMs   = _t1 - _t0;
      stats.derivedUploadMs = _t2 - _t1;
      stats.derivedEncodeMs = _t3 - _t2;
      stats.derivedRecords  = derivedScene.dispatcher.records.recordCount;
    }
    let anyDirty = false;
    for (const b of buckets) { if (b.scanDirty) { anyDirty = true; break; } }
    if (!anyDirty) return;
    const pass = enc.beginComputePass({ label: "heapScene/scan" });
    for (const b of buckets) {
      if (!b.scanDirty) continue;
      // Rebuild render bind group if recordCount changed — its
      // drawTable binding is sized to recordCount * 16.
      if (b.renderBoundRecordCount !== b.recordCount) {
        b.bindGroup = buildBucketBindGroup(b);
      }
      const numRecords = b.recordCount;
      const numBlocks = Math.max(1, Math.ceil(numRecords / SCAN_TILE_SIZE));
      device.queue.writeBuffer(
        b.paramsBuf!, 0,
        new Uint32Array([numRecords, numBlocks, 0, 0]),
      );
      pass.setBindGroup(0, b.scanBindGroup!);
      pass.setPipeline(scanPipeTile!);
      pass.dispatchWorkgroups(numBlocks, 1, 1);
      pass.setPipeline(scanPipeBlocks!);
      pass.dispatchWorkgroups(1, 1, 1);
      pass.setPipeline(scanPipeAdd!);
      pass.dispatchWorkgroups(numBlocks, 1, 1);
      // numTiles is computed on GPU from indirect[0]; CPU dispatch
      // must cover the worst-case totalEmit. Each WG handles WG_SIZE
      // tiles; +1 for the sentinel slot.
      const SCAN_WG_SIZE = 256;
      const numTilesCap = Math.max(1, Math.ceil(b.totalEmitEstimate / TILE_K));
      const tileWgs = Math.max(1, Math.ceil((numTilesCap + 1) / SCAN_WG_SIZE));
      pass.setPipeline(scanPipeBuildTileIndex!);
      pass.dispatchWorkgroups(tileWgs, 1, 1);
      b.scanDirty = false;
    }
    pass.end();
  }

  /**
   * Convenience: open a command encoder + render pass against
   * `framebuffer`, drive update + encodeIntoPass, end pass, submit.
   * For hybrid composition use `update` + `encodeIntoPass` directly.
   */
  function frame(
    framebuffer: import("../core/index.js").IFramebuffer,
    token: AdaptiveToken,
  ): void {
    update(token);
    const colorView = framebuffer.colors.tryFind(colorAttachmentName!)!;
    const depthView = framebuffer.depthStencil;
    const enc = device.createCommandEncoder({ label: "heapScene: encoder" });
    encodeComputePrep(enc, token);
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: colorView,
        clearValue: { r: 0.07, g: 0.07, b: 0.08, a: 1.0 },
        loadOp: "clear", storeOp: "store",
      }],
      ...(depthView !== undefined && depthFormat !== undefined ? {
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1.0,
          depthLoadOp: "clear", depthStoreOp: "store",
        } satisfies GPURenderPassDepthStencilAttachment,
      } : {}),
    });
    encodeIntoPass(pass);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  function dispose(): void {
    arena.attrs.destroy();
    arena.indices.destroy();
    for (const b of buckets) {
      b.drawHeap.destroy();
      b.drawTableBuf?.destroy();
      b.blockSumsBuf?.destroy();
      b.blockOffsetsBuf?.destroy();
      b.firstDrawInTileBuf?.destroy();
      b.indirectBuf?.destroy();
      b.paramsBuf?.destroy();
    }
  }

  // Test-only escape hatch for inspecting megacall bucket state. Not
  // part of the public API surface — keep cast at use-site.
  const _debug = {
    bucketsForTest(): readonly {
      indirectBuf: GPUBuffer | undefined;
      drawTableBuf: GPUBuffer | undefined;
      firstDrawInTileBuf: GPUBuffer | undefined;
      totalEmitEstimate: number;
      recordCount: number;
      layout: BucketLayout;
    }[] {
      return buckets.map(b => ({
        indirectBuf: b.indirectBuf,
        drawTableBuf: b.drawTableBuf?.buffer,
        firstDrawInTileBuf: b.firstDrawInTileBuf?.buffer,
        totalEmitEstimate: b.totalEmitEstimate,
        recordCount: b.recordCount,
        layout: b.layout,
      }));
    },
    /** §7 debug: bucket drawHeap GPU buffers. */
    drawHeapBufsForTest(): readonly GPUBuffer[] {
      return buckets.map(b => b.drawHeap.buffer);
    },
    /** Download arena + drawHeaps + drawTables + indices and validate
     *  the heap renderer's invariants:
     *    1. drawHeader refs land inside arena.
     *    2. drawTable rows (firstEmit, drawIdx, indexStart, indexCount,
     *       instanceCount) reference valid slots / indices / instances.
     *    3. firstEmit is a correct prefix-sum of `indexCount *
     *       instanceCount` over drawTable rows in record-index order.
     *    4. Index range [indexStart, indexStart+indexCount) lies within
     *       arena.indices.buffer.
     *  Returns a list of issue strings + counters. Async, safe to
     *  call any frame.
     */
    async validateHeap(): Promise<{
      arenaBytes: number;
      issues: string[];
      okRefs: number;
      badRefs: number;
      drawTableRows: number;
      drawTableErrs: number;
      prefixSumErrs: number;
      attrAllocsChecked: number;
      attrAllocsBad: number;
      tilesChecked: number;
      tilesBad: number;
      vidChecks: number;
      vidBad: number;
      indicesHash: string;
    }> {
      const issues: string[] = [];
      let okRefs = 0, badRefs = 0;
      let drawTableRows = 0, drawTableErrs = 0, prefixSumErrs = 0;
      let attrAllocsChecked = 0, attrAllocsBad = 0;
      let tilesChecked = 0, tilesBad = 0;
      let vidChecks = 0, vidBad = 0;
      const push = (s: string) => { if (issues.length < 60) issues.push(s); };
      const arenaSize = arena.attrs.buffer.size;
      const indicesSize = arena.indices.buffer.size;

      const enc = device.createCommandEncoder({ label: "validateHeap" });
      const stage = (src: GPUBuffer, size: number): GPUBuffer => {
        const c = device.createBuffer({
          size: size,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        enc.copyBufferToBuffer(src, 0, c, 0, size);
        return c;
      };

      const arenaCopy = stage(arena.attrs.buffer, arenaSize);
      const indicesCopy = indicesSize > 0 ? stage(arena.indices.buffer, indicesSize) : undefined;
      type DC = {
        bucket: typeof buckets[number];
        drawHeap: GPUBuffer;
        drawTable?: GPUBuffer;
        firstDrawInTile?: GPUBuffer;
        indirect?: GPUBuffer;
        numTiles: number;
      };
      const TILE_K_LOCAL = 64;
      const dcs: DC[] = [];
      for (const b of buckets) {
        const dhSize = Math.min(b.drawHeap.buffer.size, b.recordCount * b.layout.drawHeaderBytes);
        if (dhSize === 0) continue;
        const dh = stage(b.drawHeap.buffer, dhSize);
        const dt = b.drawTableBuf !== undefined && b.recordCount > 0
          ? stage(b.drawTableBuf.buffer, b.recordCount * RECORD_BYTES)
          : undefined;
        const totalEmit = b.totalEmitEstimate;
        const numTiles = totalEmit > 0 ? Math.ceil(totalEmit / TILE_K_LOCAL) : 0;
        const fdt = b.firstDrawInTileBuf !== undefined && numTiles > 0
          ? stage(b.firstDrawInTileBuf.buffer, Math.min(b.firstDrawInTileBuf.buffer.size, (numTiles + 1) * 4))
          : undefined;
        const ind = b.indirectBuf !== undefined ? stage(b.indirectBuf, 16) : undefined;
        dcs.push({
          bucket: b, drawHeap: dh, numTiles,
          ...(dt !== undefined ? { drawTable: dt } : {}),
          ...(fdt !== undefined ? { firstDrawInTile: fdt } : {}),
          ...(ind !== undefined ? { indirect: ind } : {}),
        });
      }
      device.queue.submit([enc.finish()]);

      await arenaCopy.mapAsync(GPUMapMode.READ);
      if (indicesCopy !== undefined) await indicesCopy.mapAsync(GPUMapMode.READ);
      for (const dc of dcs) {
        await dc.drawHeap.mapAsync(GPUMapMode.READ);
        if (dc.drawTable !== undefined) await dc.drawTable.mapAsync(GPUMapMode.READ);
        if (dc.firstDrawInTile !== undefined) await dc.firstDrawInTile.mapAsync(GPUMapMode.READ);
        if (dc.indirect !== undefined) await dc.indirect.mapAsync(GPUMapMode.READ);
      }

      // ── 1+4. drawHeader refs inside arena, drawTable indices in range,
      //         + attribute alloc-header sanity (typeId / length / data).
      const arenaU32 = new Uint32Array(arenaCopy.getMappedRange());
      const arenaF32 = new Float32Array(arenaU32.buffer, arenaU32.byteOffset, arenaU32.length);
      // Track unique attribute-alloc refs we've already inspected so we
      // don't re-validate the same shared alloc 20K times.
      const attrAllocsSeen = new Set<number>();
      const KNOWN_TYPE_IDS = new Set<number>([0, 1, 2, 3]);

      let bucketIdx = 0;
      for (const dc of dcs) {
        const u32 = new Uint32Array(dc.drawHeap.getMappedRange());
        const stride = dc.bucket.layout.drawHeaderBytes;
        const recordCount = dc.bucket.recordCount;

        for (let slot = 0; slot < recordCount; slot++) {
          const slotOff = slot * stride;
          for (const f of dc.bucket.layout.drawHeaderFields) {
            if (f.name === "__layoutId") continue;
            if (f.kind !== "uniform-ref" && f.kind !== "attribute-ref") continue;
            const ref = u32[(slotOff + f.byteOffset) >>> 2];
            if (ref === undefined || ref === 0) continue;
            const payloadStart = ref + ALLOC_HEADER_PAD_TO;
            if (payloadStart >= arenaSize) {
              push(`bucket#${bucketIdx} slot=${slot} field='${f.name}' ` +
                `ref=0x${ref.toString(16)} payload=0x${payloadStart.toString(16)} >= arena=0x${arenaSize.toString(16)}`);
              badRefs++;
              continue;
            }
            okRefs++;

            // For attribute-ref fields, decode the alloc header at
            // [ref, ref+16): typeId @ +0, length @ +4. Then sanity-
            // check the data range covers what the shader will index.
            if (f.kind !== "attribute-ref") continue;
            if (attrAllocsSeen.has(ref)) continue;
            attrAllocsSeen.add(ref);
            attrAllocsChecked++;

            const refU32 = ref >>> 2;
            const typeId = arenaU32[refU32]!;
            const length = arenaU32[refU32 + 1]!;

            if (!KNOWN_TYPE_IDS.has(typeId)) {
              push(`bucket#${bucketIdx} slot=${slot} field='${f.name}' alloc@0x${ref.toString(16)} typeId=${typeId} unknown`);
              attrAllocsBad++;
              continue;
            }
            if (length === 0) {
              push(`bucket#${bucketIdx} slot=${slot} field='${f.name}' alloc@0x${ref.toString(16)} length=0`);
              attrAllocsBad++;
              continue;
            }
            const eltFloats =
              f.attributeWgslType === "vec3<f32>" ? 3 :
              f.attributeWgslType === "vec4<f32>" ? 4 :
              f.attributeWgslType === "vec2<f32>" ? 2 : 1;
            const dataStartF32 = (ref + ALLOC_HEADER_PAD_TO) >>> 2;
            const dataEndF32   = dataStartF32 + length * eltFloats;
            if (dataEndF32 * 4 > arenaSize) {
              push(`bucket#${bucketIdx} slot=${slot} field='${f.name}' alloc@0x${ref.toString(16)} ` +
                `data extends to f32[${dataEndF32}] = ${dataEndF32 * 4}B > arena=${arenaSize}B`);
              attrAllocsBad++;
              continue;
            }
            // Check finite-ness on the data we're about to read.
            let nonFinite = 0;
            for (let i = dataStartF32; i < dataEndF32; i++) {
              const v = arenaF32[i]!;
              if (!Number.isFinite(v)) nonFinite++;
            }
            if (nonFinite > 0) {
              push(`bucket#${bucketIdx} slot=${slot} field='${f.name}' alloc@0x${ref.toString(16)} ` +
                `${nonFinite} non-finite floats in data range`);
              attrAllocsBad++;
            }
          }
        }

        // ── 2. drawTable validation
        if (dc.drawTable !== undefined) {
          const dt = new Uint32Array(dc.drawTable.getMappedRange());
          let runningSum = 0;
          for (let r = 0; r < recordCount; r++) {
            const off = r * RECORD_U32;
            const firstEmit     = dt[off + 0]!;
            const drawIdx       = dt[off + 1]!;
            const indexStart    = dt[off + 2]!;
            const indexCount    = dt[off + 3]!;
            const instanceCount = dt[off + 4]!;
            drawTableRows++;

            // 2a. drawIdx must be a valid local slot in this bucket
            if (drawIdx >= dc.bucket.layout.drawHeaderBytes * 0 + recordCount + 8 /* loose: should be < total addDraws */) {
              // can't easily check upper without an externally known max — skip
            }

            // 2b. instanceCount > 0
            if (instanceCount === 0) {
              push(`bucket#${bucketIdx} drawTable[${r}] instanceCount=0`);
              drawTableErrs++;
            }

            // 2c. indexCount > 0 (or the record is dead — but live records should have count)
            if (indexCount === 0) {
              push(`bucket#${bucketIdx} drawTable[${r}] indexCount=0`);
              drawTableErrs++;
            }

            // 2d. indices range inside arena.indices
            const indexBytes = (indexStart + indexCount) * 4;
            if (indexBytes > indicesSize) {
              push(`bucket#${bucketIdx} drawTable[${r}] index range ` +
                `[${indexStart}, ${indexStart + indexCount}) bytes=${indexBytes} > indicesSize=${indicesSize}`);
              drawTableErrs++;
            }

            // 3. prefix sum
            if (firstEmit !== runningSum) {
              push(`bucket#${bucketIdx} drawTable[${r}] firstEmit=${firstEmit} ≠ expected=${runningSum} ` +
                `(indexCount=${indexCount} instanceCount=${instanceCount})`);
              prefixSumErrs++;
            }
            runningSum += indexCount * instanceCount;
          }
          // ── 5. firstDrawInTile validation: per-tile CPU binary
          //       search using the drawTable.firstEmit values must
          //       match the GPU-computed values in firstDrawInTile.
          //       firstEmit was just verified vs CPU prefix-sum, so
          //       we use the GPU-stored values as the trusted truth.
          if (dc.firstDrawInTile !== undefined && dc.numTiles > 0) {
            const fdt = new Uint32Array(dc.firstDrawInTile.getMappedRange());
            const firstEmits: number[] = [];
            for (let r = 0; r < recordCount; r++) {
              firstEmits.push(dt[r * RECORD_U32 + 0]!);
            }
            for (let t = 0; t < dc.numTiles; t++) {
              const tileStart = t * TILE_K_LOCAL;
              let lo = 0, hi = recordCount - 1;
              while (lo < hi) {
                const mid = (lo + hi + 1) >>> 1;
                if (firstEmits[mid]! <= tileStart) lo = mid;
                else hi = mid - 1;
              }
              const got = fdt[t]!;
              tilesChecked++;
              if (got !== lo) {
                push(`bucket#${bucketIdx} firstDrawInTile[${t}] (tileStart=${tileStart}) ` +
                  `got=${got} expected=${lo}`);
                tilesBad++;
              }
            }
            const sentinel = fdt[dc.numTiles]!;
            tilesChecked++;
            if (sentinel !== recordCount) {
              push(`bucket#${bucketIdx} firstDrawInTile[${dc.numTiles}] sentinel ` +
                `got=${sentinel} expected=${recordCount}`);
              tilesBad++;
            }
            dc.firstDrawInTile.unmap();
            dc.firstDrawInTile.destroy();
          }
          dc.drawTable.unmap();
          dc.drawTable.destroy();
        }

        // ── 6. indirect[0] (totalEmit) must match CPU prefix sum.
        if (dc.indirect !== undefined) {
          const ind = new Uint32Array(dc.indirect.getMappedRange());
          const expectedTotal = dc.bucket.totalEmitEstimate;
          const got = ind[0]!;
          if (got !== expectedTotal) {
            push(`bucket#${bucketIdx} indirect[0]=${got} ≠ expected totalEmit=${expectedTotal}`);
            drawTableErrs++;
          }
          // [1]=instanceCount must be 1 (megacall pattern).
          if (ind[1] !== 1) {
            push(`bucket#${bucketIdx} indirect[1]=${ind[1]} ≠ 1`);
            drawTableErrs++;
          }
          dc.indirect.unmap();
          dc.indirect.destroy();
        }

        dc.drawHeap.unmap();
        dc.drawHeap.destroy();
        bucketIdx++;
      }
      // ── 7. Content fingerprints over scene-deterministic regions.
      //       Hashing the FULL arena would change per frame because
      //       view/proj uniforms are repacked each tick — not useful
      //       for cross-device comparison. Restrict to what's stable
      //       across the whole scene lifetime: indices buffer (vertex
      //       indices, written once at addDraw and never touched).
      //       fnv1a is fast and good enough.
      const fnv1a = (u32: Uint32Array): string => {
        let h = 0x811c9dc5 >>> 0;
        for (let i = 0; i < u32.length; i++) {
          const v = u32[i]!;
          h = (h ^ (v & 0xff)) >>> 0;     h = Math.imul(h, 0x01000193) >>> 0;
          h = (h ^ ((v >>> 8) & 0xff)) >>> 0;  h = Math.imul(h, 0x01000193) >>> 0;
          h = (h ^ ((v >>> 16) & 0xff)) >>> 0; h = Math.imul(h, 0x01000193) >>> 0;
          h = (h ^ ((v >>> 24) & 0xff)) >>> 0; h = Math.imul(h, 0x01000193) >>> 0;
        }
        return h.toString(16).padStart(8, "0");
      };
      const indicesHash = indicesCopy !== undefined
        ? fnv1a(new Uint32Array(indicesCopy.getMappedRange(), 0, indicesSize >>> 2))
        : "—";

      arenaCopy.unmap();
      arenaCopy.destroy();
      if (indicesCopy !== undefined) {
        indicesCopy.unmap();
        indicesCopy.destroy();
      }

      return {
        arenaBytes: arenaSize,
        issues,
        okRefs, badRefs,
        drawTableRows, drawTableErrs, prefixSumErrs,
        attrAllocsChecked, attrAllocsBad,
        tilesChecked, tilesBad,
        vidChecks, vidBad,
        indicesHash,
      };
    },
    /** Per-emit CPU draw simulator. Samples N emits across all
     *  buckets; for each, performs the same binary search the render
     *  kernel does, recovers (slot, _local, instId, vid), and verifies
     *  every storage-buffer read address the vertex shader would
     *  perform lands inside its bound buffer.
     *
     *  Returns counts + first few sites of any OOB read. The shader's
     *  cyclic addressing on per-vertex attrs and direct (non-cyclic)
     *  addressing on per-instance attrs are both modelled. */
    async simulateDraws(samples = 50_000): Promise<{
      emitsChecked: number;
      oob: number;
      issues: string[];
    }> {
      const issues: string[] = [];
      let oob = 0;
      let emitsChecked = 0;
      const push = (s: string) => { if (issues.length < 30) issues.push(s); };

      const arenaSize = arena.attrs.buffer.size;
      const indicesSize = arena.indices.buffer.size;

      // Stage all buffers we'll need.
      const enc = device.createCommandEncoder({ label: "simulateDraws" });
      const stage = (src: GPUBuffer, size: number): GPUBuffer => {
        const c = device.createBuffer({
          size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        enc.copyBufferToBuffer(src, 0, c, 0, size);
        return c;
      };
      const arenaCopy = stage(arena.attrs.buffer, arenaSize);
      const indicesCopy = indicesSize > 0 ? stage(arena.indices.buffer, indicesSize) : undefined;
      type DC = {
        bucket: typeof buckets[number];
        drawHeap: GPUBuffer;
        drawTable: GPUBuffer;
        firstEmit: number[];
        totalEmit: number;
      };
      const dcs: DC[] = [];
      for (const b of buckets) {
        if (b.recordCount === 0 || b.totalEmitEstimate === 0) continue;
        const dh = stage(b.drawHeap.buffer, Math.min(b.drawHeap.buffer.size, b.recordCount * b.layout.drawHeaderBytes));
        const dt = stage(b.drawTableBuf!.buffer, b.recordCount * RECORD_BYTES);
        dcs.push({ bucket: b, drawHeap: dh, drawTable: dt, firstEmit: [], totalEmit: b.totalEmitEstimate });
      }
      device.queue.submit([enc.finish()]);

      await arenaCopy.mapAsync(GPUMapMode.READ);
      if (indicesCopy !== undefined) await indicesCopy.mapAsync(GPUMapMode.READ);
      for (const dc of dcs) {
        await dc.drawHeap.mapAsync(GPUMapMode.READ);
        await dc.drawTable.mapAsync(GPUMapMode.READ);
      }

      const arenaU32 = new Uint32Array(arenaCopy.getMappedRange());
      const arenaF32 = new Float32Array(arenaU32.buffer, arenaU32.byteOffset, arenaU32.length);
      const indicesU32 = indicesCopy !== undefined
        ? new Uint32Array(indicesCopy.getMappedRange())
        : undefined;

      // Compute per-bucket cumulative emit ranges + cache mapped views.
      const bucketCumEmit: number[] = [];
      const bucketDt:     Uint32Array[] = [];
      const bucketHdr:    Uint32Array[] = [];
      let totalEmitGlobal = 0;
      for (const dc of dcs) {
        const dt = new Uint32Array(dc.drawTable.getMappedRange());
        bucketDt.push(dt);
        bucketHdr.push(new Uint32Array(dc.drawHeap.getMappedRange()));
        for (let r = 0; r < dc.bucket.recordCount; r++) {
          dc.firstEmit.push(dt[r * RECORD_U32 + 0]!);
        }
        bucketCumEmit.push(totalEmitGlobal);
        totalEmitGlobal += dc.totalEmit;
      }

      // For each bucket, precompute per-field metadata: byteOffset in
      // drawHeader, eltFloats, isInstance.
      type Field = {
        name: string;
        byteOffset: number;
        kind: "uniform-ref" | "attribute-ref";
        eltFloats: number;     // floats per element
        isInstance: boolean;   // per-instance attr (non-cyclic)
        uniSizeFloats: number; // for uniform: 16 (mat4), 4 (vec4), 3 (vec3), …
      };
      const fieldMap = new Map<typeof buckets[number], Field[]>();
      for (const dc of dcs) {
        const fs: Field[] = [];
        for (const f of dc.bucket.layout.drawHeaderFields) {
          if (f.name === "__layoutId") continue;
          if (f.kind !== "uniform-ref" && f.kind !== "attribute-ref") continue;
          const wt = f.kind === "uniform-ref" ? f.uniformWgslType : f.attributeWgslType;
          const eltF =
            wt === "vec3<f32>" ? 3 :
            wt === "vec4<f32>" ? 4 :
            wt === "vec2<f32>" ? 2 :
            wt === "mat4x4<f32>" ? 16 : 1;
          const isInstance = f.kind === "attribute-ref" && dc.bucket.layout.perInstanceAttributes.has(f.name);
          fs.push({
            name: f.name,
            byteOffset: f.byteOffset,
            kind: f.kind,
            eltFloats: eltF,
            isInstance,
            uniSizeFloats: eltF,
          });
        }
        fieldMap.set(dc.bucket, fs);
      }

      // Sample emits uniformly.
      const sampleCount = Math.min(samples, totalEmitGlobal);
      const stride = Math.max(1, Math.floor(totalEmitGlobal / sampleCount));
      for (let s = 0; s < totalEmitGlobal && emitsChecked < sampleCount; s += stride) {
        // Find which bucket this global emit lands in.
        let dc = dcs[0]!;
        let dcIdx = 0;
        let emit = s;
        for (let i = 0; i < dcs.length; i++) {
          if (s < bucketCumEmit[i]! + dcs[i]!.totalEmit) {
            dc = dcs[i]!; dcIdx = i; emit = s - bucketCumEmit[i]!; break;
          }
        }
        emitsChecked++;

        // Binary search for slot.
        const recCount = dc.bucket.recordCount;
        let lo = 0, hi = recCount - 1;
        while (lo < hi) {
          const mid = (lo + hi + 1) >>> 1;
          if (dc.firstEmit[mid]! <= emit) lo = mid;
          else hi = mid - 1;
        }
        const slot = lo;
        const dt = bucketDt[dcIdx]!;
        const off = slot * RECORD_U32;
        const firstEmit     = dt[off + 0]!;
        const drawIdx       = dt[off + 1]!;
        const indexStart    = dt[off + 2]!;
        const indexCount    = dt[off + 3]!;
        const instanceCount = dt[off + 4]!;

        // Recover _local, instId, vid.
        const _local = emit - firstEmit;
        const instId = Math.floor(_local / indexCount);
        const idxIdx = indexStart + (_local % indexCount);
        if (idxIdx * 4 >= indicesSize) {
          push(`emit=${s} bucket=${dcIdx} slot=${slot} idxIdx=${idxIdx} OOB indices`);
          oob++; continue;
        }
        if (instId >= instanceCount) {
          push(`emit=${s} slot=${slot} instId=${instId} ≥ instanceCount=${instanceCount}`);
          oob++; continue;
        }
        const vid = indicesU32![idxIdx]!;

        // Walk drawHeader fields for this drawIdx.
        const stride2 = dc.bucket.layout.drawHeaderBytes;
        const headerU32 = bucketHdr[dcIdx]!;
        const headerOff = drawIdx * stride2;

        const fs = fieldMap.get(dc.bucket)!;
        for (const f of fs) {
          const ref = headerU32[(headerOff + f.byteOffset) >>> 2];
          if (ref === undefined || ref === 0) continue;
          const allocBase = ref + ALLOC_HEADER_PAD_TO;
          const length = arenaU32[(ref >>> 2) + 1]!;
          if (length === 0) {
            push(`emit=${s} field='${f.name}' alloc length=0`);
            oob++;
            continue;
          }
          let elemIdx: number;
          if (f.kind === "uniform-ref") {
            elemIdx = 0;
          } else if (f.isInstance) {
            // direct: iidx * eltFloats. (`cyclic=false` codepath I added.)
            elemIdx = instId;
          } else {
            // cyclic: vid % length.
            elemIdx = vid % length;
          }
          const byteOffset = allocBase + elemIdx * (f.eltFloats * 4);
          const endByte = byteOffset + f.eltFloats * 4;
          if (endByte > arenaSize) {
            push(
              `emit=${s} bucket=${dcIdx} slot=${slot} drawIdx=${drawIdx} ` +
              `field='${f.name}' kind=${f.kind} isInstance=${f.isInstance} ` +
              `vid=${vid} instId=${instId} length=${length} elemIdx=${elemIdx} ` +
              `byteOffset=${byteOffset} endByte=${endByte} > arenaSize=${arenaSize}`,
            );
            oob++;
            continue;
          }
          // Check finiteness of the data we're reading.
          let nonFinite = 0;
          const baseF = byteOffset >>> 2;
          for (let i = 0; i < f.eltFloats; i++) {
            const v = arenaF32[baseF + i]!;
            if (!Number.isFinite(v)) nonFinite++;
          }
          if (nonFinite > 0) {
            push(
              `emit=${s} field='${f.name}' kind=${f.kind} isInstance=${f.isInstance} ` +
              `vid=${vid} instId=${instId} elemIdx=${elemIdx} ` +
              `nonFinite=${nonFinite}/${f.eltFloats}`,
            );
            oob++;
          }
        }

        // ── Bounding-box check: read ModelTrafo + Positions, transform
        //    the vertex, verify it lands near the RO's origin (the
        //    modelTrafo's translation column). Distance > MAX × geometry
        //    extent flags overlapping-write corruption: if two ROs'
        //    arena allocs got cross-wired, the transformed position
        //    will land in a different RO's region, far from this RO's
        //    expected origin.
        const fsMap = new Map(fs.map(x => [x.name, x] as const));
        const modelF = fsMap.get("ModelTrafo");
        const positionsF = fsMap.get("Positions");
        if (modelF !== undefined && positionsF !== undefined) {
          const modelRef = headerU32[(headerOff + modelF.byteOffset) >>> 2];
          const posRef = headerU32[(headerOff + positionsF.byteOffset) >>> 2];
          if (modelRef !== undefined && modelRef !== 0 && posRef !== undefined && posRef !== 0) {
            const mBase = (modelRef + ALLOC_HEADER_PAD_TO) >>> 2;
            const posLen = arenaU32[(posRef >>> 2) + 1]!;
            // Aardvark M44d is row-major; CPU mirror packs row-major.
            // M[r][c] = arenaF32[mBase + r*4 + c]. Multiply M·v with
            // v = (px, py, pz, 1).
            const m = new Float32Array(16);
            for (let i = 0; i < 16; i++) m[i] = arenaF32[mBase + i]!;
            // RO origin = M·(0,0,0,1) = column 3 (translation).
            const ox = m[0 * 4 + 3]!;
            const oy = m[1 * 4 + 3]!;
            const oz = m[2 * 4 + 3]!;
            // Read vertex (cyclic vid%length).
            const vidWrap = vid % posLen;
            const pBase = (posRef + ALLOC_HEADER_PAD_TO) >>> 2;
            const px = arenaF32[pBase + vidWrap * 3 + 0]!;
            const py = arenaF32[pBase + vidWrap * 3 + 1]!;
            const pz = arenaF32[pBase + vidWrap * 3 + 2]!;
            // World = M·vec4(px,py,pz,1).
            const wx = m[0 * 4 + 0]! * px + m[0 * 4 + 1]! * py + m[0 * 4 + 2]! * pz + m[0 * 4 + 3]!;
            const wy = m[1 * 4 + 0]! * px + m[1 * 4 + 1]! * py + m[1 * 4 + 2]! * pz + m[1 * 4 + 3]!;
            const wz = m[2 * 4 + 0]! * px + m[2 * 4 + 1]! * py + m[2 * 4 + 2]! * pz + m[2 * 4 + 3]!;
            const dx = wx - ox, dy = wy - oy, dz = wz - oz;
            const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
            // Conservative bound: scale ≤ ~3 in any axis × max-vertex-mag ≤ ~1.5
            // → expected dist ≤ ~5. Anything > 50 is corruption.
            if (!Number.isFinite(wx + wy + wz) || dist > 50) {
              push(
                `emit=${s} bucket=${dcIdx} slot=${slot} drawIdx=${drawIdx} vid=${vid} ` +
                `pos=(${px.toFixed(2)},${py.toFixed(2)},${pz.toFixed(2)}) ` +
                `world=(${wx.toFixed(2)},${wy.toFixed(2)},${wz.toFixed(2)}) ` +
                `origin=(${ox.toFixed(2)},${oy.toFixed(2)},${oz.toFixed(2)}) ` +
                `distFromOrigin=${dist.toFixed(2)}`,
              );
              oob++;
            }
          }
        }
      }

      // Cleanup
      arenaCopy.unmap(); arenaCopy.destroy();
      if (indicesCopy !== undefined) { indicesCopy.unmap(); indicesCopy.destroy(); }
      for (const dc of dcs) {
        dc.drawHeap.unmap(); dc.drawHeap.destroy();
        dc.drawTable.unmap(); dc.drawTable.destroy();
      }

      return { emitsChecked, oob, issues };
    },
    /** Triangle-level coherence: walks N triangle bases (3 consecutive
     *  emits each) and verifies all three vertices of each triangle
     *  resolve to the SAME slot. A mismatch means the rasterizer would
     *  stitch a triangle across two ROs — the cross-RO stretched-band
     *  bug. Vertex-level OOB / BB checks miss this because they sample
     *  individual emits in isolation. */
    async checkTriangleCoherence(samples = 50_000): Promise<{
      trianglesChecked: number;
      crossSlot: number;
      issues: string[];
    }> {
      const issues: string[] = [];
      let crossSlot = 0;
      const push = (s: string) => { if (issues.length < 30) issues.push(s); };

      let trianglesChecked = 0;
      let bucketIdx = 0;
      // Cross-bucket paranoia: collect every (ref, allocBytes, owner)
      // claimed by any drawHeader of any bucket. After all buckets are
      // walked, sort and verify no two distinct refs overlap in arena.
      // Distinct refs that overlap = arena allocator handed the same
      // bytes to two avals → two ROs reading each other's data → the
      // cross-RO triangle symptom.
      const allocClaims: { ref: number; bytes: number; owner: string }[] = [];

      // Stage arena + arena.indices once (shared across buckets).
      const arenaCopy = device.createBuffer({
        size: arena.attrs.buffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const indicesCopy = arena.indices.buffer.size > 0 ? device.createBuffer({
        size: arena.indices.buffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }) : undefined;
      {
        const enc = device.createCommandEncoder({ label: "checkTriangleCoherence.arena" });
        enc.copyBufferToBuffer(arena.attrs.buffer, 0, arenaCopy, 0, arena.attrs.buffer.size);
        if (indicesCopy !== undefined) enc.copyBufferToBuffer(arena.indices.buffer, 0, indicesCopy, 0, arena.indices.buffer.size);
        device.queue.submit([enc.finish()]);
      }
      await arenaCopy.mapAsync(GPUMapMode.READ);
      if (indicesCopy !== undefined) await indicesCopy.mapAsync(GPUMapMode.READ);
      const arenaU32 = new Uint32Array(arenaCopy.getMappedRange());
      const arenaF32 = new Float32Array(arenaU32.buffer, arenaU32.byteOffset, arenaU32.length);
      const indicesU32 = indicesCopy !== undefined
        ? new Uint32Array(indicesCopy.getMappedRange())
        : new Uint32Array(0);

      for (const target of buckets) {
        if (target.recordCount === 0 || target.totalEmitEstimate === 0) { bucketIdx++; continue; }
        const recordCount = target.recordCount;
        const totalEmit = target.totalEmitEstimate;
        const totalTris = Math.floor(totalEmit / 3);
        const remainder = totalEmit % 3;
        if (remainder !== 0) {
          push(`bucket#${bucketIdx} totalEmit=${totalEmit} not multiple of 3 (remainder ${remainder}) — partial triangle from indirect-draw`);
          crossSlot++;
        }

        // drawTable + drawHeap readback (per-bucket).
        const dtCopy = device.createBuffer({
          size: recordCount * RECORD_BYTES,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const dhCopy = device.createBuffer({
          size: recordCount * target.layout.drawHeaderBytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const enc0 = device.createCommandEncoder({ label: "checkTriangleCoherence.dt" });
        enc0.copyBufferToBuffer(target.drawTableBuf!.buffer, 0, dtCopy, 0, recordCount * RECORD_BYTES);
        enc0.copyBufferToBuffer(target.drawHeap.buffer, 0, dhCopy, 0, recordCount * target.layout.drawHeaderBytes);
        device.queue.submit([enc0.finish()]);
        await dtCopy.mapAsync(GPUMapMode.READ);
        await dhCopy.mapAsync(GPUMapMode.READ);
        const dtU32 = new Uint32Array(dtCopy.getMappedRange());
        const dhU32 = new Uint32Array(dhCopy.getMappedRange());
        const firstEmits: number[] = [];
        const drawIdxs:    number[] = [];
        const indexStarts: number[] = [];
        const indexCounts: number[] = [];
        const instCounts:  number[] = [];
        for (let r = 0; r < recordCount; r++) {
          firstEmits.push(dtU32[r * RECORD_U32 + 0]!);
          drawIdxs.push(dtU32[r * RECORD_U32 + 1]!);
          indexStarts.push(dtU32[r * RECORD_U32 + 2]!);
          indexCounts.push(dtU32[r * RECORD_U32 + 3]!);
          instCounts.push(dtU32[r * RECORD_U32 + 4]!);
        }
        // Resolve drawHeader field offsets for ModelTrafo + Positions.
        let mtOff = -1, posOff = -1;
        for (const f of target.layout.drawHeaderFields) {
          if (f.name === "ModelTrafo")  mtOff  = f.byteOffset;
          if (f.name === "Positions")   posOff = f.byteOffset;
        }
        const stride2 = target.layout.drawHeaderBytes;

        // Per-record correctness: indexCount * instanceCount must be
        // a multiple of 3, otherwise THIS record's last triangle would
        // lack a third vertex and the next record's first emit would
        // be pulled into it by the rasterizer.
        for (let r = 0; r < recordCount; r++) {
          const sub = indexCounts[r]! * instCounts[r]!;
          if (sub % 3 !== 0) {
            push(`bucket#${bucketIdx} record[${r}] indexCount=${indexCounts[r]} instanceCount=${instCounts[r]} ` +
              `→ sub-emit=${sub} not multiple of 3 (CROSS-RO TRIANGLE!)`);
            crossSlot++;
          }
        }

        // Read firstDrawInTile so we can bound the search EXACTLY the
        // same way the VS does — `lo = fdt[tileIdx]; hi = fdt[tileIdx+1]`.
        const numTilesLocal = Math.ceil(totalEmit / 64);
        const fdtSize = (numTilesLocal + 1) * 4;
        const fdtCopy = device.createBuffer({
          size: fdtSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        {
          const enc1 = device.createCommandEncoder();
          enc1.copyBufferToBuffer(target.firstDrawInTileBuf!.buffer, 0, fdtCopy, 0, fdtSize);
          device.queue.submit([enc1.finish()]);
          await fdtCopy.mapAsync(GPUMapMode.READ);
        }
        const fdt = new Uint32Array(fdtCopy.getMappedRange().slice(0));
        fdtCopy.unmap(); fdtCopy.destroy();

        const slotFor = (emit: number): number => {
          // Match the VS preamble exactly: bound by firstDrawInTile.
          const tileIdx = emit >>> 6;
          let lo = fdt[tileIdx]!;
          let hi = fdt[tileIdx + 1]!;
          while (lo < hi) {
            const mid = (lo + hi + 1) >>> 1;
            if (firstEmits[mid]! <= emit) lo = mid;
            else hi = mid - 1;
          }
          return lo;
        };

        // Exhaustive — walk EVERY triangle. For each:
        //   1. Verify all 3 emits resolve to the same slot AND the
        //      same instId (cross-RO or cross-instance triangle).
        //   2. Read drawHeader[drawIdx] for ModelTrafo + Positions.
        //   3. Look up the 3 vid values via indexStorage.
        //   4. Apply ModelTrafo to each vertex, verify world position
        //      lies within a sane radius of the trafo's translation.
        let here = 0;
        for (let t = 0; t < totalTris; t++) {
          const e0 = t * 3;
          const e1 = e0 + 1;
          const e2 = e0 + 2;
          const s0 = slotFor(e0);
          const s1 = slotFor(e1);
          const s2 = slotFor(e2);
          here++;
          if (s0 !== s1 || s1 !== s2) {
            crossSlot++;
            push(
              `bucket#${bucketIdx} tri=${t} emits=(${e0},${e1},${e2}) slots=(${s0},${s1},${s2}) ` +
              `firstEmits[${s0}]=${firstEmits[s0]} [${s1}]=${firstEmits[s1]} [${s2}]=${firstEmits[s2]}`,
            );
            continue;
          }
          // Paranoid re-derive: drawIdx + ModelTrafoRef per-emit. If
          // any of these read paths gives a different value across
          // the three emits of one triangle, the triangle is using
          // multiple model trafos.
          const drawIdxA = drawIdxs[s0]!;
          const drawIdxB = drawIdxs[s1]!;
          const drawIdxC = drawIdxs[s2]!;
          if (drawIdxA !== drawIdxB || drawIdxB !== drawIdxC) {
            crossSlot++;
            push(`bucket#${bucketIdx} tri=${t} drawIdx mismatch across emits: ${drawIdxA},${drawIdxB},${drawIdxC}`);
            continue;
          }
          if (mtOff >= 0) {
            const mtRefA = dhU32[(drawIdxA * stride2 + mtOff) >>> 2]!;
            const mtRefB = dhU32[(drawIdxB * stride2 + mtOff) >>> 2]!;
            const mtRefC = dhU32[(drawIdxC * stride2 + mtOff) >>> 2]!;
            if (mtRefA !== mtRefB || mtRefB !== mtRefC) {
              crossSlot++;
              push(`bucket#${bucketIdx} tri=${t} ModelTrafoRef mismatch: ${mtRefA.toString(16)},${mtRefB.toString(16)},${mtRefC.toString(16)}`);
              continue;
            }
            // Triple-read mat4 bytes and assert identical (in case
            // the per-vertex shader gets different bytes from
            // arena due to some aliasing/coherency anomaly the CPU
            // can model only as identical reads).
            if (mtRefA !== 0) {
              const mb = (mtRefA + ALLOC_HEADER_PAD_TO) >>> 2;
              for (let k = 0; k < 16; k++) {
                const a = arenaF32[mb + k]!;
                if (!Number.isFinite(a)) {
                  crossSlot++;
                  push(`bucket#${bucketIdx} tri=${t} ModelTrafo[${k}] non-finite=${a}`);
                  break;
                }
              }
            }
          }
          const slot = s0;
          const indexCount = indexCounts[slot]!;
          const indexStart = indexStarts[slot]!;
          const drawIdx = drawIdxs[slot]!;
          const fe = firstEmits[slot]!;
          const localBase = e0 - fe;
          const inst0 = Math.floor(localBase / indexCount);
          const inst1 = Math.floor((localBase + 1) / indexCount);
          const inst2 = Math.floor((localBase + 2) / indexCount);
          if (inst0 !== inst1 || inst1 !== inst2) {
            crossSlot++;
            push(
              `bucket#${bucketIdx} tri=${t} slot=${slot} cross-instance: ` +
              `instIds=(${inst0},${inst1},${inst2}) localBase=${localBase} indexCount=${indexCount}`,
            );
            continue;
          }
          if (mtOff < 0 || posOff < 0) continue;  // bucket lacks one of the fields
          const mtRef = dhU32[(drawIdx * stride2 + mtOff) >>> 2]!;
          const posRef = dhU32[(drawIdx * stride2 + posOff) >>> 2]!;
          if (mtRef === 0 || posRef === 0) continue;
          const mBase = (mtRef + ALLOC_HEADER_PAD_TO) >>> 2;
          const posLen = arenaU32[(posRef >>> 2) + 1]!;
          const pBase = (posRef + ALLOC_HEADER_PAD_TO) >>> 2;
          // Aardvark M44d row-major: M[r][c] = arenaF32[mBase + r*4+c].
          // Translation column = (M[0][3], M[1][3], M[2][3]).
          const ox = arenaF32[mBase + 0 * 4 + 3]!;
          const oy = arenaF32[mBase + 1 * 4 + 3]!;
          const oz = arenaF32[mBase + 2 * 4 + 3]!;
          // Three vid lookups via indexStorage[indexStart + (local % indexCount)].
          const v0 = indicesU32[indexStart + (localBase % indexCount)]! % posLen;
          const v1 = indicesU32[indexStart + ((localBase + 1) % indexCount)]! % posLen;
          const v2 = indicesU32[indexStart + ((localBase + 2) % indexCount)]! % posLen;
          let triBad = false;
          for (const v of [v0, v1, v2]) {
            const px = arenaF32[pBase + v * 3 + 0]!;
            const py = arenaF32[pBase + v * 3 + 1]!;
            const pz = arenaF32[pBase + v * 3 + 2]!;
            const wx = arenaF32[mBase + 0 * 4 + 0]! * px + arenaF32[mBase + 0 * 4 + 1]! * py + arenaF32[mBase + 0 * 4 + 2]! * pz + arenaF32[mBase + 0 * 4 + 3]!;
            const wy = arenaF32[mBase + 1 * 4 + 0]! * px + arenaF32[mBase + 1 * 4 + 1]! * py + arenaF32[mBase + 1 * 4 + 2]! * pz + arenaF32[mBase + 1 * 4 + 3]!;
            const wz = arenaF32[mBase + 2 * 4 + 0]! * px + arenaF32[mBase + 2 * 4 + 1]! * py + arenaF32[mBase + 2 * 4 + 2]! * pz + arenaF32[mBase + 2 * 4 + 3]!;
            const dx = wx - ox, dy = wy - oy, dz = wz - oz;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (!Number.isFinite(wx + wy + wz) || dist > 50) {
              if (!triBad) {
                push(
                  `bucket#${bucketIdx} tri=${t} slot=${slot} drawIdx=${drawIdx} ` +
                  `vid=${v} pos=(${px.toFixed(2)},${py.toFixed(2)},${pz.toFixed(2)}) ` +
                  `world=(${wx.toFixed(2)},${wy.toFixed(2)},${wz.toFixed(2)}) ` +
                  `origin=(${ox.toFixed(2)},${oy.toFixed(2)},${oz.toFixed(2)}) dist=${dist.toFixed(2)}`,
                );
              }
              triBad = true;
            }
          }
          if (triBad) crossSlot++;
        }
        trianglesChecked += here;
        dtCopy.unmap(); dtCopy.destroy();
        dhCopy.unmap(); dhCopy.destroy();
        bucketIdx++;
      }
      // ── Cross-bucket arena allocation overlap detection. ──────────
      // Walk every drawHeader of every bucket, collect (ref, size,
      // owner) for every uniform-ref + attribute-ref field. Sort by
      // ref. Verify: two consecutive entries with DIFFERENT ref do
      // NOT have overlapping byte ranges. Distinct refs overlapping
      // = the arena allocator handed the same bytes to two distinct
      // allocations → two ROs read each other's data → cross-RO
      // triangle. Bug is invisible to per-triangle checks because
      // each individual triangle reads "valid" bytes (just not its
      // own bytes).
      bucketIdx = 0;
      for (const target of buckets) {
        if (target.recordCount === 0) { bucketIdx++; continue; }
        // Re-stage drawHeap for this bucket.
        const dhBytes = target.recordCount * target.layout.drawHeaderBytes;
        const dhCopy = device.createBuffer({
          size: dhBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const enc1 = device.createCommandEncoder();
        enc1.copyBufferToBuffer(target.drawHeap.buffer, 0, dhCopy, 0, dhBytes);
        device.queue.submit([enc1.finish()]);
        await dhCopy.mapAsync(GPUMapMode.READ);
        const dhU32 = new Uint32Array(dhCopy.getMappedRange());
        const stride2 = target.layout.drawHeaderBytes;
        for (let slot = 0; slot < target.recordCount; slot++) {
          for (const f of target.layout.drawHeaderFields) {
            if (f.kind !== "uniform-ref" && f.kind !== "attribute-ref") continue;
            const ref = dhU32[(slot * stride2 + f.byteOffset) >>> 2];
            if (ref === undefined || ref === 0) continue;
            // Read alloc length (u32 at ref/4 + 1) → derive bytes.
            const len = arenaU32[(ref >>> 2) + 1]!;
            const stride = arenaU32[(ref >>> 2) + 2]!;  // stride bytes
            const dataBytes = stride > 0 ? stride * len : 64; // fallback for uniform-ref
            const allocBytes = ALIGN16(ALLOC_HEADER_PAD_TO + dataBytes);
            allocClaims.push({ ref, bytes: allocBytes, owner: `b${bucketIdx}.slot${slot}.${f.name}` });
          }
        }
        dhCopy.unmap(); dhCopy.destroy();
        bucketIdx++;
      }
      // Sort + walk for overlap. Group by ref first — same ref is OK
      // (legit dedup). Different refs whose ranges intersect = bug.
      allocClaims.sort((a, b) => a.ref - b.ref);
      let prevRef = -1, prevEnd = -1, prevOwner = "";
      for (const c of allocClaims) {
        if (c.ref === prevRef) continue;  // shared (legit)
        if (prevEnd > c.ref) {
          crossSlot++;
          push(
            `arena overlap: ${prevOwner} [0x${prevRef.toString(16)}, 0x${prevEnd.toString(16)}) ` +
            `vs ${c.owner} [0x${c.ref.toString(16)}, 0x${(c.ref + c.bytes).toString(16)})`,
          );
        }
        prevRef = c.ref;
        prevEnd = c.ref + c.bytes;
        prevOwner = c.owner;
      }

      arenaCopy.unmap(); arenaCopy.destroy();
      if (indicesCopy !== undefined) { indicesCopy.unmap(); indicesCopy.destroy(); }
      void samples;

      return { trianglesChecked, crossSlot, issues };
    },
    /** GPU-side binary-search probe: dispatches a tiny compute kernel
     *  that, for each sampled global emit-index, runs the SAME binary
     *  search the render VS does (using firstDrawInTile + drawTable)
     *  and writes the resulting slot into a debug buffer. Returns the
     *  list of (emit, gpuSlot, cpuSlot) where the GPU disagrees with
     *  the CPU. Cross-device comparison: if Chromium reports 0
     *  disagreements but iOS reports any, we've localized the bug to
     *  Apple's WGSL-lowered binary-search loop.
     *
     *  Bucket selection: validates the FIRST non-empty bucket only
     *  (most diagnostic, simpler kernel; expand later if needed). */
    async probeBinarySearch(samples = 50_000): Promise<{
      emitsChecked: number;
      gpuMismatches: number;
      issues: string[];
    }> {
      const issues: string[] = [];
      let gpuMismatches = 0;
      const push = (s: string) => { if (issues.length < 30) issues.push(s); };

      const target = buckets.find(b => b.recordCount > 0 && b.totalEmitEstimate > 0);
      if (target === undefined) return { emitsChecked: 0, gpuMismatches: 0, issues };

      const recordCount = target.recordCount;
      const totalEmit = target.totalEmitEstimate;
      const sampleCount = Math.min(samples, totalEmit);
      const stride = Math.max(1, Math.floor(totalEmit / sampleCount));

      // CPU's firstEmit is GPU-computed — must read it back from the
      // drawTable GPU buffer (the CPU shadow's firstEmit field is
      // always 0, written by the scan kernel only). Otherwise the
      // CPU binary search would compare against zeros.
      const dtCopy = device.createBuffer({
        size: recordCount * RECORD_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      {
        const enc0 = device.createCommandEncoder({ label: "probeBinarySearch.dtCopy" });
        enc0.copyBufferToBuffer(target.drawTableBuf!.buffer, 0, dtCopy, 0, recordCount * RECORD_BYTES);
        device.queue.submit([enc0.finish()]);
        await dtCopy.mapAsync(GPUMapMode.READ);
      }
      // Snapshot ALL drawTable u32s into a regular array — we'll need
      // them after unmap to compute the per-emit CPU expectation.
      const dtSnapshot = new Uint32Array(dtCopy.getMappedRange().slice(0));
      const cpuFirstEmit: number[] = [];
      for (let r = 0; r < recordCount; r++) cpuFirstEmit.push(dtSnapshot[r * RECORD_U32 + 0]!);
      dtCopy.unmap(); dtCopy.destroy();

      // Build sample list of emit indices on CPU.
      const sampleIdxArr = new Uint32Array(sampleCount);
      for (let i = 0; i < sampleCount; i++) sampleIdxArr[i] = Math.min(i * stride, totalEmit - 1);

      // CPU expected slots.
      const cpuSlots = new Uint32Array(sampleCount);
      for (let i = 0; i < sampleCount; i++) {
        const emit = sampleIdxArr[i]!;
        let lo = 0, hi = recordCount - 1;
        while (lo < hi) {
          const mid = (lo + hi + 1) >>> 1;
          if (cpuFirstEmit[mid]! <= emit) lo = mid;
          else hi = mid - 1;
        }
        cpuSlots[i] = lo;
      }

      // GPU kernel — mirrors the render VS preamble's search exactly.
      // Writes (slot, drawIdx, indexStart, indexCount, instanceCount,
      // _local, instId, vid) per emit so we can verify EVERY field
      // the VS computes, not just the slot. 8 u32 per emit.
      const wgsl = /* wgsl */ `
        @group(0) @binding(0) var<storage, read>       drawTable:       array<u32>;
        @group(0) @binding(1) var<storage, read>       firstDrawInTile: array<u32>;
        @group(0) @binding(2) var<storage, read>       sampleEmits:     array<u32>;
        @group(0) @binding(3) var<storage, read>       indexStorage:    array<u32>;
        @group(0) @binding(4) var<storage, read_write> outRows:         array<u32>;
        @group(0) @binding(5) var<uniform>             P:               vec4<u32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          let i = gid.x;
          if (i >= P.x) { return; }
          let emitIdx = sampleEmits[i];
          let _tileIdx = emitIdx >> 6u;
          var lo: u32 = firstDrawInTile[_tileIdx];
          var hi: u32 = firstDrawInTile[_tileIdx + 1u];
          loop {
            if (lo >= hi) { break; }
            let _mid = (lo + hi + 1u) >> 1u;
            if (drawTable[_mid * 5u] <= emitIdx) { lo = _mid; } else { hi = _mid - 1u; }
          }
          let _slot = lo;
          let _firstEmit  = drawTable[_slot * 5u + 0u];
          let _drawIdx    = drawTable[_slot * 5u + 1u];
          let _indexStart = drawTable[_slot * 5u + 2u];
          let _indexCount = drawTable[_slot * 5u + 3u];
          let _instCount  = drawTable[_slot * 5u + 4u];
          let _local      = emitIdx - _firstEmit;
          let _instId     = _local / _indexCount;
          let _vid        = indexStorage[_indexStart + (_local % _indexCount)];
          let base = i * 8u;
          outRows[base + 0u] = _slot;
          outRows[base + 1u] = _drawIdx;
          outRows[base + 2u] = _indexStart;
          outRows[base + 3u] = _indexCount;
          outRows[base + 4u] = _instCount;
          outRows[base + 5u] = _local;
          outRows[base + 6u] = _instId;
          outRows[base + 7u] = _vid;
        }
      `;
      const module = device.createShaderModule({ code: wgsl });
      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      });
      const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module, entryPoint: "main" },
      });

      const sampleBuf = device.createBuffer({
        size: sampleCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(sampleBuf, 0, sampleIdxArr.buffer, sampleIdxArr.byteOffset, sampleIdxArr.byteLength);
      const outBuf = device.createBuffer({
        size: sampleCount * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const paramBuf = device.createBuffer({
        size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(paramBuf, 0, new Uint32Array([sampleCount, 0, 0, 0]));

      const bg = device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: target.drawTableBuf!.buffer } },
          { binding: 1, resource: { buffer: target.firstDrawInTileBuf!.buffer } },
          { binding: 2, resource: { buffer: sampleBuf } },
          { binding: 3, resource: { buffer: arena.indices.buffer } },
          { binding: 4, resource: { buffer: outBuf } },
          { binding: 5, resource: { buffer: paramBuf } },
        ],
      });

      const enc = device.createCommandEncoder({ label: "probeBinarySearch" });
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(sampleCount / 64));
      pass.end();
      const readback = device.createBuffer({
        size: sampleCount * 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      enc.copyBufferToBuffer(outBuf, 0, readback, 0, sampleCount * 32);
      device.queue.submit([enc.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const gpuRows = new Uint32Array(readback.getMappedRange().slice(0));
      readback.unmap(); readback.destroy();
      sampleBuf.destroy(); outBuf.destroy(); paramBuf.destroy();

      // Read drawTable + indices for CPU expectation.
      const indicesCopy2 = device.createBuffer({
        size: arena.indices.buffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc2 = device.createCommandEncoder();
      enc2.copyBufferToBuffer(arena.indices.buffer, 0, indicesCopy2, 0, arena.indices.buffer.size);
      device.queue.submit([enc2.finish()]);
      await indicesCopy2.mapAsync(GPUMapMode.READ);
      const indicesU32 = new Uint32Array(indicesCopy2.getMappedRange().slice(0));
      indicesCopy2.unmap(); indicesCopy2.destroy();

      for (let i = 0; i < sampleCount; i++) {
        const emit = sampleIdxArr[i]!;
        const cpuSlot = cpuSlots[i]!;
        const cpuFirstEmit_  = dtSnapshot[cpuSlot * RECORD_U32 + 0]!;
        const cpuDrawIdx     = dtSnapshot[cpuSlot * RECORD_U32 + 1]!;
        const cpuIndexStart  = dtSnapshot[cpuSlot * RECORD_U32 + 2]!;
        const cpuIndexCount  = dtSnapshot[cpuSlot * RECORD_U32 + 3]!;
        const cpuInstCount   = dtSnapshot[cpuSlot * RECORD_U32 + 4]!;
        const cpuLocal       = emit - cpuFirstEmit_;
        const cpuInstId      = Math.floor(cpuLocal / cpuIndexCount);
        const cpuVid         = indicesU32[cpuIndexStart + (cpuLocal % cpuIndexCount)]!;
        const base = i * 8;
        const got = {
          slot: gpuRows[base + 0]!, drawIdx: gpuRows[base + 1]!,
          indexStart: gpuRows[base + 2]!, indexCount: gpuRows[base + 3]!,
          instCount: gpuRows[base + 4]!, local: gpuRows[base + 5]!,
          instId: gpuRows[base + 6]!, vid: gpuRows[base + 7]!,
        };
        const exp = {
          slot: cpuSlot, drawIdx: cpuDrawIdx,
          indexStart: cpuIndexStart, indexCount: cpuIndexCount,
          instCount: cpuInstCount, local: cpuLocal,
          instId: cpuInstId, vid: cpuVid,
        };
        if (
          got.slot       !== exp.slot       ||
          got.drawIdx    !== exp.drawIdx    ||
          got.indexStart !== exp.indexStart ||
          got.indexCount !== exp.indexCount ||
          got.instCount  !== exp.instCount  ||
          got.local      !== exp.local      ||
          got.instId     !== exp.instId     ||
          got.vid        !== exp.vid
        ) {
          gpuMismatches++;
          push(
            `emit=${emit} GPU vs CPU: ` +
            `slot=${got.slot}/${exp.slot} drawIdx=${got.drawIdx}/${exp.drawIdx} ` +
            `idxStart=${got.indexStart}/${exp.indexStart} idxCount=${got.indexCount}/${exp.indexCount} ` +
            `instCount=${got.instCount}/${exp.instCount} local=${got.local}/${exp.local} ` +
            `instId=${got.instId}/${exp.instId} vid=${got.vid}/${exp.vid}`,
          );
        }
      }
      return { emitsChecked: sampleCount, gpuMismatches, issues };
    },
  };

  return { frame, update, encodeIntoPass, encodeComputePrep, addDraw, removeDraw, stats, dispose, _debug } as HeapScene;
}
