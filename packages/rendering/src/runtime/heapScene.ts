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

import { Trafo3d, V3d, V4f, type V2f, type M44d } from "@aardworx/wombat.base";
import type { ITexture } from "../core/texture.js";
import type { ISampler } from "../core/sampler.js";
import { AVal, AdaptiveObject, AdaptiveToken } from "@aardworx/wombat.adaptive";
import type { aval, aset, IAdaptiveObject, IDisposable, IHashSetReader } from "@aardworx/wombat.adaptive";
import type { Effect } from "@aardworx/wombat.shader";
import type { PipelineState } from "../core/pipelineState.js";
import type { BufferView } from "../core/bufferView.js";
import type { IBuffer, HostBufferSource } from "../core/buffer.js";
import {
  buildBucketLayout, compileHeapEffect,
  type BucketLayout, type FragmentOutputLayout,
} from "./heapEffect.js";
import { compileHeapEffectIR } from "./heapEffectIR.js";
import {
  ATLAS_PAGE_FORMATS, atlasFormatIndex,
  type AtlasPage, type AtlasPageFormat, type AtlasPool,
} from "./textureAtlas/atlasPool.js";
import {
  ATLAS_ARRAY_SIZE, ATLAS_LINEAR_BINDING_BASE,
  ATLAS_SRGB_BINDING_BASE, ATLAS_SAMPLER_BINDING,
} from "./heapEffect.js";

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
  // M44d._data is the row-major Float64Array; not part of the public
  // type surface. toArray() round-trips through a fresh number[] which
  // we copy into the f32 staging buffer below.
  const r = m.toArray();
  for (let i = 0; i < 16; i++) dst[off + i] = r[i]!;
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

function packerForWgslType(wgslType: string): WgslPacker {
  switch (wgslType) {
    case "mat4x4<f32>": return PACKER_MAT4;
    case "vec4<f32>":   return PACKER_VEC4;
    case "vec3<f32>":   return PACKER_VEC3;
    case "vec2<f32>":   return PACKER_VEC2;
    case "f32":         return PACKER_F32;
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
 */
class IndexPool {
  private readonly byAval = new Map<aval<Uint32Array>, { firstIndex: number; count: number; refcount: number }>();

  acquire(
    device: GPUDevice,
    indices: IndexAllocator,
    aval: aval<Uint32Array>,
    arr: Uint32Array,
  ): { firstIndex: number; count: number } {
    const existing = this.byAval.get(aval);
    if (existing !== undefined) {
      existing.refcount++;
      return { firstIndex: existing.firstIndex, count: existing.count };
    }
    const firstIndex = indices.alloc(arr.length);
    indices.write(
      firstIndex * 4,
      new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength),
    );
    void device;
    this.byAval.set(aval, { firstIndex, count: arr.length, refcount: 1 });
    return { firstIndex, count: arr.length };
  }

  release(indices: IndexAllocator, aval: aval<Uint32Array>): void {
    const e = this.byAval.get(aval);
    if (e === undefined) return;
    e.refcount--;
    if (e.refcount > 0) return;
    indices.release(e.firstIndex, e.count);
    this.byAval.delete(aval);
  }
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
    this.freeList.push({ off: ref, size: allocBytes });
    // Coalesce adjacent free entries for cleanliness.
    this.freeList.sort((a, b) => a.off - b.off);
    for (let i = 0; i < this.freeList.length - 1; ) {
      const a = this.freeList[i]!, b = this.freeList[i + 1]!;
      if (a.off + a.size === b.off) { a.size += b.size; this.freeList.splice(i + 1, 1); }
      else i++;
    }
  }
  onResize(cb: () => void): IDisposable { return this.buf.onResize(cb); }
  destroy(): void { this.buf.destroy(); }
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
    this.freeList.push({ off, size: elements });
    this.freeList.sort((a, b) => a.off - b.off);
    for (let i = 0; i < this.freeList.length - 1; ) {
      const a = this.freeList[i]!, b = this.freeList[i + 1]!;
      if (a.off + a.size === b.off) { a.size += b.size; this.freeList.splice(i + 1, 1); }
      else i++;
    }
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

const HEAP_SCAN_WGSL = `
struct Params {
  numRecords: u32,
  numBlocks:  u32,
  _pad0:      u32,
  _pad1:      u32,
};

struct Record {
  firstEmit:  u32,
  drawIdx:    u32,
  indexStart: u32,
  indexCount: u32,
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
  if (i0 < n) { v0 = drawTable[i0].indexCount; }
  if (i1 < n) { v1 = drawTable[i1].indexCount; }
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
    firstDrawInTile[tileIdx] = params.numRecords;
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

  /** Live local slots (drives the render loop). */
  readonly drawSlots: number[];
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
  /**
   * Optional per-instance values (shape 2 — "I know this is one mesh
   * with N transforms"). When present, `count` is the instance count
   * and each entry of `values` overrides the same-named uniform from
   * `inputs`: it's read once per instance via `instance_index` rather
   * than once per draw. The pool stores it as a packed array
   * (count × stride bytes) keyed on aval identity.
   *
   * The keys of `values` MUST be uniforms declared by the effect's
   * schema; they shadow any same-named entry in `inputs`.
   *
   * Each instanced spec gets its own bucket (single slot). Non-
   * instanced draws are unaffected.
   */
  readonly instances?: {
    readonly count: aval<number> | number;
    readonly values: { readonly [name: string]: aval<unknown> | unknown };
  };
}

export interface HeapSceneStats {
  readonly groups: number;
  totalDraws: number;
  drawBytes: number;
  geometryBytes: number;
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
   * Megacall mode: collapse N drawIndexed-per-bucket into one
   * `pass.drawIndirect(...)` per bucket. Per-record `firstEmit` and the
   * indirect-draw args are computed on-GPU each frame via a Blelloch
   * scan in `encodeComputePrep`. Throws on instanced specs.
   */
  readonly megacall?: boolean;
  /**
   * Atlas pool that owns the per-format independent page textures.
   * heapScene reads `pool.pagesFor(format)` to wire each page's
   * `GPUTexture` into the matching slot of the bucket's BGL ladder
   * and subscribes to `pool.onPageAdded` so bind groups rebuild when
   * a fresh page joins. Required when any RO produces an atlas-
   * variant `HeapTextureSet`.
   */
  readonly atlasPool?: AtlasPool;
}

export function buildHeapScene(
  device: GPUDevice,
  sig: import("../core/framebufferSignature.js").FramebufferSignature,
  initialDraws: readonly HeapDrawSpec[] | aset<HeapDrawSpec>,
  opts: BuildHeapSceneOptions = {},
): HeapScene {
  const megacall = opts.megacall ?? false;
  const atlasPool = opts.atlasPool;
  const colorAttachmentName = sig.colorNames[0];
  if (colorAttachmentName === undefined) {
    throw new Error("buildHeapScene: framebuffer signature has no color attachment");
  }
  const colorFormat = sig.colors.tryFind(colorAttachmentName)!;
  const depthFormat = sig.depthStencil?.format;

  // ─── Global arena (uniform/attribute data + index buffer) ────────
  // Initial capacities are just hints; both buffers pow2-grow on
  // demand. Skip per-draw enumeration since aval-keyed sharing makes
  // the actual allocated size hard to predict (10K instanced draws
  // sharing the same Positions array → 1 alloc, not 10K).
  const arena = buildArenaState(
    device, 64 * 1024, 16 * 1024, "heapScene",
    megacall ? GPUBufferUsage.STORAGE : 0,
  );

  // ─── Per-draw global bookkeeping (sparse, indexed by drawId) ──────
  const drawIdToBucket:    (Bucket | undefined)[] = [];
  const drawIdToLocalSlot: (number | undefined)[] = [];
  /** Per-draw index aval — for `indexPool.release` on removeDraw. */
  const drawIdToIndexAval: (aval<Uint32Array> | undefined)[] = [];
  let nextDrawId = 0;

  /** Unwrap an aval to its inner value, or pass through a plain value. */
  function readPlain<T>(v: aval<T> | T): T {
    if (typeof v === "object" && v !== null && typeof (v as { force?: unknown }).force === "function") {
      return (v as aval<T>).force();
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
      }
    }
  }
  const sceneObj = new HeapSceneObj();

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
    const key = `t${layout.textureBindings.length}|s${layout.samplerBindings.length}|m${layout.megacall ? 1 : 0}|a${withAtlasArrays ? 1 : 0}`;
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
    if (layout.megacall) {
      entries.push(
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      );
    }
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
  if (megacall) {
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
  const psIds = new WeakMap<PipelineState, string>();
  let psCounter = 0;
  const psIdOf = (ps: PipelineState | undefined): string => {
    if (ps === undefined) return "ps#default";
    let id = psIds.get(ps);
    if (id === undefined) { id = `ps#${psCounter++}`; psIds.set(ps, id); }
    return id;
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
    if (bucket.layout.megacall) {
      if (bucket.drawTableBuf === undefined) {
        throw new Error("heapScene: megacall bucket without drawTableBuf");
      }
      // Bind drawTable with size = recordCount * 16 so the VS prelude's
      // `arrayLength(&drawTable)/4u` returns exactly the live record
      // count — keeps stale tail entries out of the binary search.
      // Minimum 16 bytes (one zero-record) to satisfy WebGPU non-zero
      // size constraint when the bucket is empty.
      const dtBytes = Math.max(16, bucket.recordCount * 16);
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
  // Megacall: indexStorage is bound at slot 5 from arena.indices, so
  // a grow there also invalidates every bucket's bind group.
  if (megacall) {
    arena.indices.onResize(() => {
      for (const b of buckets) b.bindGroup = buildBucketBindGroup(b);
    });
  }

  // ─── findOrCreateBucket ───────────────────────────────────────────
  let instancedBucketCounter = 0;
  // Atlas-variant ROs share buckets by (effect, pipelineState) regardless
  // of which atlas pages they happen to land on — the bucket extends its
  // page set lazily as new ROs join. Standalone-variant ROs keep the
  // texture-identity wedge in the key (today's behavior).
  function bucketTextureKey(textures: HeapTextureSet | undefined): string {
    if (textures === undefined) return "tex#none";
    if (textures.kind === "atlas") return "tex#atlas";
    return texIdOf(textures);
  }
  function findOrCreateBucket(
    effect: Effect,
    textures: HeapTextureSet | undefined,
    pipelineState: PipelineState | undefined,
    instanceOpts: { isInstanced: boolean; perInstanceUniforms: ReadonlySet<string> },
  ): Bucket {
    const psKey = psIdOf(pipelineState);
    const texKey = bucketTextureKey(textures);
    const bk = instanceOpts.isInstanced
      ? `${idOf(effect)}|${texKey}|${psKey}|inst#${instancedBucketCounter++}`
      : `${idOf(effect)}|${texKey}|${psKey}`;
    const existing = bucketByKey.get(bk);
    if (existing !== undefined) return existing;
    const ps = resolvePipelineState(pipelineState);

    const isAtlasBucket = textures !== undefined && textures.kind === "atlas";
    const baseLayoutOpts = {
      isInstanced: instanceOpts.isInstanced,
      perInstanceUniforms: instanceOpts.perInstanceUniforms,
      megacall,
    };
    const compiled = compileHeapEffect(effect, opts.fragmentOutputLayout);
    const atlasNames = isAtlasBucket
      ? new Set(compiled.schema.textures.map(t => t.name))
      : new Set<string>();
    const layout: BucketLayout = buildBucketLayout(
      compiled.schema, textures !== undefined,
      { ...baseLayoutOpts, atlasTextureBindings: atlasNames },
    );
    const ir = compileHeapEffectIR(effect, layout, {
      target: "wgsl",
      ...(opts.fragmentOutputLayout !== undefined ? { fragmentOutputLayout: opts.fragmentOutputLayout } : {}),
    });
    const vsModule = device.createShaderModule({ code: ir.vs, label: `heapScene/${bk}/vs` });
    const fsModule = device.createShaderModule({ code: ir.fs, label: `heapScene/${bk}/fs` });
    const vsEntry = ir.vsEntry;
    const fsEntry = ir.fsEntry;
    const { pipelineLayout } = getBgl(layout, isAtlasBucket);

    const pipeline = device.createRenderPipeline({
      label: `heapScene/${bk}/pipeline`,
      layout: pipelineLayout,
      vertex:   { module: vsModule, entryPoint: vsEntry, buffers: [] },
      fragment: { module: fsModule, entryPoint: fsEntry, targets: [{ format: colorFormat }] },
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
      label: bk, textures, layout, pipeline,
      bindGroup: null as unknown as GPUBindGroup,
      drawHeap,
      drawHeaderStaging: new Float32Array(drawHeapBuf.capacity / 4),
      headerDirtyMin: Infinity, headerDirtyMax: 0,
      localPosRefs: [], localNorRefs: [],
      localEntries: [], localToDrawId: [],
      localPerDrawAvals: [], localPerDrawRefs: [],
      drawSlots: [], dirty: new Set<number>(),
      drawTableDirtyMin: Infinity, drawTableDirtyMax: 0,
      recordCount: 0, slotToRecord: [], recordToSlot: [],
      totalEmitEstimate: 0,
      scanDirty: false,
      isAtlasBucket,
      localAtlasReleases: [],
      localAtlasTextures: [],
    };
    if (megacall) {
      const dtBuf = new GrowBuffer(
        device, `heapScene/${bk}/drawTable`,
        GPUBufferUsage.STORAGE,
        1024,
      );
      const blockSumsBuf = new GrowBuffer(
        device, `heapScene/${bk}/blockSums`,
        GPUBufferUsage.STORAGE,
        4 * Math.max(1, Math.ceil((dtBuf.capacity / 16) / SCAN_TILE_SIZE)),
      );
      const blockOffsetsBuf = new GrowBuffer(
        device, `heapScene/${bk}/blockOffsets`,
        GPUBufferUsage.STORAGE,
        4 * Math.max(1, Math.ceil((dtBuf.capacity / 16) / SCAN_TILE_SIZE)),
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
        bucket.drawTableDirtyMax = bucket.recordCount * 16;
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
    });
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
  ): void {
    const dst = bucket.drawHeaderStaging;
    const u32 = new Uint32Array(dst.buffer, dst.byteOffset, dst.length);
    const baseFloat = (localSlot * bucket.layout.drawHeaderBytes) / 4;
    for (const f of bucket.layout.drawHeaderFields) {
      // texture-ref fields carry inline values (pageRef/formatBits as u32,
      // origin/size as vec2<f32>) and are filled by packAtlasTextureFields.
      if (f.kind === "texture-ref") continue;
      const fOff = baseFloat + f.byteOffset / 4;
      const ref = perDrawRefs.get(f.name);
      if (ref === undefined) {
        throw new Error(`heapScene: missing ref for '${f.name}' (kind=${f.kind}) on local slot ${localSlot}`);
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
  };
  Object.defineProperty(stats, "groups", { get: () => buckets.length, configurable: true });

  // ─── addDraw / removeDraw ─────────────────────────────────────────
  function addDraw(spec: HeapDrawSpec): number {
    const drawId = nextDrawId++;
    const isInstanced = spec.instances !== undefined;
    const perInstanceUniforms = isInstanced
      ? new Set(Object.keys(spec.instances!.values))
      : new Set<string>();
    const bucket = findOrCreateBucket(spec.effect, spec.textures, spec.pipelineState, {
      isInstanced, perInstanceUniforms,
    });

    // Indices live in their own INDEX-usage buffer (WebGPU constraint).
    // Aval-keyed: 19K instanced clones of the same mesh share one
    // index allocation + one upload.
    const indicesAval = asAval(spec.indices) as aval<Uint32Array>;
    const indicesArr = readPlain(spec.indices) as Uint32Array;
    const idxAlloc = indexPool.acquire(device, arena.indices, indicesAval, indicesArr);

    const localSlot = bucket.drawHeap.alloc();
    const instanceCount = isInstanced
      ? readPlain(spec.instances!.count) as number
      : 1;

    // Walk the bucket's schema-driven DrawHeader fields. Per-instance
    // uniforms (instanceOpts.perInstanceUniforms) pull from
    // `spec.instances.values` and pack into an array allocation;
    // everything else pulls from `spec.inputs` and packs as a single
    // value. Both go through the same pool — sharing emerges from
    // aval identity either way.
    const perDrawAvals: aval<unknown>[] = [];
    const perDrawRefs = new Map<string, number>();
    sceneObj.evaluateAlways(AdaptiveToken.top, (tok) => {
      for (const f of bucket.layout.drawHeaderFields) {
        // Atlas-variant texture bindings carry inline values rather than
        // pool refs; packAtlasTextureFields fills them after this loop.
        if (f.kind === "texture-ref") continue;
        const isPerInstanceField =
          f.kind === "uniform-ref" && perInstanceUniforms.has(f.name);
        const provided = isPerInstanceField
          ? spec.instances!.values[f.name]
          : spec.inputs[f.name];
        if (provided === undefined) {
          const where = isPerInstanceField ? "spec.instances.values" : "spec.inputs";
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
        let av: aval<unknown>;
        let value: unknown;
        let placement: ReturnType<typeof poolPlacementFor>;
        if (f.kind === "attribute-ref" && !isPerInstanceField && isBufferView(provided)) {
          const bv = provided;
          placement = bufferViewPlacement(f, bv);
          av = bv.buffer as aval<unknown>;
          value = bv.buffer.getValue(tok);
        } else {
          av = asAval(provided as aval<unknown> | unknown);
          value = av.getValue(tok);
          placement = isPerInstanceField
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
    });

    bucket.localPosRefs[localSlot] = perDrawRefs.get("Positions");
    bucket.localNorRefs[localSlot] = perDrawRefs.get("Normals");
    bucket.localEntries[localSlot] = {
      indexCount: idxAlloc.count, firstIndex: idxAlloc.firstIndex, instanceCount,
    };
    bucket.localToDrawId[localSlot] = drawId;
    bucket.drawSlots.push(localSlot);
    bucket.localPerDrawAvals[localSlot] = perDrawAvals;
    bucket.localPerDrawRefs[localSlot]  = perDrawRefs;

    packBucketHeader(bucket, localSlot, perDrawRefs);
    if (bucket.isAtlasBucket && spec.textures !== undefined && spec.textures.kind === "atlas") {
      packAtlasTextureFields(bucket, localSlot, spec.textures);
      bucket.localAtlasReleases[localSlot] = spec.textures.release;
      bucket.localAtlasTextures[localSlot] = spec.textures;
    }
    const byteOff = localSlot * bucket.layout.drawHeaderBytes;
    if (byteOff < bucket.headerDirtyMin) bucket.headerDirtyMin = byteOff;
    const end = byteOff + bucket.layout.drawHeaderBytes;
    if (end > bucket.headerDirtyMax) bucket.headerDirtyMax = end;

    if (bucket.layout.megacall) {
      const dtBuf = bucket.drawTableBuf!;
      const recIdx = bucket.recordCount;
      if (recIdx >= SCAN_MAX_RECORDS) {
        throw new Error(
          `heapScene: megacall bucket exceeds SCAN_MAX_RECORDS (${SCAN_MAX_RECORDS}); ` +
          `extend the scan to multi-level if you need more`,
        );
      }
      const byteOff = recIdx * 16;
      dtBuf.ensureCapacity(byteOff + 16);
      dtBuf.setUsed(Math.max(dtBuf.usedBytes, byteOff + 16));
      // Grow scan-side buffers if recordCount crosses a tile boundary.
      const needBlocks = Math.max(1, Math.ceil((recIdx + 1) / SCAN_TILE_SIZE));
      bucket.blockSumsBuf!.ensureCapacity(needBlocks * 4);
      bucket.blockOffsetsBuf!.ensureCapacity(needBlocks * 4);
      const shadow = bucket.drawTableShadow!;
      // firstEmit is GPU-overwritten by the prefix-sum pass; 0 is fine.
      shadow[recIdx * 4 + 0] = 0;
      shadow[recIdx * 4 + 1] = localSlot;
      shadow[recIdx * 4 + 2] = idxAlloc.firstIndex;
      shadow[recIdx * 4 + 3] = idxAlloc.count;
      bucket.recordCount = recIdx + 1;
      bucket.slotToRecord[localSlot] = recIdx;
      bucket.recordToSlot[recIdx] = localSlot;
      if (byteOff < bucket.drawTableDirtyMin) bucket.drawTableDirtyMin = byteOff;
      if (byteOff + 16 > bucket.drawTableDirtyMax) bucket.drawTableDirtyMax = byteOff + 16;
      bucket.totalEmitEstimate += idxAlloc.count;
      const newNumTiles = Math.max(1, Math.ceil(bucket.totalEmitEstimate / TILE_K));
      bucket.firstDrawInTileBuf!.ensureCapacity((newNumTiles + 1) * 4);
      bucket.scanDirty = true;
    }

    drawIdToBucket[drawId]    = bucket;
    drawIdToLocalSlot[drawId] = localSlot;
    drawIdToIndexAval[drawId] = indicesAval;
    stats.totalDraws++;
    stats.geometryBytes = arenaBytes(arena);
    return drawId;
  }

  function removeDraw(drawId: number): void {
    const bucket    = drawIdToBucket[drawId];
    const localSlot = drawIdToLocalSlot[drawId];
    if (bucket === undefined || localSlot === undefined) return;
    if (bucket.layout.megacall) {
      const removedCount = bucket.localEntries[localSlot]?.indexCount ?? 0;
      bucket.totalEmitEstimate = Math.max(0, bucket.totalEmitEstimate - removedCount);
      // Swap-pop: move the last record into the freed slot, decrement
      // recordCount. firstEmit is GPU-rewritten by the next scan, so
      // we only fix (drawIdx, indexStart, indexCount).
      const recIdx     = bucket.slotToRecord[localSlot]!;
      const lastRecIdx = bucket.recordCount - 1;
      const shadow = bucket.drawTableShadow!;
      if (recIdx !== lastRecIdx) {
        const dst = recIdx * 4;
        const src = lastRecIdx * 4;
        shadow[dst + 0] = 0;
        shadow[dst + 1] = shadow[src + 1]!;
        shadow[dst + 2] = shadow[src + 2]!;
        shadow[dst + 3] = shadow[src + 3]!;
        const movedSlot = bucket.recordToSlot[lastRecIdx]!;
        bucket.slotToRecord[movedSlot] = recIdx;
        bucket.recordToSlot[recIdx] = movedSlot;
        const byteOff = recIdx * 16;
        if (byteOff < bucket.drawTableDirtyMin) bucket.drawTableDirtyMin = byteOff;
        if (byteOff + 16 > bucket.drawTableDirtyMax) bucket.drawTableDirtyMax = byteOff + 16;
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
    bucket.localAtlasReleases[localSlot] = undefined;
    bucket.localAtlasTextures[localSlot] = undefined;

    bucket.localPerDrawAvals[localSlot] = undefined;
    bucket.localPerDrawRefs[localSlot]  = undefined;
    bucket.localPosRefs[localSlot]  = undefined;
    bucket.localNorRefs[localSlot]  = undefined;
    bucket.localEntries[localSlot]  = undefined;
    bucket.localToDrawId[localSlot] = undefined;
    const idx = bucket.drawSlots.indexOf(localSlot);
    if (idx >= 0) bucket.drawSlots.splice(idx, 1);
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
        const id = addDraw(op.value);
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
    for (const d of initialDraws as readonly HeapDrawSpec[]) addDraw(d);
  } else {
    asetReader = (initialDraws as aset<HeapDrawSpec>).getReader();
    sceneObj.evaluateAlways(AdaptiveToken.top, (tok) => drainAsetWith(tok));
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

      // 2. Per-bucket: (rare) header re-pack — only fires when the
      //    bucket's drawHeap GrowBuffer reallocated and we need to
      //    re-write all live slots into the new staging mirror.
      for (const bucket of buckets) {
        if (bucket.dirty.size === 0) continue;
        for (const localSlot of bucket.dirty) {
          const refs = bucket.localPerDrawRefs[localSlot];
          if (refs === undefined) continue;
          packBucketHeader(bucket, localSlot, refs);
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
      if (bucket.layout.megacall && bucket.drawTableDirtyMax > bucket.drawTableDirtyMin) {
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
    if (!megacall) pass.setIndexBuffer(arena.indices.buffer, "uint32");
    let curBg: GPUBindGroup | null = null;
    for (const b of buckets) {
      if (b.bindGroup !== curBg) { pass.setBindGroup(0, b.bindGroup); curBg = b.bindGroup; }
      pass.setPipeline(b.pipeline);
      if (b.layout.megacall) {
        if (b.recordCount > 0) pass.drawIndirect(b.indirectBuf!, 0);
        continue;
      }
      for (const localSlot of b.drawSlots) {
        const e = b.localEntries[localSlot]!;
        // Instanced buckets force `drawIdx = 0u` in WGSL (single slot),
        // so firstInstance must be 0 too — `instance_index` then runs
        // 0..instanceCount-1 and is read as `iidx`. Non-instanced
        // buckets keep the firstInstance=slot trick.
        const firstInstance = b.layout.isInstanced ? 0 : localSlot;
        pass.drawIndexed(e.indexCount, e.instanceCount, e.firstIndex, 0, firstInstance);
      }
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
  function encodeComputePrep(enc: GPUCommandEncoder, _token: AdaptiveToken): void {
    if (!megacall) return;
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
  };

  return { frame, update, encodeIntoPass, encodeComputePrep, addDraw, removeDraw, stats, dispose, _debug } as HeapScene;
}
