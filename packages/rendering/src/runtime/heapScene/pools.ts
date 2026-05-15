// Arena + pool allocators that back the heap renderer's per-RO
// storage. Pulled out of heapScene.ts as a unit because the five
// classes here are tightly interleaved:
//
//   AttributeArena    — byte-bump over a STORAGE GrowBuffer
//   IndexAllocator    — element-bump over an INDEX GrowBuffer
//   DrawHeap          — slot-indexed allocator over a STORAGE GrowBuffer
//                       (one slot per RO drawHeader)
//   UniformPool       — aval-keyed refcounted alloc on top of AttributeArena
//   IndexPool         — aval-keyed refcounted alloc on top of IndexAllocator,
//                       with value-tuple dedup for constant avals (§5b).
//
// Plus the per-allocation header constants used by all of them and
// the `buildArenaState` / `arenaBytes` / `writeAttribute` / `asAval`
// / `isBufferView` / `asFloat32` helpers that the heap-scene factory
// uses to assemble the arena.

import { AVal } from "@aardworx/wombat.adaptive";
import type { aval, IDisposable } from "@aardworx/wombat.adaptive";
import type { BufferView } from "../../core/bufferView.js";
import type { HostBufferSource } from "../../core/buffer.js";
import { GrowBuffer, ALIGN16, DEFAULT_MAX_BUFFER_BYTES } from "./growBuffer.js";
import { Freelist } from "./freelist.js";

// ─── Per-allocation header layout ──────────────────────────────────────

/** Per-allocation header: (u32 typeId, u32 length). Data follows
 *  the header aligned up to 16 bytes (so positions/normals/etc. line
 *  up for future vec4 reads). */
export const ALLOC_HEADER_BYTES   = 8;
export const ALLOC_HEADER_PAD_TO  = 16;

/** Encoding-tag enum (low 16 bits of typeId). */
export const ENC_V3F_TIGHT = 1; // tightly-packed array of vec3<f32> (12 B/elt)

/** Semantic-tag enum (high 16 bits of typeId). Optional metadata —
 *  the shader doesn't branch on this. */
export const SEM_POSITIONS = 1;
export const SEM_NORMALS   = 2;

// ─── UniformPool ───────────────────────────────────────────────────────

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
export class UniformPool {
  // Keyed by `aval<unknown>` *by reference* (a plain JS `Map`). These
  // keys are overwhelmingly reactive `cval`s (per-object trafos,
  // colours, …) and the hot path is `acquire`/`release` ~once per
  // drawHeader field per RO.
  private readonly byAval = new Map<aval<unknown>, PoolEntry>();

  has(av: aval<unknown>): boolean { return this.byAval.has(av); }
  entry(av: aval<unknown>): PoolEntry | undefined { return this.byAval.get(av); }

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
    av: aval<unknown>,
    value: unknown,
    dataBytes: number,
    typeId: number,
    length: number,
    pack: (val: unknown, dst: Float32Array, off: number) => void,
  ): number {
    const existing = this.byAval.get(av);
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
    // pick V3- vs V4-tight load expressions for vec4 attributes.
    u32[2] = length > 0 ? Math.floor(dataBytes / length) : 0;
    pack(value, f32, ALLOC_HEADER_PAD_TO / 4);
    arena.write(ref, new Uint8Array(buf));
    void device;
    this.byAval.set(av, { ref, dataBytes, typeId, pack, refcount: 1 });
    return ref;
  }

  /** Decrement refcount; if zero, free the arena allocation. */
  release(arena: AttributeArena, av: aval<unknown>): void {
    const e = this.byAval.get(av);
    if (e === undefined) return;
    e.refcount--;
    if (e.refcount > 0) return;
    arena.release(e.ref, ALIGN16(ALLOC_HEADER_PAD_TO + e.dataBytes));
    this.byAval.delete(av);
  }

  /** Re-pack one entry's data region into the arena's CPU shadow. */
  repack(device: GPUDevice, arena: AttributeArena, av: aval<unknown>, val: unknown): void {
    const e = this.byAval.get(av);
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

// ─── IndexPool ─────────────────────────────────────────────────────────

/**
 * Aval-keyed pool over the `IndexAllocator`. Two draws referencing
 * the same `Uint32Array` (or aval thereof) share an index range —
 * 19K instanced clones of the same mesh share one allocation, one
 * upload.
 *
 * **Value-equality dedup for constant avals (§5b):** when an
 * incoming aval has `isConstant === true`, the pool also keys by
 * the underlying `ArrayBuffer` tuple `(buffer, byteOffset,
 * byteLength)`. Two distinct constant avals wrapping the same
 * `Uint32Array` view collapse to one allocation.
 */
export class IndexPool {
  private readonly byAval = new Map<
    aval<Uint32Array>,
    { entry: IndexPoolEntry; perAvalCount: number }
  >();
  private readonly byValueKey = new Map<string, IndexPoolEntry>();
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
    av: aval<Uint32Array>,
    arr: Uint32Array,
  ): { firstIndex: number; count: number } {
    const bound = this.byAval.get(av);
    if (bound !== undefined) {
      bound.perAvalCount++;
      bound.entry.totalRefcount++;
      return { firstIndex: bound.entry.firstIndex, count: bound.entry.count };
    }
    let valueKey: string | undefined;
    if (av.isConstant) {
      valueKey = `${this.bufferIdOf(arr.buffer)}:${arr.byteOffset}:${arr.byteLength}`;
      const shared = this.byValueKey.get(valueKey);
      if (shared !== undefined) {
        shared.totalRefcount++;
        this.byAval.set(av, { entry: shared, perAvalCount: 1 });
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
    this.byAval.set(av, { entry, perAvalCount: 1 });
    if (valueKey !== undefined) this.byValueKey.set(valueKey, entry);
    return { firstIndex, count: arr.length };
  }

  release(indices: IndexAllocator, av: aval<Uint32Array>): void {
    const bound = this.byAval.get(av);
    if (bound === undefined) return;
    bound.perAvalCount--;
    bound.entry.totalRefcount--;
    if (bound.perAvalCount === 0) this.byAval.delete(av);
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

// ─── DrawHeap (slot-indexed) ───────────────────────────────────────────

/**
 * Slot-indexed allocator over a GrowBuffer. `slotBytes` is set per-
 * instance — each bucket sizes its DrawHeader from its effect's
 * schema, so a bucket whose layout is e.g. 96 B / slot uses a
 * DrawHeap with `slotBytes=96`.
 */
export class DrawHeap {
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

// ─── AttributeArena (byte-bump) ────────────────────────────────────────

/**
 * Byte-bump allocator over a GrowBuffer for variable-size attribute
 * allocations. Each allocation gets a 16-byte aligned start (8-byte
 * (typeId, length) header at the start, data 16 bytes in). Frees go
 * onto a sorted free list with coalesce on insert.
 */
export class AttributeArena {
  private cursor = 0;
  private readonly freelist = new Freelist();
  /** CPU shadow of the entire GPU buffer; writes go here first then
   *  flush() emits one writeBuffer per dirty contiguous range. */
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
  /** Non-throwing alloc — returns `undefined` when the request
   *  would exceed the underlying GrowBuffer's `maxCapacity` cap.
   *  Used by `ChunkedAttributeArena` to probe before spilling to
   *  the next chunk. */
  tryAlloc(dataBytes: number): number | undefined {
    const allocBytes = ALIGN16(ALLOC_HEADER_PAD_TO + dataBytes);
    const reused = this.freelist.alloc(allocBytes);
    if (reused !== undefined) return reused;
    const next = this.cursor + allocBytes;
    if (next > this.buf.maxCapacity) return undefined;
    this.cursor = next;
    this.buf.ensureCapacity(next);
    this.buf.setUsed(next);
    return next - allocBytes;
  }
  alloc(dataBytes: number): number {
    const r = this.tryAlloc(dataBytes);
    if (r === undefined) {
      throw new Error(
        `AttributeArena: allocation of ${dataBytes}B (with header alignment) exceeds chunk cap`,
      );
    }
    return r;
  }
  release(ref: number, dataBytes: number): void {
    const allocBytes = ALIGN16(ALLOC_HEADER_PAD_TO + dataBytes);
    this.freelist.release(ref, allocBytes);
    // §5: shrink the bump cursor back if release exposed a free
    // tail touching it (cascading — `freelist.release` may have
    // coalesced into a single block ending at `cursor`). Reclaims
    // the high-watermark over time in long-lived high-churn scenes.
    while (true) {
      const top = this.freelist.takeBlockEndingAt(this.cursor);
      if (top === undefined) break;
      this.cursor = top.off;
      this.buf.setUsed(this.cursor);
    }
  }
  onResize(cb: () => void): IDisposable { return this.buf.onResize(cb); }
  destroy(): void { this.buf.destroy(); }
}

// ─── IndexAllocator (element-bump) ─────────────────────────────────────

/**
 * Element-bump allocator over an index GrowBuffer (units = u32). Each
 * draw's index range is allocated as one block; on release the block
 * is returned to a Freelist and can be reused best-fit.
 */
export class IndexAllocator {
  private cursor = 0;     // in u32s, not bytes
  private readonly freelist = new Freelist();
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
  /** Non-throwing alloc — returns `undefined` when the request
   *  would exceed the underlying GrowBuffer's `maxCapacity` cap.
   *  Used by `ChunkedIndexAllocator` to probe before spilling. */
  tryAlloc(elements: number): number | undefined {
    const reused = this.freelist.alloc(elements);
    if (reused !== undefined) return reused;
    const nextElts = this.cursor + elements;
    if (nextElts * 4 > this.buf.maxCapacity) return undefined;
    this.cursor = nextElts;
    this.buf.ensureCapacity(nextElts * 4);
    this.buf.setUsed(nextElts * 4);
    return nextElts - elements;
  }
  alloc(elements: number): number {
    const r = this.tryAlloc(elements);
    if (r === undefined) {
      throw new Error(`IndexAllocator: allocation of ${elements} elements exceeds chunk cap`);
    }
    return r;
  }
  release(off: number, elements: number): void {
    this.freelist.release(off, elements);
    // §5: cursor-shrink mirror of AttributeArena.release. Cursor
    // is in u32 elements here, and setUsed on the GrowBuffer
    // expects bytes — multiply by 4.
    while (true) {
      const top = this.freelist.takeBlockEndingAt(this.cursor);
      if (top === undefined) break;
      this.cursor = top.off;
      this.buf.setUsed(this.cursor * 4);
    }
  }
  onResize(cb: () => void): IDisposable { return this.buf.onResize(cb); }
  destroy(): void { this.buf.destroy(); }
}

// ─── ArenaState + helpers ──────────────────────────────────────────────

/**
 * Global arena state: attribute / uniform data lives in `attrs`
 * (multi-typed-view storage); indices live in `indices` (separate
 * INDEX-usage buffer).
 */
export interface ArenaState {
  readonly attrs:    AttributeArena;
  readonly indices:  IndexAllocator;
}

export function buildArenaState(
  device: GPUDevice,
  attrBytesHint: number,
  idxBytesHint: number,
  label: string,
  idxExtraUsage: GPUBufferUsageFlags = 0,
  maxChunkBytes: number | undefined = undefined,
): ArenaState {
  // §3: cap chunk size at min(adapter's maxStorageBufferBindingSize,
  // GrowBuffer's DEFAULT_MAX_BUFFER_BYTES). Lets us grow as far as
  // the hardware lets us (typically 2 GiB+ on desktop, ≥ 256 MB on
  // mobile/integrated) while keeping a sensible internal ceiling.
  // Caller can override.
  const adapterCap = device.limits.maxStorageBufferBindingSize;
  const cap = maxChunkBytes ?? Math.min(adapterCap, DEFAULT_MAX_BUFFER_BYTES);
  const attrs = new AttributeArena(new GrowBuffer(
    device, `${label}/attrs`, GPUBufferUsage.STORAGE,
    attrBytesHint, cap,
  ));
  const indices = new IndexAllocator(new GrowBuffer(
    device, `${label}/idx`, GPUBufferUsage.INDEX | idxExtraUsage,
    idxBytesHint, cap,
  ));
  return { attrs, indices };
}

export function arenaBytes(arena: ArenaState): number {
  return arena.attrs.usedBytes + arena.indices.usedElements * 4;
}

/** Upload a single attribute — header (typeId, length) + data — into the arena at byte offset `ref`. */
export function writeAttribute(
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

export function asAval<T>(v: aval<T> | T): aval<T> {
  return (typeof v === "object" && v !== null && typeof (v as { getValue?: unknown }).getValue === "function")
    ? (v as aval<T>)
    : AVal.constant(v as T);
}

/** Heuristic predicate — BufferView has `buffer: aval<IBuffer>` + elementType. */
export function isBufferView(v: unknown): v is BufferView {
  if (typeof v !== "object" || v === null) return false;
  const o = v as { buffer?: unknown; elementType?: unknown };
  return typeof o.buffer === "object" && o.buffer !== null
      && typeof (o.buffer as { getValue?: unknown }).getValue === "function"
      && typeof o.elementType === "object" && o.elementType !== null;
}

/** Float32 view over a host-side buffer source. Used by the BufferView
 *  packer to hand the pool a typed array it can `set()` from. */
export function asFloat32(data: HostBufferSource): Float32Array {
  if (data instanceof Float32Array) return data;
  if (ArrayBuffer.isView(data)) {
    return new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  }
  return new Float32Array(data); // ArrayBuffer
}
