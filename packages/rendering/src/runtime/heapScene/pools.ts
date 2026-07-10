// Arena + pool allocators that back the heap renderer's per-RO
// storage. Pulled out of heapScene.ts as a unit because the five
// classes here are tightly interleaved:
//
//   AttributeArena    — byte-bump over a STORAGE GrowBuffer
//   (index arrays are AttributeArena regions too — VS storage-decodes them)
//   DrawHeap          — slot-indexed allocator over a STORAGE GrowBuffer
//                       (one slot per RO drawHeader)
//   UniformPool       — aval-keyed refcounted alloc on top of AttributeArena
//   IndexPool         — aval-keyed refcounted alloc on top of AttributeArena,
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
import { ChunkedAttributeArena } from "./chunkedArena.js";

// ─── Debug toggles ─────────────────────────────────────────────────────

/** Opt-in O(live-allocs) overlap validation on every alloc. Set
 *  `globalThis.__wombatDebugAllocOverlap = true` BEFORE scene creation
 *  to enable. Default OFF: the scan made scene build O(n²) — 65 s for a
 *  20 k-object scene, renderer-crash territory at 68 k. The O(1)
 *  release-side checks (double-free, size mismatch) stay always-on. */
const DEBUG_ALLOC_OVERLAP: boolean =
  (globalThis as Record<string, unknown> & typeof globalThis)
    .__wombatDebugAllocOverlap === true;

// ─── Per-allocation header layout ──────────────────────────────────────

/** Per-allocation header: (u32 typeId, u32 length). Data follows
 *  the header aligned up to 16 bytes (so positions/normals/etc. line
 *  up for future vec4 reads). */
export const ALLOC_HEADER_BYTES   = 8;
export const ALLOC_HEADER_PAD_TO  = 16;

/** Default fragmentation-waste floor (bytes) below which `AttributeArena`
 *  declines to compact — relocating a few KB isn't worth a GPU copy +
 *  ref re-seat. Matches Aardvark's 4 MiB compaction floor. */
export const COMPACTION_WASTE_FLOOR_BYTES = 4 * 1024 * 1024;

/** Encoding-tag enum (low 16 bits of typeId). The generated VS decode
 *  branches on this per field (2-arm select; header word 0 shares the
 *  cache line with the broadcast length in word 1). */
export const ENC_V3F_TIGHT = 1; // tightly-packed array of vec3<f32> (12 B/elt)
export const ENC_OCT32     = 2; // oct-packed unit vectors, 2×unorm16 in one u32 (4 B/elt)
export const ENC_C4B       = 3; // RGBA8-unorm colors, one u32/elt → unpack4x8unorm (4 B/elt)

/** Semantic-tag enum (high 16 bits of typeId). Optional metadata —
 *  the shader doesn't branch on this. */
export const SEM_POSITIONS = 1;
export const SEM_NORMALS   = 2;

// ─── UniformPool ───────────────────────────────────────────────────────

/**
 * One pool entry tracks a single (aval, chunkIdx) pair. Refs are
 * byte offsets within the chunk's GPUBuffer; chunkIdx is implicit
 * in which chunk's bind group is active at draw time (shaders
 * never decode it).
 */
interface PoolEntry {
  readonly chunkIdx: number;
  /** Byte offset within the chunk's GPUBuffer. Mutated by
   *  `UniformPool.remapRefs` when a waste-triggered arena compaction
   *  relocates the allocation. */
  ref: number;
  readonly dataBytes: number;
  readonly typeId: number;
  readonly pack: (val: unknown, dst: Float32Array, off: number) => void;
  refcount: number;
}

/**
 * `(aval, chunkIdx)`-keyed pool of chunked-arena allocations. An
 * aval that's referenced from multiple chunks (e.g. two ROs sharing
 * a uniform but landing in different chunks) gets one entry per
 * chunk — accepted duplication cost of the multi-draw-call
 * per-chunk render path (§3). Within a single chunk, ROs sharing
 * an aval still share the underlying allocation.
 */
export class UniformPool {
  private readonly byAval = new Map<aval<unknown>, Map<number, PoolEntry>>();

  has(av: aval<unknown>, chunkIdx: number): boolean {
    return this.byAval.get(av)?.has(chunkIdx) ?? false;
  }
  /** True when the pool holds at least one entry for `av` in any
   *  chunk. Used by the scene's `inputChanged` dispatch to decide
   *  whether a marking aval is one of our tracked allocations. */
  hasAny(av: aval<unknown>): boolean {
    return (this.byAval.get(av)?.size ?? 0) > 0;
  }
  entry(av: aval<unknown>, chunkIdx: number): PoolEntry | undefined {
    return this.byAval.get(av)?.get(chunkIdx);
  }
  /** First chunk this aval is allocated in, if any. Used by `addRO`
   *  to prefer co-locating a new RO with its already-allocated
   *  shared inputs (avoids unnecessary duplication). */
  firstChunkContaining(av: aval<unknown>): number | undefined {
    const byChunk = this.byAval.get(av);
    if (byChunk === undefined || byChunk.size === 0) return undefined;
    return byChunk.keys().next().value as number;
  }

  /**
   * Acquire (or share) an allocation for `aval` in `chunkIdx`.
   * Returns the byte offset within that chunk's GPUBuffer. If a
   * fresh allocation is made, the value is packed + uploaded
   * immediately to the chunk's GPU buffer (mirror-less).
   */
  acquire(
    device: GPUDevice,
    arena: ChunkedAttributeArena,
    av: aval<unknown>,
    chunkIdx: number,
    value: unknown,
    dataBytes: number,
    typeId: number,
    length: number,
    pack: (val: unknown, dst: Float32Array, off: number) => void,
  ): number {
    let byChunk = this.byAval.get(av);
    const existing = byChunk?.get(chunkIdx);
    if (existing !== undefined) {
      existing.refcount++;
      return existing.ref;
    }
    const r = arena.alloc(dataBytes, chunkIdx);
    // `arena.alloc` may spill to a different chunk if `chunkIdx`'s
    // GrowBuffer is at its maxCapacity. Callers (heapScene.addRO)
    // bind a bucket's bind groups to a single chunk's buffer — a
    // silent spill makes the bucket's drawHeader refs point into
    // the wrong chunk's buffer (garbage reads / typeId corruption).
    //
    // Hard-fail with diagnostic info instead of returning quietly.
    // The right long-term fix is for addRO to pre-probe the chunk,
    // open a new bucket bound to the spill chunk, and re-route; but
    // until that lands, a clear error beats unexplained artefacts.
    if (r.chunkIdx !== chunkIdx) {
      throw new Error(
        `UniformPool.acquire: allocator spilled from chunk ${chunkIdx} to chunk ${r.chunkIdx} ` +
        `(${dataBytes} bytes). Caller's bucket is bound to chunk ${chunkIdx}; this would silently ` +
        `corrupt the drawHeader→arena reads. Open a new bucket bound to chunk ${r.chunkIdx} instead.`,
      );
    }
    const finalChunk = r.chunkIdx;
    const allocBytes = ALIGN16(ALLOC_HEADER_PAD_TO + dataBytes);
    const buf = new ArrayBuffer(allocBytes);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);
    u32[0] = typeId;
    u32[1] = length;
    u32[2] = length > 0 ? Math.floor(dataBytes / length) : 0;
    pack(value, f32, ALLOC_HEADER_PAD_TO / 4);
    arena.write(finalChunk, r.off, new Uint8Array(buf));
    void device;
    if (byChunk === undefined) {
      byChunk = new Map();
      this.byAval.set(av, byChunk);
    }
    byChunk.set(finalChunk, {
      chunkIdx: finalChunk, ref: r.off, dataBytes, typeId, pack, refcount: 1,
    });
    return r.off;
  }

  release(arena: ChunkedAttributeArena, av: aval<unknown>, chunkIdx: number): void {
    const byChunk = this.byAval.get(av);
    const e = byChunk?.get(chunkIdx);
    if (e === undefined) return;
    e.refcount--;
    if (e.refcount > 0) return;
    // BUG FIX: arena.release's third arg is `dataBytes` (raw), and the
    // chunk's AttributeArena.release internally adds the header padding
    // via `ALIGN16(ALLOC_HEADER_PAD_TO + dataBytes)`. We MUST pass the
    // raw data size here — passing pre-aligned `allocBytes` (the old
    // code) double-pads, freeing 16 extra bytes per release. The freed
    // surplus then gets coalesced with the neighbouring scalar-uniform
    // allocs (any 32 B u32/f32) into a single combined block, and the
    // next attribute alloc handed out from that pool ends up STARTING
    // 16 B inside the previous alloc → overlapping allocations →
    // garbage `typeId`/`length` in the attribute header.
    arena.release(e.chunkIdx, e.ref, e.dataBytes);
    byChunk!.delete(chunkIdx);
    if (byChunk!.size === 0) this.byAval.delete(av);
  }

  /** Re-seat every entry's `ref` in `chunkIdx` after a waste-triggered
   *  arena compaction relocated allocations. `remap` maps OLD→NEW byte
   *  offset. Entries whose ref didn't move are left as-is. O(entries). */
  remapRefs(chunkIdx: number, remap: ReadonlyMap<number, number>): void {
    if (remap.size === 0) return;
    for (const byChunk of this.byAval.values()) {
      const e = byChunk.get(chunkIdx);
      if (e === undefined) continue;
      const nn = remap.get(e.ref);
      if (nn !== undefined) e.ref = nn;
    }
  }

  /** Re-pack one entry's data region into every chunk that holds
   *  this aval. When an aval is shared across N chunks (each chunk
   *  has its own duplicate alloc), every duplicate needs the
   *  refresh so all the chunks' ROs see the new value. */
  repack(
    device: GPUDevice,
    arena: ChunkedAttributeArena,
    av: aval<unknown>,
    val: unknown,
  ): void {
    const byChunk = this.byAval.get(av);
    if (byChunk === undefined) return;
    let dst: Float32Array | undefined;
    for (const e of byChunk.values()) {
      if (dst === undefined || dst.length !== e.dataBytes / 4) {
        dst = new Float32Array(e.dataBytes / 4);
      }
      e.pack(val, dst, 0);
      arena.write(
        e.chunkIdx,
        e.ref + ALLOC_HEADER_PAD_TO,
        new Uint8Array(dst.buffer, dst.byteOffset, e.dataBytes),
      );
    }
    void device;
  }
  /** Total bytes touched by `repack` for this aval — sum across
   *  chunks. Used by the dirty-bytes diagnostics. */
  totalDataBytes(av: aval<unknown>): number {
    const byChunk = this.byAval.get(av);
    if (byChunk === undefined) return 0;
    let s = 0;
    for (const e of byChunk.values()) s += e.dataBytes;
    return s;
  }
}

// ─── IndexPool ─────────────────────────────────────────────────────────

/**
 * Aval-keyed pool over the (attribute) arena. Two draws referencing
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
  /** Per-aval, per-chunk bookkeeping. Same aval acquired in two
   *  chunks gets two entries (accepted §3 duplication). */
  private readonly byAval = new Map<
    aval<Uint32Array>,
    Map<number, { entry: IndexPoolEntry; perAvalCount: number }>
  >();
  /** Per-chunk constant-aval value-dedup (§5b in-chunk). */
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

  firstChunkContaining(av: aval<Uint32Array>): number | undefined {
    const byChunk = this.byAval.get(av);
    if (byChunk === undefined || byChunk.size === 0) return undefined;
    return byChunk.keys().next().value as number;
  }

  acquire(
    device: GPUDevice,
    arena: ChunkedAttributeArena,
    av: aval<Uint32Array>,
    chunkIdx: number,
    arr: Uint32Array,
  ): { chunkIdx: number; firstIndex: number; count: number } {
    let byChunk = this.byAval.get(av);
    const bound = byChunk?.get(chunkIdx);
    if (bound !== undefined) {
      bound.perAvalCount++;
      bound.entry.totalRefcount++;
      return { chunkIdx: bound.entry.chunkIdx, firstIndex: bound.entry.firstIndex, count: bound.entry.count };
    }
    let valueKey: string | undefined;
    if (av.isConstant) {
      // Value-dedup is scoped per chunk — two constant avals in the
      // same chunk collapse to one allocation; two constant avals in
      // DIFFERENT chunks each get their own (the index range lives in
      // the chunk's GPUBuffer).
      valueKey = `${chunkIdx}:${this.bufferIdOf(arr.buffer)}:${arr.byteOffset}:${arr.byteLength}`;
      const shared = this.byValueKey.get(valueKey);
      if (shared !== undefined) {
        shared.totalRefcount++;
        if (byChunk === undefined) {
          byChunk = new Map();
          this.byAval.set(av, byChunk);
        }
        byChunk.set(chunkIdx, { entry: shared, perAvalCount: 1 });
        return { chunkIdx: shared.chunkIdx, firstIndex: shared.firstIndex, count: shared.count };
      }
    }
    // Index data lives in the SAME arena as attributes/uniforms (the heap VS
    // storage-decodes indices — `indexStorage[_indexStart + i]` — so they need
    // no separate index buffer). Allocate a normal header+data region; the
    // raw u32 indices sit at `ref + 16`, so the element offset the drawTable
    // stores is `(ref + 16) >> 2`. The header is unread by index gathers but
    // kept for a uniform region layout (and so compaction treats it like any
    // other region).
    const byteLen = arr.byteLength;
    const r = arena.alloc(byteLen, chunkIdx);
    const dataOff = r.off + ALLOC_HEADER_PAD_TO;
    const hdr = new Uint32Array([0, arr.length, 4, 0]); // (typeId, length, strideBytes, pad)
    arena.write(r.chunkIdx, r.off, new Uint8Array(hdr.buffer));
    arena.write(r.chunkIdx, dataOff, new Uint8Array(arr.buffer, arr.byteOffset, byteLen));
    void device;
    const firstIndex = dataOff >>> 2;
    const entry: IndexPoolEntry = {
      chunkIdx: r.chunkIdx, ref: r.off, firstIndex, count: arr.length,
      totalRefcount: 1, valueKey,
    };
    if (byChunk === undefined) {
      byChunk = new Map();
      this.byAval.set(av, byChunk);
    }
    byChunk.set(r.chunkIdx, { entry, perAvalCount: 1 });
    if (valueKey !== undefined) this.byValueKey.set(valueKey, entry);
    return { chunkIdx: r.chunkIdx, firstIndex, count: arr.length };
  }

  /** The allocation's byte header offset (`ref`) of `av`'s entry in `chunkIdx`,
   *  or undefined if not allocated there. The heap compaction pass looks the
   *  region up in the arena remap by this ref to compute the draw's new
   *  `_indexStart`. */
  baseFor(av: aval<Uint32Array>, chunkIdx: number): number | undefined {
    return this.byAval.get(av)?.get(chunkIdx)?.entry.ref;
  }

  /** Re-seat every entry after a compaction relocated its region. `remap` maps
   *  OLD→NEW byte header offset (`ref`); `firstIndex` is recomputed from the
   *  new ref. Entries are shared by value-dedup, so each is visited once. */
  remapRefs(chunkIdx: number, remap: ReadonlyMap<number, number>): void {
    if (remap.size === 0) return;
    const seen = new Set<IndexPoolEntry>();
    for (const byChunk of this.byAval.values()) {
      const bound = byChunk.get(chunkIdx);
      if (bound === undefined || seen.has(bound.entry)) continue;
      seen.add(bound.entry);
      const nn = remap.get(bound.entry.ref);
      if (nn !== undefined) {
        bound.entry.ref = nn;
        bound.entry.firstIndex = (nn + ALLOC_HEADER_PAD_TO) >>> 2;
      }
    }
  }

  release(arena: ChunkedAttributeArena, av: aval<Uint32Array>, chunkIdx: number): void {
    const byChunk = this.byAval.get(av);
    const bound = byChunk?.get(chunkIdx);
    if (bound === undefined) return;
    bound.perAvalCount--;
    bound.entry.totalRefcount--;
    if (bound.perAvalCount === 0) {
      byChunk!.delete(chunkIdx);
      if (byChunk!.size === 0) this.byAval.delete(av);
    }
    if (bound.entry.totalRefcount > 0) return;
    arena.release(bound.entry.chunkIdx, bound.entry.ref, bound.entry.count * 4);
    if (bound.entry.valueKey !== undefined) {
      this.byValueKey.delete(bound.entry.valueKey);
    }
  }
}

interface IndexPoolEntry {
  chunkIdx: number;
  /** Byte header offset of the region in the (attribute) arena — the unit the
   *  compaction remap is keyed by. */
  ref: number;
  /** Element (u32) offset to the index DATA (`(ref + 16) >> 2`) — the value
   *  the drawTable's `_indexStart` stores. */
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
  /** DEBUG: tracks slot live/free state. true = live (in use). */
  private readonly live = new Set<number>();
  constructor(private readonly buf: GrowBuffer, private readonly slotBytes: number) {}
  get buffer(): GPUBuffer { return this.buf.buffer; }
  /** Bytes per slot — caller multiplies by slot index for byte offsets. */
  get bytesPerSlot(): number { return this.slotBytes; }
  /** High-water mark in bytes (used to size bind-group entry on rebuild). */
  get usedBytes(): number { return this.nextSlot * this.slotBytes; }
  alloc(): number {
    const slot = this.free.length > 0 ? this.free.pop()! : this.nextSlot++;
    if (this.live.has(slot)) {
      throw new Error(`DrawHeap.alloc: returned already-live slot ${slot}`);
    }
    this.live.add(slot);
    this.buf.ensureCapacity((slot + 1) * this.slotBytes);
    this.buf.setUsed(Math.max(this.buf.usedBytes, (slot + 1) * this.slotBytes));
    return slot;
  }
  release(slot: number): void {
    if (!this.live.has(slot)) {
      throw new Error(`DrawHeap.release: slot ${slot} was not live — double-release or release-of-never-allocated.`);
    }
    this.live.delete(slot);
    this.free.push(slot);
  }
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
  /** Sum of `allocBytes` over all live allocations. `cursor - liveBytes`
   *  is the fragmentation waste below the high-water mark — what
   *  `compact()` reclaims. */
  private liveBytes = 0;
  private readonly freelist = new Freelist();
  /** DEBUG: tracks every live allocation by start byte → size. Set
   *  on alloc, cleared on release. Used by `assertNoOverlap()` to
   *  detect allocator returning overlapping ranges. Enabled in
   *  development; cheap (one Map insert/delete per alloc). */
  private readonly liveAllocs = new Map<number, number>();
  // MIRROR-LESS: no CPU shadow of the arena. Writes go straight to
  // `queue.writeBuffer` (queue-ordered with submits, so they serialize
  // correctly against GrowBuffer's grow-copy and the compaction bounce —
  // both of which preserve contents GPU-side). This halves the heap's
  // host-memory footprint: the source-of-truth for values is the user's
  // avals (they must stay alive — changeability is the point), and the
  // GPU buffer is the render copy; a third full-size CPU mirror bought
  // only dirty-range batching + debug readback comparisons.
  constructor(
    private readonly device: GPUDevice,
    private readonly buf: GrowBuffer,
  ) {}
  get buffer(): GPUBuffer { return this.buf.buffer; }
  get capacity(): number { return this.buf.capacity; }
  get usedBytes(): number { return this.cursor; }
  write(dst: number, data: Uint8Array): void {
    // Immediate upload; Chrome's queue staging does the batching. The
    // write targets the CURRENT buffer handle — a later grow copies it
    // forward (queue-ordered), so ordering stays correct.
    this.device.queue.writeBuffer(this.buf.buffer, dst, data as BufferSource);
  }
  /** No-op — kept for call-site compatibility (mirror-less arena uploads
   *  in `write`). */
  flush(_device: GPUDevice): void {}
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
    if (reused !== undefined) {
      this.recordAlloc(reused, allocBytes, "freelist");
      this.liveBytes += allocBytes;
      return reused;
    }
    const next = this.cursor + allocBytes;
    if (next > this.buf.maxCapacity) return undefined;
    this.cursor = next;
    this.buf.ensureCapacity(next);
    this.buf.setUsed(next);
    const ref = next - allocBytes;
    this.recordAlloc(ref, allocBytes, "bump");
    this.liveBytes += allocBytes;
    return ref;
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
    const tracked = this.liveAllocs.get(ref);
    if (tracked === undefined) {
      throw new Error(
        `AttributeArena.release: ref=${ref} (size=${allocBytes}) was not in liveAllocs — ` +
        `double-free or release for an alloc that never happened. ` +
        `Live alloc count=${this.liveAllocs.size}.`,
      );
    }
    if (tracked !== allocBytes) {
      throw new Error(
        `AttributeArena.release: size mismatch at ref=${ref} — recorded=${tracked} but ` +
        `release args→ allocBytes=${allocBytes} (dataBytes=${dataBytes}). ` +
        `Caller is releasing the wrong size; this WILL corrupt the freelist via over- or under-shrinking.`,
      );
    }
    this.liveAllocs.delete(ref);
    this.liveBytes -= allocBytes;
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
  /** Tracks the alloc; with `__wombatDebugAllocOverlap` set, also
   *  throws if it overlaps any live allocation (O(live) per alloc —
   *  made scene build O(n²) when always-on, hence opt-in). */
  private recordAlloc(off: number, size: number, source: string): void {
    if (DEBUG_ALLOC_OVERLAP) {
      // [a, a+sa) vs [b, b+sb) overlap iff a < b+sb && b < a+sa
      for (const [liveOff, liveSize] of this.liveAllocs) {
        if (off < liveOff + liveSize && liveOff < off + size) {
          throw new Error(
            `AttributeArena.tryAlloc(${source}): returned [${off},${off + size}) ` +
            `(size=${size}) overlaps live alloc at [${liveOff},${liveOff + liveSize}) ` +
            `(size=${liveSize}). Allocator is handing out shared memory!`,
          );
        }
      }
    }
    this.liveAllocs.set(off, size);
  }
  /** Bytes the bump cursor can still grow into before hitting the chunk cap
   *  (ignores freelist holes — a conservative lower bound on free space, used
   *  by group placement to decide whether a draw's group fits here). */
  get bumpHeadroom(): number { return this.buf.maxCapacity - this.cursor; }
  /** Live bytes (sum of all live allocations, header+pad included). */
  get liveByteCount(): number { return this.liveBytes; }
  /** Fragmentation waste below the high-water mark: free bytes the tail-
   *  reclaim couldn't recover because they sit between live allocations. */
  get wasteBytes(): number { return this.cursor - this.liveBytes; }

  /**
   * Waste-triggered compaction. Relocates live allocations to the front of
   * the buffer, eliminating the high-water ratchet that exact-size freelist
   * reuse alone can't fix (a drifting allocation-size distribution never
   * reuses its holes and grows toward the high-water forever).
   *
   * The byte move happens GPU-side via `copyBufferToBuffer` through a scratch
   * buffer — no CPU→GPU re-upload — and is mirrored into the CPU shadow so it
   * stays authoritative for future partial writes. Same-buffer copies can't
   * overlap, hence the scratch round-trip.
   *
   * Returns a map of OLD→NEW byte offset for every allocation that moved; the
   * caller must re-seat every cached ref (drawHeader cells, uniform-pool
   * entries, derived-uniform handles, partition master refs). Returns an
   * empty map when it declines to compact (waste below `wasteFloorBytes` or
   * less than half the extent wasted).
   *
   * Precondition: the arena's pending shadow writes must already be flushed
   * to the GPU buffer (the GPU copy reads the live buffer). O(live).
   */
  compact(device: GPUDevice, wasteFloorBytes: number, force = false): Map<number, number> {
    const waste = this.cursor - this.liveBytes;
    if (!force && (this.liveBytes * 2 >= this.cursor || waste < wasteFloorBytes)) {
      return new Map();
    }
    // Live allocations sorted by current offset.
    const live = [...this.liveAllocs.entries()].sort((a, b) => a[0] - b[0]);
    // Assign packed offsets (bump from 0); build the OLD→NEW remap and the
    // source-contiguous runs to copy. A run is a maximal set of live blocks
    // adjacent in the source; under the bump it lands dest-contiguous too, so
    // the compacted prefix [0, newExtent) is fully covered by the runs.
    const remap = new Map<number, number>();
    const runs: { srcStart: number; dstStart: number; len: number }[] = [];
    let w = 0;
    let cur: { srcStart: number; dstStart: number; len: number } | undefined;
    for (const [off, size] of live) {
      if (w !== off) remap.set(off, w);
      if (cur !== undefined && off === cur.srcStart + cur.len) {
        cur.len += size;
      } else {
        cur = { srcStart: off, dstStart: w, len: size };
        runs.push(cur);
      }
      w += size;
    }
    const newExtent = w;
    if (remap.size === 0) return remap; // already tight

    // GPU move: arena → scratch (compacted), then scratch → arena[0,newExtent).
    // Both copies are between distinct buffers, so no overlap constraint.
    const scratch = device.createBuffer({
      label: "AttributeArena/compact-scratch",
      size: newExtent,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder({ label: "AttributeArena/compact" });
    for (const r of runs) {
      enc.copyBufferToBuffer(this.buf.buffer, r.srcStart, scratch, r.dstStart, r.len);
    }
    enc.copyBufferToBuffer(scratch, 0, this.buf.buffer, 0, newExtent);
    device.queue.submit([enc.finish()]);
    scratch.destroy();

    // Rebuild allocator bookkeeping: live allocs at their new offsets, no free
    // blocks (everything is now contiguous), cursor at the new high-water.
    // (Mirror-less: the GPU bounce above IS the move — nothing else to sync.)
    const moved: Array<[number, number]> = [];
    for (const [off, size] of live) moved.push([remap.get(off) ?? off, size]);
    this.liveAllocs.clear();
    for (const [o, s] of moved) this.liveAllocs.set(o, s);
    this.freelist.clear();
    this.cursor = newExtent;
    this.buf.setUsed(newExtent);
    return remap;
  }

  onResize(cb: () => void): IDisposable { return this.buf.onResize(cb); }
  destroy(): void { this.buf.destroy(); }
}

// ─── ArenaState + helpers ──────────────────────────────────────────────

/**
 * Global arena state: ONE chunked arena holds all heap data (per-draw
 * uniforms, attributes, and index arrays) as multi-typed views over the same
 * buffers. Each chunk owns a separate GPUBuffer + Freelist. Refs that
 * downstream code stores are byte offsets within a chunk; the chunk identity
 * is carried alongside (via `bucket.chunkIdx`) and made implicit at draw time
 * by which chunk's bind group is active.
 */
export interface ArenaState {
  /** The single arena holding ALL heap data — per-draw uniforms, vertex /
   *  instance attributes, AND index arrays (the heap VS storage-decodes
   *  indices, so they need no separate index buffer). */
  readonly attrs: ChunkedAttributeArena;
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
  // GrowBuffer's DEFAULT_MAX_BUFFER_BYTES). Lets chunks grow as far
  // as the hardware lets us (typically 2 GiB+ on desktop, ≥ 256 MB
  // on mobile/integrated) while keeping a sensible internal ceiling.
  // Caller can override (the heap-demo uses a tiny 4 MB cap to
  // exercise multi-chunk routing).
  const adapterCap = device.limits.maxStorageBufferBindingSize;
  const cap = maxChunkBytes ?? Math.min(adapterCap, DEFAULT_MAX_BUFFER_BYTES);
  void idxBytesHint; void idxExtraUsage; // legacy params (separate index buffer removed)
  // Clamp the pre-size hint to the per-chunk cap: GrowBuffer RAISES its max
  // to fit a larger initial size, which would mint an oversized page (and
  // §7 derived-record handles cap page offsets at 2^29 B = 512 MB). A
  // multi-page scene simply pre-sizes each page at the cap as it opens.
  const attrs = new ChunkedAttributeArena(
    device, `${label}/arena`, GPUBufferUsage.STORAGE,
    Math.min(attrBytesHint, cap), cap,
  );
  return { attrs };
}

export function arenaBytes(arena: ArenaState): number {
  return arena.attrs.totalUsedBytes();
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
