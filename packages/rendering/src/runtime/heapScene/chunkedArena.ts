// Multi-chunk arena allocators — §3 of `docs/heap-future-work.md`.
//
// Each chunk owns a separate GPUBuffer with its own pow2-grow + best-
// fit Freelist. When the newest chunk hits its per-chunk cap, the
// allocator opens a fresh chunk and routes subsequent allocations
// there. Each (chunkIdx, offset) is the address; refs in master
// records / drawHeaders carry only the offset — chunkIdx is implicit
// in which chunk's bind group the current draw call is using.
//
// Cap policy: `maxChunkBytes` ≤ `device.limits.maxStorageBufferBindingSize`
// per chunk. Default: `min(adapter, DEFAULT_MAX_BUFFER_BYTES)`. Tests
// can pass a tiny cap (e.g. 4 MB) to exercise multi-chunk behaviour
// without massive workloads.

import type { IDisposable } from "@aardworx/wombat.adaptive";
import { GrowBuffer, DEFAULT_MAX_BUFFER_BYTES } from "./growBuffer.js";
import { AttributeArena, IndexAllocator } from "./pools.js";

/** Address inside a chunked arena. */
export interface ChunkedRef {
  readonly chunkIdx: number;
  readonly off: number;
}

/**
 * Multi-chunk wrapper around `AttributeArena`. Each chunk is one
 * GrowBuffer / Freelist pair, capped at `maxChunkBytes`. Caller
 * tracks chunkIdx alongside the offset — refs in shaders are just
 * the offset, because shaders bind one chunk at a time.
 */
export class ChunkedAttributeArena {
  private readonly _chunks: AttributeArena[] = [];
  private readonly onChunkAddedCbs = new Set<(chunkIdx: number) => void>();

  constructor(
    private readonly device: GPUDevice,
    private readonly label: string,
    private readonly usage: GPUBufferUsageFlags,
    private readonly initialChunkBytes: number,
    private readonly maxChunkBytes: number = DEFAULT_MAX_BUFFER_BYTES,
  ) {
    this.openChunk();
  }

  get chunkCount(): number { return this._chunks.length; }
  get chunks(): ReadonlyArray<AttributeArena> { return this._chunks; }
  chunk(i: number): AttributeArena { return this._chunks[i]!; }

  /**
   * Allocate `dataBytes` of attribute storage. Tries the `hint`
   * chunk first when given (caller passes the RO's chunk to keep all
   * uniforms in the same buffer); falls back to the newest chunk;
   * opens a new chunk when none fits.
   */
  alloc(dataBytes: number, hint?: number): ChunkedRef {
    if (hint !== undefined && hint >= 0 && hint < this._chunks.length) {
      const off = this._chunks[hint]!.tryAlloc(dataBytes);
      if (off !== undefined) return { chunkIdx: hint, off };
    }
    // Newest-first ordering keeps freshly-opened chunks busy and
    // older chunks naturally drain as their ROs release (helping
    // future drop-empty-chunks cleanup land cheaply).
    for (let i = this._chunks.length - 1; i >= 0; i--) {
      if (i === hint) continue;
      const off = this._chunks[i]!.tryAlloc(dataBytes);
      if (off !== undefined) return { chunkIdx: i, off };
    }
    const newIdx = this.openChunk();
    const off = this._chunks[newIdx]!.tryAlloc(dataBytes);
    if (off === undefined) {
      throw new Error(
        `ChunkedAttributeArena '${this.label}': allocation of ${dataBytes}B exceeds maxChunkBytes ${this.maxChunkBytes}`,
      );
    }
    return { chunkIdx: newIdx, off };
  }

  release(chunkIdx: number, off: number, dataBytes: number): void {
    const c = this._chunks[chunkIdx];
    if (c === undefined) return;
    c.release(off, dataBytes);
  }

  /** Push the dirty CPU shadow of every chunk to its GPU buffer. */
  flush(device: GPUDevice): void {
    for (const c of this._chunks) c.flush(device);
  }

  /** Write into a specific chunk's CPU shadow. */
  write(chunkIdx: number, dst: number, data: Uint8Array): void {
    this._chunks[chunkIdx]!.write(dst, data);
  }

  /** Subscribe to "a new chunk was opened" events. Used by the
   *  bucket-rebuild path to widen its per-chunk bookkeeping. */
  onChunkAdded(cb: (chunkIdx: number) => void): IDisposable {
    this.onChunkAddedCbs.add(cb);
    return { dispose: () => { this.onChunkAddedCbs.delete(cb); } };
  }

  /** Subscribe to "any chunk resized (incl. brand-new chunk)" events.
   *  Wires the callback to every current chunk's `onResize` AND
   *  fires it for each future-added chunk after subscribing.
   *  Used by the scene-level rebuild path that follows arena growth. */
  onAnyResize(cb: (chunkIdx: number) => void): IDisposable {
    const disposables: IDisposable[] = [];
    for (let i = 0; i < this._chunks.length; i++) {
      const idx = i;
      disposables.push(this._chunks[i]!.onResize(() => cb(idx)));
    }
    disposables.push(this.onChunkAdded((idx) => {
      disposables.push(this._chunks[idx]!.onResize(() => cb(idx)));
      // Also fire once for the new chunk so listeners can wire bind
      // groups to it.
      cb(idx);
    }));
    return { dispose: () => { for (const d of disposables) d.dispose(); } };
  }

  /** Total bytes used across all chunks (high-watermark sum). */
  totalUsedBytes(): number {
    let s = 0;
    for (const c of this._chunks) s += c.usedBytes;
    return s;
  }

  destroy(): void {
    for (const c of this._chunks) c.destroy();
    this._chunks.length = 0;
  }

  private openChunk(): number {
    const idx = this._chunks.length;
    const buf = new GrowBuffer(
      this.device, `${this.label}/c${idx}`, this.usage,
      this.initialChunkBytes, this.maxChunkBytes,
    );
    this._chunks.push(new AttributeArena(buf));
    for (const cb of this.onChunkAddedCbs) cb(idx);
    return idx;
  }
}

/** Multi-chunk wrapper around `IndexAllocator`. Element-bump units
 *  are u32. Same chunk-open / hint / fallback semantics as
 *  `ChunkedAttributeArena`. */
export class ChunkedIndexAllocator {
  private readonly _chunks: IndexAllocator[] = [];
  private readonly onChunkAddedCbs = new Set<(chunkIdx: number) => void>();

  constructor(
    private readonly device: GPUDevice,
    private readonly label: string,
    private readonly usage: GPUBufferUsageFlags,
    private readonly initialChunkBytes: number,
    private readonly maxChunkBytes: number = DEFAULT_MAX_BUFFER_BYTES,
  ) {
    this.openChunk();
  }

  get chunkCount(): number { return this._chunks.length; }
  get chunks(): ReadonlyArray<IndexAllocator> { return this._chunks; }
  chunk(i: number): IndexAllocator { return this._chunks[i]!; }

  alloc(elements: number, hint?: number): ChunkedRef {
    if (hint !== undefined && hint >= 0 && hint < this._chunks.length) {
      const off = this._chunks[hint]!.tryAlloc(elements);
      if (off !== undefined) return { chunkIdx: hint, off };
    }
    for (let i = this._chunks.length - 1; i >= 0; i--) {
      if (i === hint) continue;
      const off = this._chunks[i]!.tryAlloc(elements);
      if (off !== undefined) return { chunkIdx: i, off };
    }
    const newIdx = this.openChunk();
    const off = this._chunks[newIdx]!.tryAlloc(elements);
    if (off === undefined) {
      throw new Error(
        `ChunkedIndexAllocator '${this.label}': allocation of ${elements} elements exceeds maxChunkBytes/4 ${this.maxChunkBytes / 4}`,
      );
    }
    return { chunkIdx: newIdx, off };
  }

  release(chunkIdx: number, off: number, elements: number): void {
    const c = this._chunks[chunkIdx];
    if (c === undefined) return;
    c.release(off, elements);
  }

  flush(device: GPUDevice): void {
    for (const c of this._chunks) c.flush(device);
  }

  write(chunkIdx: number, dst: number, data: Uint8Array): void {
    this._chunks[chunkIdx]!.write(dst, data);
  }

  onChunkAdded(cb: (chunkIdx: number) => void): IDisposable {
    this.onChunkAddedCbs.add(cb);
    return { dispose: () => { this.onChunkAddedCbs.delete(cb); } };
  }

  onAnyResize(cb: (chunkIdx: number) => void): IDisposable {
    const disposables: IDisposable[] = [];
    for (let i = 0; i < this._chunks.length; i++) {
      const idx = i;
      disposables.push(this._chunks[i]!.onResize(() => cb(idx)));
    }
    disposables.push(this.onChunkAdded((idx) => {
      disposables.push(this._chunks[idx]!.onResize(() => cb(idx)));
      cb(idx);
    }));
    return { dispose: () => { for (const d of disposables) d.dispose(); } };
  }

  totalUsedElements(): number {
    let s = 0;
    for (const c of this._chunks) s += c.usedElements;
    return s;
  }

  destroy(): void {
    for (const c of this._chunks) c.destroy();
    this._chunks.length = 0;
  }

  private openChunk(): number {
    const idx = this._chunks.length;
    const buf = new GrowBuffer(
      this.device, `${this.label}/c${idx}`, this.usage,
      this.initialChunkBytes, this.maxChunkBytes,
    );
    this._chunks.push(new IndexAllocator(buf));
    for (const cb of this.onChunkAddedCbs) cb(idx);
    return idx;
  }
}
