// Freelist — best-fit free-block manager used by AttributeArena and
// IndexAllocator. Replaces the earlier `{off,size}[]` sorted-by-off
// freelist (whose alloc was O(N) first-fit and release was O(log N)
// binary-search-then-coalesce).
//
// Design:
//
//   bySize:  size  → Set<off>                — exact-size buckets
//   sizes:   sorted ascending array of distinct sizes   — for lower_bound
//   byStart: off   → size                    — for right-coalesce
//   byEnd:   off+size → off                  — for left-coalesce
//
// `alloc(size)`  → lower_bound(sizes, size) picks the smallest bucket
//                 with size ≥ requested. O(log K) lookup over distinct
//                 sizes K (typically dozens), then constant-time pop
//                 from the size's offset set. If the bucket size
//                 exceeds the request, the remainder reinserts at
//                 (off+size, leftover).
// `release(o,s)` → check `byEnd[o]` (left neighbour ends here) and
//                 `byStart[o+s]` (right neighbour starts here) for
//                 O(1) adjacency lookups. Coalesce both sides then
//                 reinsert one merged block.
//
// On realistic heap-renderer workloads the number of *distinct*
// free-block sizes is typically much smaller than the number of
// blocks (allocations cluster around the schema's drawHeader
// fields: mat4=64B, vec4=16B, …), so the K-sized sorted array's
// splice cost stays cheap even at thousands of live free blocks.
// For truly heterogeneous workloads, swap `sizes` for a balanced
// tree — none of the API changes.

/**
 * A best-fit free-block manager. All offsets and sizes are in the
 * caller's units (bytes for the attribute arena, u32 elements for
 * the index allocator); the freelist doesn't interpret them.
 */
export class Freelist {
  /** size → set of starting offsets currently free at that size. */
  private readonly bySize = new Map<number, Set<number>>();
  /** Sorted ascending list of every size present in `bySize`. */
  private readonly sizes: number[] = [];
  /** off → size, for the right-coalesce probe in `release`. */
  private readonly byStart = new Map<number, number>();
  /** off+size (= end offset) → off, for the left-coalesce probe. */
  private readonly byEnd = new Map<number, number>();

  /** Number of distinct free blocks currently tracked. */
  get blockCount(): number { return this.byStart.size; }

  /**
   * Return any offset of a free block whose size is ≥ `size`, splitting
   * off any leftover bytes back into the freelist. Returns `undefined`
   * when no block fits.
   */
  alloc(size: number): number | undefined {
    const idx = lowerBound(this.sizes, size);
    if (idx >= this.sizes.length) return undefined;
    const foundSize = this.sizes[idx]!;
    const set = this.bySize.get(foundSize)!;
    // Set iteration order is insertion order — using the first inserted
    // gives deterministic alloc patterns under matched test workloads.
    const off = set.values().next().value as number;
    this.removeBlock(off, foundSize);
    if (foundSize > size) this.addBlock(off + size, foundSize - size);
    return off;
  }

  /**
   * Reintroduce a `(off, size)` block, coalescing with any immediate
   * neighbour on the left (some other block ending at `off`) or right
   * (some other block starting at `off + size`).
   */
  release(off: number, size: number): void {
    let mergeOff = off;
    let mergeSize = size;
    // Left neighbour: a block whose end offset is exactly `off`.
    const leftStart = this.byEnd.get(mergeOff);
    if (leftStart !== undefined) {
      const leftSize = this.byStart.get(leftStart)!;
      this.removeBlock(leftStart, leftSize);
      mergeOff = leftStart;
      mergeSize += leftSize;
    }
    // Right neighbour: a block whose start offset is exactly `off + size`.
    const rightStart = mergeOff + mergeSize;
    const rightSize = this.byStart.get(rightStart);
    if (rightSize !== undefined) {
      this.removeBlock(rightStart, rightSize);
      mergeSize += rightSize;
    }
    this.addBlock(mergeOff, mergeSize);
  }

  /**
   * Take and remove the free block whose end offset is exactly
   * `endOff`, if any exists. Used by `AttributeArena` /
   * `IndexAllocator` to shrink the bump cursor back when releases
   * expose a free tail (§5 — cursor-shrink hygiene for long-lived
   * high-churn scenes).
   */
  takeBlockEndingAt(endOff: number): { off: number; size: number } | undefined {
    const off = this.byEnd.get(endOff);
    if (off === undefined) return undefined;
    const size = this.byStart.get(off)!;
    this.removeBlock(off, size);
    return { off, size };
  }

  /** Drop every block. */
  clear(): void {
    this.bySize.clear();
    this.sizes.length = 0;
    this.byStart.clear();
    this.byEnd.clear();
  }

  /** Snapshot for tests / debugging — sorted by offset. */
  toArray(): Array<{ off: number; size: number }> {
    const out: Array<{ off: number; size: number }> = [];
    for (const [off, size] of this.byStart) out.push({ off, size });
    out.sort((a, b) => a.off - b.off);
    return out;
  }

  private addBlock(off: number, size: number): void {
    let set = this.bySize.get(size);
    if (set === undefined) {
      set = new Set();
      this.bySize.set(size, set);
      const idx = lowerBound(this.sizes, size);
      this.sizes.splice(idx, 0, size);
    }
    set.add(off);
    this.byStart.set(off, size);
    this.byEnd.set(off + size, off);
  }

  private removeBlock(off: number, size: number): void {
    const set = this.bySize.get(size);
    if (set === undefined) return;
    set.delete(off);
    if (set.size === 0) {
      this.bySize.delete(size);
      const idx = lowerBound(this.sizes, size);
      if (this.sizes[idx] === size) this.sizes.splice(idx, 1);
    }
    this.byStart.delete(off);
    this.byEnd.delete(off + size);
  }
}

/** Index of the first element of `arr` that is `>= target`. */
function lowerBound(arr: ReadonlyArray<number>, target: number): number {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid]! < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
