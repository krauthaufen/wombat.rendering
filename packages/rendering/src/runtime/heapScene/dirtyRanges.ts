// DirtyRanges — disjoint dirty-interval tracking for CPU-shadow →
// GPU-buffer flushes.
//
// The naive alternative (track one min..max span) degrades brutally
// under SCATTERED writes: k random small edits across a multi-MB
// arena dirty a span covering nearly the whole buffer, so flush
// re-uploads megabytes of clean bytes every frame. Measured on the
// 68k-object CAD bench: 16–28 ms/frame for k = 3…316 edits whose
// actual dirty bytes are a few KB.
//
// Ranges merge on insert when their gap is ≤ `mergeGap` (uploading a
// small clean gap is cheaper than another writeBuffer call), and the
// list is compacted by merging the closest pair whenever it exceeds
// `maxRanges` — bounding both per-write cost and flush call count.

export class DirtyRanges {
  // sorted, disjoint, half-open [start, end) intervals
  private starts: number[] = [];
  private ends: number[] = [];

  constructor(
    private readonly mergeGap: number = 4096,
    private readonly maxRanges: number = 64,
  ) {}

  get isEmpty(): boolean { return this.starts.length === 0; }
  get count(): number { return this.starts.length; }

  add(start: number, end: number): void {
    if (end <= start) return;
    const s = this.starts, e = this.ends;
    // first range that could merge with [start, end): its end + gap >= start
    let lo = 0, hi = s.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (e[mid]! + this.mergeGap < start) lo = mid + 1;
      else hi = mid;
    }
    // last range (exclusive) whose start is within end + gap
    let hi2 = lo;
    while (hi2 < s.length && s[hi2]! <= end + this.mergeGap) hi2++;
    if (lo === hi2) {
      s.splice(lo, 0, start);
      e.splice(lo, 0, end);
    } else {
      const ns = Math.min(start, s[lo]!);
      const ne = Math.max(end, e[hi2 - 1]!);
      s.splice(lo, hi2 - lo, ns);
      e.splice(lo, hi2 - lo, ne);
    }
    if (s.length > this.maxRanges) this.compact();
  }

  /** Merge closest-gap pairs until under the cap. */
  private compact(): void {
    const s = this.starts, e = this.ends;
    while (s.length > this.maxRanges) {
      let best = 0, bestGap = Infinity;
      for (let i = 0; i + 1 < s.length; i++) {
        const gap = s[i + 1]! - e[i]!;
        if (gap < bestGap) { bestGap = gap; best = i; }
      }
      e[best] = e[best + 1]!;
      s.splice(best + 1, 1);
      e.splice(best + 1, 1);
    }
  }

  /** Visit every dirty range in ascending order, then clear. */
  drain(emit: (start: number, end: number) => void): void {
    for (let i = 0; i < this.starts.length; i++) {
      emit(this.starts[i]!, this.ends[i]!);
    }
    this.starts.length = 0;
    this.ends.length = 0;
  }
}
