# Heap-renderer debug tools

Diagnostic methods on `HeapScene._debug` (also exposed via `HybridScene` /
`IRenderTask` and the heap-demo's URL hooks). Built for hunting an
elusive cross-RO triangle bug; kept around because the next obscure
heap-renderer bug will likely need the same kind of tool.

The pattern is: stage every relevant GPU buffer via `copyBufferToBuffer`
into a `MAP_READ` buffer, `mapAsync`, walk the bytes on CPU, compare
to expectation. None of these are wired into the per-frame path —
they're one-shot, on-demand.

## `validateHeap()` — static heap-state coherence

Walks every drawHeader of every bucket and verifies:

| Check | What it catches |
|---|---|
| Each ref ∈ uniform-ref / attribute-ref field is in `[0, arena.attrs.size)` | A drawHeader pointing past the arena buffer. |
| `drawTable[i].instanceCount > 0`, `indexCount > 0` | Live records with bogus count fields. |
| `(indexStart + indexCount) * 4 ≤ arena.indices.size` | An indexed-draw range overflowing into unallocated bytes. |
| `firstEmit[i]` matches CPU prefix sum of `indexCount × instanceCount` | Scan-kernel output drift. |
| Attribute alloc-header (`typeId ∈ {0,1,2,3}`, `length > 0`, data-range fits, all-finite floats) | Corrupted alloc headers, NaN/Inf attribute values. |
| `firstDrawInTile[t]` matches CPU binary-search of CPU prefix sum at tile boundary `t × 64` | Scan-kernel's per-tile bound is wrong. |
| `indirect[0]` (mega-call vertexCount) equals `bucket.totalEmitEstimate` | The drawIndirect command is dispatching a wrong number of vertices. |
| `indirect[1]` is exactly 1 | The instanceCount in the drawIndirect args got clobbered. |
| `indicesHash` (fnv1a of `arena.indices`) | Cross-device fingerprint — paste from one device, compare from another. |

Demo URL hook: `?validate=1` prints a one-line summary 2 s after init.
Status overlay turns red on any error.

## `simulateDraws(samples)` — per-emit CPU shader simulation

For `samples` random emits across all buckets, runs the same logic the
render VS preamble does (firstDrawInTile-bounded binary search →
`(slot, _local, instId, vid)`) and verifies every storage-buffer read
address the VS would compute is in-bounds AND reads finite data. Also
applies the slot's `ModelTrafo` to the looked-up vertex Position and
checks the world position lies within a sane radius of the trafo's
translation column.

Catches: out-of-bounds vertex pulling, NaN/Inf in matrices or
positions, corrupt drawHeader fields that pass `validateHeap` but
produce garbage at draw time.

Demo URL hook: `?simulate=N`.

## `probeBinarySearch(samples)` — GPU-vs-CPU bit-for-bit comparison

Dispatches a small compute kernel that **mirrors the render VS preamble
exactly** (firstDrawInTile-bounded binary search, all derived field
reads). Writes per-emit `(slot, drawIdx, indexStart, indexCount,
instanceCount, _local, instId, vid)` to a debug buffer. Then walks the
sample list on CPU, recomputes the same fields from a `mapAsync`-read
drawTable, and reports any disagreement.

This is the killer test. If anything in the GPU's view of the heap
state diverges from what `mapAsync` shows the CPU, we find it here.

**This is the test that found the cross-RO triangle bug.** The GPU's
binary search converged on `slot = numRecords` (the OOB sentinel) for
the last few emits per bucket, while CPU correctly capped to
`numRecords - 1`. Bug in the scan kernel's sentinel value.

Demo URL hook: `?probebs=N`.

## `checkTriangleCoherence(samples)` — exhaustive triangle-level check

For every triangle (3 emits) — exhaustive, no sampling — verifies:

1. All 3 emits resolve to the same slot (no cross-RO triangle).
2. All 3 emits resolve to the same instId within that slot (no
   cross-instance triangle).
3. The slot's `drawIdx` and `ModelTrafoRef` are bit-identical when
   re-derived from each of the 3 emits.
4. Each transformed vertex (ModelTrafo · Position) lies near the
   ModelTrafo's translation column.
5. Per-record: `indexCount × instanceCount` is a multiple of 3
   (otherwise the rasterizer assembles the last triangle of one
   record from the next record's first emit).
6. After all buckets: walks every (ref, allocBytes) pair across all
   drawHeaders, sorts, verifies no two distinct refs overlap (would
   mean the arena allocator handed the same bytes to two ROs).

Slow (~10 M triangles per scene) but conclusive.

Demo URL hook: `?triangles=N` (the parameter is ignored; the check is
exhaustive).

## How to add a new check

The pattern is the same every time:

```ts
async myCheck(): Promise<{ … }> {
  const stage = (src: GPUBuffer, size: number): GPUBuffer => {
    const c = device.createBuffer({
      size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    enc.copyBufferToBuffer(src, 0, c, 0, size);
    return c;
  };
  const enc = device.createCommandEncoder({ label: "myCheck" });
  const copy = stage(theBuffer, theBuffer.size);
  device.queue.submit([enc.finish()]);
  await copy.mapAsync(GPUMapMode.READ);
  const view = new Uint32Array(copy.getMappedRange().slice(0));
  // … walk view, compare to expectation, push issues …
  copy.unmap(); copy.destroy();
}
```

Snapshot mapped buffers via `.slice(0)` if you'll need their bytes
after `unmap()`. The mapped range is invalidated by `unmap()`.

Wire to the public surface:
- `HeapScene._debug.<name>` — the actual implementation.
- `HybridScene.<name>` — pass-through.
- `IRenderTask.<name>` — pass-through that aggregates over scenes.
- Demo's `main.ts` — URL hook + status-overlay reporting.
