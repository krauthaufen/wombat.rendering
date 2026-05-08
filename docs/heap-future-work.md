# Heap renderer — future work

Open design notes for the heap fast path. None of these are committed
plans; they're architectural threads worth pulling once a real workload
forces the issue. Listed in rough order of "likely to matter first."

## 1. Tile-bounded binary search (in flight)

The VS prelude binary-searches `drawTable[*].firstEmit` to derive
`(drawIdx, vid)` from `vertex_index`. Today the search is unbounded:
`hi = arrayLength(&drawTable)/4u - 1u`, so cost is `log2(numDraws)` —
~14 storage reads per vertex at 10K draws.

Half-way fix: precompute `firstDrawInTile[T] = first draw whose
emit-range covers T*K` for tile size K=64. VS prelude becomes:

```wgsl
let tileIdx = emitIdx >> 6u;
var lo = firstDrawInTile[tileIdx];
var hi = firstDrawInTile[tileIdx + 1u];
// bounded binary search …
```

- Memory: `(numTiles + 1) × 4B` ≈ 45 KB for 720K emits at K=64.
- Cost: 2 reads for `lo`/`hi` + `log2(drawsPerTile)` BS steps.
  - Cubes (24 verts) → drawsPerTile ≈ 3 → ~2 BS steps.
  - Total ~4 reads vs ~14. ~3.5× fewer per-vertex reads.
- Built by an extra compute dispatch after `addOffsets`: one thread per
  tile, each binary-searches drawTable to find the covering draw.
  ~150K reads total for a 10K bucket. Sub-ms on GPU.
- Worst case: 1-vert draws → drawsPerTile = K = 64 → 6 BS steps. Still
  bounded; degrades gracefully.

Default K=64. Configurable per scene if a workload wants to tune.

## 2. Large-object eject

Single huge meshes (terrain, point clouds, photogrammetry) embedded in
the heap arena force the arena's pow2 grow to allocate gigantic
contiguous buffers + GPU-side copy of the whole arena on resize. They
also don't benefit from the heap's batching — a one-RO bucket emits
the same one drawIndexed it would have under the legacy path.

Fix: add a size predicate to `isHeapEligible`. ROs whose total
attribute+index byte payload exceeds a threshold (e.g. 16 MB) route
through the legacy per-RO renderer with their own dedicated
GPUBuffers. ~5 LOC change in `heapEligibility.ts`.

## 3. Chunked arena heaps

Right design once we cross ~64 MB total arena. Replaces the single
GrowBuffer with a list of chunks.

**Chunk policy:**

- New chunks start small (1 MB) and pow2-grow internally up to a cap
  (64 MB). Initial small chunks avoid VRAM blowup for small scenes.
- Once a chunk seals at 64 MB cap, alloc opens a new chunk with the
  same policy.
- Allocator iterates chunks low-to-high, first-fit. Live chunk only
  grows when no chunk has room. Sealed chunks reuse freed space via
  their own freelist.

**Encode-side trade:**

Bucket gains `chunks: Chunk[]` instead of one `(buffer, drawTable)`.
Each chunk owns its own drawTable + indirect + bind group. Render loop
becomes:

```ts
for (const c of bucket.chunks) {
  pass.setBindGroup(0, c.bindGroup);
  pass.drawIndirect(c.indirectBuf, 0);
}
```

Encode goes from O(buckets) to O(buckets × chunks). At 64 MB chunks
and typical scenes: chunks-per-bucket = 1–3. Negligible.

**Affinity policy for assigning records to chunks:**

When addDraw fires, pick the chunk to host all of this RO's data
(attrs + indices + drawTable record) by first-fit on total RO bytes.
A single draw never spans chunks — drawTable records are partitioned
by chunk trivially, and each chunk's bind group only references its
own buffers. Simpler than striping.

## 4. Free-block management

Today each arena holds a freelist `{off, size}[]` sorted by offset.
`alloc(size)` is O(N) first-fit; `release` is O(N log N) (splice +
re-sort + coalesce). Fine for static / slow-churn; visible at high
churn.

**Replace with a balanced BST** keyed on `(size, offset)` for alloc
lookups + a `Map<offset, node>` for coalesce neighbor lookups. Best-
fit alloc is `lower_bound(size)` → O(log N). Release is two map
lookups + at most two merges → O(log N).

Implementation: ~150 LOC of RBTree / AVL in TS, or pull in
`js-sdsl` / `sorted-btree`. Not urgent.

**Future-future**: pow2 size classes → fixed array of bucket lists,
O(1) alloc/release. Trades external fragmentation for some internal
fragmentation. Worth swapping in only if profiling shows the BST as
hot.

## 5. Defrag / compaction

The cursor-only-grows model means a long-lived scene with high churn
ends up with a high watermark = peak live bytes ever, even if the
current live set is small. Two paths:

- **Periodic compaction**: walk live allocations, relocate to a fresh
  arena chunk, fix all refs in drawHeap. Triggered when fragmentation
  ratio (= free / capacity) exceeds a threshold. O(live bytes).
- **Drop empty chunks**: under chunked heaps (item 3), when a sealed
  chunk's live count drops to 0, release the GPUBuffer. Cheaper than
  defrag and addresses the same problem in the typical case.

Drop-empty-chunks pairs naturally with affinity allocation — RO data
locality keeps chunks either live or near-empty, rarely
bimodally fragmented.

## 6. Uber-shader unification

Observation: every effect in the heap path that doesn't use textures
shares the same input signature (Positions, Normals, … + uniforms via
arena). Today each unique effect compiles its own shader module +
pipeline.

Idea: a single uber-VS + uber-FS with a top-level switch on a
`shaderId` field in the drawHeader. One pipeline, one bind group
layout, all non-textured ROs collapse into one bucket regardless of
effect.

Wins:
- Bucket count drops dramatically. Encode loop becomes one drawIndirect
  for the entire scene's non-textured fast path.
- No per-effect shader-module cache, no per-effect pipeline cache.

Costs:
- Shader compile time scales with branch count. With many effects,
  the merged shader could be huge and slow to compile (one-time, but
  meaningful for cold start).
- Branch divergence in the warp: vertices with different shaderIds
  in the same warp execute both branches. With good slot-locality
  (same effect = contiguous drawTable region) divergence is rare,
  but worst-case is real.
- Loses per-effect dead code elimination — uniforms only one effect
  reads still need to be loadable (or guard every load behind the
  shader-id branch, which costs predication).
- The IR rewriter's `linkFragmentOutputs` pass currently DCEs unused
  outputs per-effect. Uber-FS would have to either preserve all
  fragment outputs or do DCE against the union of effects' outputs.

Realistic path: apply uber-shader unification *within a small set of
"compatible" effects* (e.g. lambert + flat + wave: identical attribs,
same FS output, just different math). Different texture/sampler
shapes still bucket separately. Could provide 2-4× bucket reduction
for scenes that mix a few similar materials, without paying full
uber-shader compile cost.

## 7. Per-vertex (drawIdx, vid) lookup table

The full version of item 1 — replace per-vertex binary search with
ONE storage read of a precomputed `(drawIdx, vid)` per emit position.

Memory: `totalEmit × 8B`. At 720K verts → 5.76 MB. Real but not
crippling.

Per-vertex cost: 1 storage read. Hard to beat.

Built by a compute pre-pass: each thread = one emit position, does
one binary search on drawTable, writes its result. Total work =
totalEmit × log2(numDraws) reads. Bigger than tile-index build (item
1) by a factor of K.

Worth chasing only if the tile-bounded BS still leaves the GPU
bottlenecked, AND memory headroom exists. For mobile GPUs, the
~5 MB extra pressure could push the working set out of cache and
make the "fast" version slower — measure before assuming.
