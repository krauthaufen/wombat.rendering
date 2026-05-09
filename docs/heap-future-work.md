# Heap renderer — future work

Open design notes for the heap fast path. None of these are committed
plans; they're architectural threads worth pulling once a real workload
forces the issue. Listed in rough order of "likely to matter first."

## 0. Scope — what the heap path is for

The heap path's job: take many small-to-medium ROs that all want
similar treatment and crush them into ~one drawIndirect per
(pipeline-state, texture-set, family). It's not a replacement for
the legacy per-RO renderer; it's the fast path for the *tail* of
"all the little things" in a scene.

In a typical scientific-viz workload:

- **Legacy path** (`isHeapEligible(ro) === false`): the load-bearing
  bespoke cases. 10B pointclouds, photogrammetry mesh tiles,
  streaming residency layers, anything where the user wants to
  manage their own `GPUBuffer`s, run their own compute, do their
  own LOD. Heap path doesn't try to compete here.
- **Heap path** (`isHeapEligible(ro) === true`): markers, labels,
  gizmos, brushes, axis indicators, vehicle props, hover
  affordances, GIS feature symbols, point/line clouds below some
  threshold, any RO whose total payload fits comfortably (see §2).
  Mixed effects, mixed trafos, mixed textures (with §6 + §9).
  Hundreds to tens of thousands of these, all collapsing to ~3
  drawIndirect calls.

The success metric isn't "render a billion points faster than your
hand-rolled C++." It's "make the tail effectively free so users
stop architecting around the small-stuff overhead."

`isHeapEligible` evolves from a defensive screen ("reject obvious
unsupported wedges") into a positive declaration ("this RO opts
into the fast path because it fits"). Anything that fails the
predicate sails through the legacy renderer untouched.

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

## 5b. Value-equality for constant avals (pool-level)

Today's pools (UniformPool, IndexPool, AtlasPool) key on aval
**identity**: `WeakMap<aval, entry>`. Two `AVal.constant(matrixA)`
calls produce two distinct avals that share neither the entry nor
the underlying GPU allocation, even though their values are
identical.

This has bitten us repeatedly. The viewProj.map fix (per-RO
`.map(trafoToM44f)` defeating identity for ViewProjTrafo) and the
indices.map fix (per-RO `.map(IBuffer→Uint32Array)` defeating
identity for index buffers) were both symptoms of users
constructing fresh avals where the values were dedupable. Each
fix was a manual hoist; the next user hits the same gotcha.

**Generalize: dedup constant avals by value.**

`aval` already exposes `isConstant: boolean`. Pools become hybrid:
- `WeakMap<aval, entry>` for non-constant avals (their value can
  change; identity is the only stable key).
- `Map<valueHash, entry>` for `aval.isConstant === true` —
  multiple constant avals with the same value collapse to one
  entry.

**Mechanism:**

On `acquire(aval, value)`:
- If `!aval.isConstant`: identity lookup as today.
- If `aval.isConstant`: hash `value`, look up by hash. Hit →
  refcount++ on existing entry, return its ref. Miss → allocate,
  register in both maps (so the same `aval` reference resolves
  fast on subsequent acquires too), refcount = 1.

On `release`: refcount--; at 0, drop from both maps.

**Per-pool key strategy** — the cost / win ratio depends on
payload size, so each pool picks its dedup mechanism:

- **UniformPool** (small payloads — mat4 = 64 B, vec4 = 16 B,
  scalars). Hash bytes via xxhash / FNV-1a. ~50–100 ns per
  acquire. Catches the `AVal.constant(new Float32Array([…]))`
  pattern where two distinct allocations have identical values
  (common when user code constructs uniforms inline per RO).
- **IndexPool** (large payloads — Uint32Arrays often kilobytes
  to megabytes). **ArrayBuffer-tuple key**:
  `(arr.buffer === b.buffer && arr.byteOffset === b.byteOffset
   && arr.byteLength === b.byteLength)`. Doesn't catch "two
  distinct ArrayBuffers with identical content," but that's a
  rare authoring pattern; the common case is "one ArrayBuffer
  shared across multiple typed-array views or multiple aval
  wrappers," which IS caught. Hashing kilobytes of indices on
  every acquire is wasteful when this O(1) test handles the
  realistic case.
- **AtlasPool** (image data — usually large `ImageBitmap` /
  `ImageData`). Same: ArrayBuffer-tuple key. The "one
  ImageBitmap shared across ROs" pattern is exactly the common
  case for atlasing — N markers all pointing at one decoded
  PNG. Hashing every upload is wasteful and not worth catching
  the "two ImageBitmaps with the same pixels" edge case.

64-bit hash + post-hash byte comparator eliminates collision
risk for the UniformPool case without much cost. Or skip the
comparator and live with theoretical 1-in-2⁶⁴ collisions; for
our use case, fine.

The pool's `valueKey` hook abstracts this — each pool plugs in
its preferred strategy at construction. Tiny scalar pools (if
any) can stay identity-only.

**User-visible effect:**

The natural authoring pattern starts working:

```ts
makeRO({ modelTrafo: AVal.constant(trafoToM44f(t)), ... })
// fresh aval per RO. Pool dedups by value, so identical
// trafos collapse to one allocation. No hoisting required.
```

The "fresh `.map(...)` per RO blows up your memory" footgun goes
away. Users stop having to think about aval identity for the
common-case "I'm just wrapping a value." Reactive avals (cval,
custom) still need explicit sharing where intended, but the
constant case stops biting.

This is a small focused change — ~50 LOC per pool plus a shared
hash utility. Pairs naturally with the AtlasPool aval-reactivity
work (item 0 of textures plan, also pending).

## 5c. memoMap — pure-function dedup at the aval layer

(Lives in `wombat.adaptive`, not the rendering package — but in
the same architectural thread as §5b. Listed here because the
heap path is the loudest victim of the identity gotcha it fixes.)

§5b dedupes constant avals by value. The remaining identity
gotcha is **reactive avals derived via `.map(f)`** where `f` is
pure but called fresh per call site:

```ts
// Per RO:
const vp = viewProj.map(trafoToM44f);  // fresh derived aval each call
makeRO({ uniforms: { ViewProj: vp }, ... });
```

50K ROs → 50K distinct `vp` avals computing the same trafo. Pool
sees 50K identities, allocates 50K entries. The original viewProj
.map fix was to hoist `viewProjM44 = viewProj.map(trafoToM44f)`
out of the loop — manual identity sharing. Same shape as the
constant-aval issue; same systemic fix wanted.

**The trick: function identity is a free key (== is reference
equal in JS), but only safe to dedupe when `f` is pure / has no
captured state.**

JS doesn't expose purity statically. Three cases:
1. Module-level pure function (`trafoToM44f`): same ref
   everywhere. Dedup correct and useful.
2. Closure capturing variable: different ref per call, different
   behavior. Dedup would be incorrect.
3. Inline-pure lambda: pure but fresh ref. Same behavior as 1,
   but not identity-equal — can't be deduped without a content
   hash, and `.toString()` hashing isn't safe (closures'
   captured variables don't appear in source).

**Solution: opt-in via `memoMap`:**

```ts
aval.memoMap(f)   // user asserts f is pure + reference-stable
```

Internally: `WeakMap<aval, WeakMap<Function, aval>>`. First call
with `(av, f)` creates and caches; subsequent calls return the
same reference. Both keys weakly held; cache entries die with
their sources.

User opt-in is fine — same shape as React's `useMemo` (caller
asserts dependency-correctness) or Reactor's `cache`. One-line
API note: "use memoMap for pure module-level functions; plain
map for closures with captured state."

**Combined with §5b, the identity-footgun surface closes:**

| Aval shape | Today | With §5b + §5c |
|---|---|---|
| `AVal.constant(value)` per call, same value | N entries | 1 entry (5b) |
| Reactive aval `.map(pureFn)` per call | N entries | 1 entry (5c) |
| Reactive aval `.map(closure)` per call | N entries | N entries (correct — different behavior) |

Users stop having to think about identity in the
common-pure cases. Reactive-with-captured-state remains explicit,
which is the right level — reading code with a closure should
prompt the question "what does this capture."

Implementation cost: ~30 LOC in `wombat.adaptive`. Zero changes
in rendering layer; the existing identity-keyed pools just see
fewer distinct avals.

## 6. Uber-shader families with shared header pool

Currently the bucket key is `(effect, pipelineState, textures)` — two
ROs with different effects always fall into different buckets even if
their pipeline state and texture binding match. Encode = N buckets =
N drawIndirect calls.

Goal: collapse multiple effects into one bucket / one drawIndirect.
Earlier framing was "uber-shader unification" — merge effects into
one shader by unifying their schemas. Cleaner approach: don't unify
schemas at all. Each effect keeps its native drawHeader layout. The
machinery that gets shared is the storage and the dispatch, not the
data format.

**Mechanism:**

- One **shared `drawHeaders` pool** (heap-style, variable-sized
  allocations) replaces today's per-bucket header buffer. Each RO's
  drawHeader gets allocated by byte offset, size = its effect's
  declared header size.
- Each draw record carries `(firstEmit, layoutId, drawHeaderRef,
  indexStart, indexCount)`. `drawHeaderRef` = byte offset into the
  pool. `layoutId` = which effect this RO belongs to within the
  family.
- The **uber-shader** has a top-level switch on `layoutId`. Each case
  is the compiled-in code for THAT effect, including its specific
  drawHeader load expressions. No runtime layout interpretation —
  the layout knowledge is baked into each branch.
- **Bucket key reduces to `(uber-shader-family, pipelineState,
  textureSet)`**. Effects within the family share a bucket and a
  drawIndirect.

**Cost:** one extra storage read per VS — `drawHeaderRef` from the
draw record. Same shape as the indirections we already pay
(`heapU32[ref / 4u + offset]` etc.); essentially free.

**Branch divergence** when adjacent emit slots have different
layoutIds. Mitigated by sorting the drawTable by layoutId during
the scan pass so consecutive ranges share a branch. Cheap,
optional.

**Compile time** scales with branch count in the uber-shader, but
shape stays linear: each effect contributes its own block. Bounded
by partitioning effects into "families" (compatible pipeline state
+ texture shape).

**Wins:**
- Bucket count collapses to ~1 per family/PS/texture combo.
- No drawHeader-layout unification needed; no wasted bytes from
  union-padding; no migration when adding a new effect to a family.
- Adding a new effect = recompile the family's uber-shader + assign
  a new layoutId. Data formats untouched.

The "family" framing matters for compile cost — one mega-uber-shader
for the entire app would be huge. Realistic granularity: one family
per (pipeline-state class, texture-binding shape). Geodetic-df32
becomes a natural family.

**Trace-based, opt-in, async compile:**

Don't merge upfront — do it adaptively when the workload says it
helps. Keep separate buckets by default; track per-scene stats:
`(bucketCount, slotsPerBucket histogram, encodeTime)`. When
`bucketCount > N AND median(slotsPerBucket) < M` (something like
N=8, M=200), the death-by-thousand-tiny-buckets shape is real and
merging is worth the compile cost.

Trigger logic:
- Pick the family with the most buckets.
- Kick off `device.createRenderPipelineAsync(...)` for the merged
  uber-shader in the background. Non-blocking; encode keeps running
  with the current separate buckets.
- When the pipeline is ready, the next encode swaps. Concat-and-
  renumber the old per-bucket drawTables into one unified table and
  re-run the prefix-sum pass; copy drawHeaders into a unified pool.
  Old buckets and their per-bucket pipelines drop their references.
- Don't unmerge on subsequent workload shifts. One-direction
  transition; keeps the runtime state machine small.

Failure modes:
- Compile error on the merged uber-shader → abort, keep separate
  buckets, log + remember "this family doesn't merge cleanly" so
  future scenes don't retry.
- Compile time > budget (say 1s) → record the cost; next scene with
  the same family can either pre-warm during idle frames or skip
  merging entirely.

The pre-merge state is what's already shipping — so this is a pure
addition. Worst case (no buckets ever cross the threshold) costs
nothing. Best case (heavy material variety) collapses encode to
near-O(family) regardless of effect count.

## 7a. Derived uniforms as a first-class DSL primitive

The implementation in §7 (next) handwrites a WGSL compute kernel,
wires it manually into a bucket, and declares its dependencies in
TS code. That's fine for ModelView in isolation but doesn't scale —
every new derivation is bespoke runtime + shader plumbing.

The principled version: extend wombat.shader so users *declare*
derivations in the same DSL they already use for vertex/fragment
stages.

```ts
deriveUniform("ModelView",     (u) => u.ViewTrafo.mul(u.ModelTrafo));
deriveUniform("NormalMatrix",  (u) => u.ModelView.upper3x3.inverse.transpose);
deriveUniform("ScreenBBox",    (u) => u.ViewProj.mul(u.WorldBBox));
deriveUniform("DistanceFade",  (u) => fade(u.WorldPos.distance(u.CameraPos), u.FadeNear, u.FadeFar));
```

The wombat.shader compiler:

- Detects "this lambda is a derivation" — different stage from
  `vertex()` / `fragment()` but reuses the same IR substitution +
  DCE machinery.
- Statically analyzes the lambda's reads (`u.X`) → emits the
  dependency set.
- Emits a `@compute @workgroup_size(N) fn deriveModelView(...)`
  entry point that reads the input slots from the arena, applies
  the user's logic, writes the output slot.
- Plumbs into the runtime's per-class dirty-list machinery
  automatically. No manual bucket wiring.

**Scene-graph inheritance turns this into a programming model:**

- A `Camera` SG node publishes `ViewTrafo`. Children inherit.
- A `Geodetic` modifier publishes `precision: df32`. The compiler
  picks compensated-arithmetic codegen for any derivation
  downstream.
- A `Trafo` SG node publishes `ModelTrafo` (composing with parent's).
- A leaf RO that *uses* `ModelView` (in its vertex shader) gets it
  derived automatically — the SG tells the runtime which avals
  feed it and which precision policy applies.

The user never writes a compute pass. They write a function. The
system figures out:
- Which avals are inputs (from the lambda's reads).
- Which RO slots are dirty when those avals mark.
- What WGSL to emit (with df32 promotion if a `Geodetic` ancestor
  demands it; with whatever precision policy applies otherwise).
- When to dispatch (sparse per-class dirty lists, §7).

**Why this is strategically distinctive:**

Game engines can't offer this because their shaders are slot-
configuration; arbitrary derivations don't fit the material-graph
model.

FRP libraries can't offer this because they don't compile to GPU;
"derive ModelView from View × Model for 50K objects per frame" in
JS is a non-starter.

Scientific-viz toolkits don't offer this because they don't have
arbitrary shader code at all; their derivations are built-in.

The combination — reactive scene composition + custom shader DSL +
GPU-evaluated derivations — is the unique asset wombat brings. The
position statement falls out: **"a programming model where
reactive scene composition, custom shaders, and derived GPU state
are all one DSL, evaluated lazily across CPU + GPU according to
where the work belongs."**

LOD selection, occlusion testing, animation pose, distance fade,
splat sorting, custom culling — all become one-line declarations.
Each composes through the SG. Each gets sparse reactive change
tracking + GPU compute for free.

The implementation backbone (§7) is what this DSL extension lowers
to. Build §7 first; this section is the user-facing API once the
machinery exists.

## 7. GPU-computed derived uniforms (ModelView, df32 geodetic)

Aardvark has a "derived uniforms" concept: ModelView, NormalMatrix,
etc. are not stored per-RO but computed from base inputs. The
geodetic precision trick is the load-bearing case:

> Geodetic coordinates put data at ~6.4×10⁶ m from origin. f32 mantissa
> precision at that magnitude is ~0.4 m — useless for cm-scale
> rendering. Solution: compute `view × model` in double precision so
> the camera-relative offsets cancel cleanly, then truncate the result
> to f32 for the shader. The shader sees a small-magnitude matrix and
> all error compounds at f32 scale, not at planet scale.

CPU implementation works but means recomputing ModelView for every RO
on every camera move. With 50K ROs that's 50K mat4 × mat4 multiplies
+ 50K writeBuffer uploads per frame. Terrible.

**GPU-side via "df32" / double-single:**

Represent each scalar as `(hi: f32, lo: f32)` where value ≈ hi + lo.
Compensated arithmetic via Knuth's TwoSum (add) + Dekker's TwoProduct
(multiply). Not full IEEE 754 double — but enough to cancel the
big-magnitude offsets and produce a clean small-magnitude result.

**Heap-renderer integration:**

Add a "derived uniform" field type to BucketLayout:
```ts
{ name, kind: "derived", dependencies, computeKernel, outputType }
```

- Pool allocates output storage normally (one slot per RO).
- Dependencies (e.g. ModelTrafo_df32, ViewTrafo_df32) live in the
  arena like any other uniform but with df32 type.
- A new compute pass `computeDerived` dispatches one thread per RO
  per dirty dependency. The thread reads inputs via existing
  arena-fetch helpers, runs the kernel (mat4 × mat4 compensated
  product), truncates to f32, writes to the output slot.
- Reactivity is sparse and dependency-tracked. Each derived class
  maintains a dirty list of RO slot indices whose output needs
  recomputation. Per-RO dependency avals (e.g. that one RO's
  `ModelTrafo`) register marking callbacks that add THAT slot.
  Scene-global dependencies (`ViewTrafo`, `ProjTrafo`) toggle a
  "global dirty" flag covering all slots in one stroke.

  Per frame:
  - Dirty list empty AND no global flag → skip the compute dispatch
    entirely. Zero per-frame work for static-scene-stationary-camera.
  - Otherwise → upload the slot-index list, dispatch one compute
    thread per dirty slot (or all slots when the global flag is set),
    clear the list.

  Camera orbit → ViewTrafo dirty → one full dispatch per frame
  (~50K threads ≈ 1 ms). Single moving vehicle → one thread per
  frame. Idle viewer → nothing dispatched. The compute pass costs
  exactly what the dependency graph says it should.

  Naturally extends to N derived uniforms — each class tracks its
  own dirty list, the runtime iterates only the classes with work.
  Renderers that don't use derived uniforms pay nothing.

**Costs:**

- Storage: 2× per df32 input. ModelTrafo = 128 B instead of 64. Output
  ModelView still 64 B (plain f32). Net: maybe +64 B per RO for the
  static input, no per-frame ViewProj ref/upload churn.
- Compute: ~16 mat4 entries × ~10 ops per df32 multiply-add ≈ 160 ops
  per RO. At 50K ROs → 8M ops per camera move. Sub-millisecond on
  any modern GPU.
- Bandwidth: one camera move uploads 128 B for the new ViewTrafo
  instead of N writeBuffers for N ModelView matrices. Strict win.

**Architectural cleanup:**

User-facing API simplifies: you provide ModelTrafo + ViewTrafo +
ProjTrafo separately (in df32 where needed), the renderer derives
the correctly-precision-managed combined matrices. The current
demo's `makeUniforms` would shrink considerably.

This is one of the biggest user-visible wins available — geodetic
rendering at 50K+ objects with cm precision and no per-RO CPU work
on camera moves.

## 8. Per-vertex (drawIdx, vid) lookup table

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

## 9. Texture array atlasing

The merging story in §6 has a hard precondition: ROs that share a
bucket must reference the SAME concrete texture/sampler resources.
Two ROs both wanting "one albedo + one sampler" can share a bucket
only if they want the *same* albedo. WebGPU 1.0 has no bindless
textures, so heterogeneous-texture scenes still bucket per
material — exactly the case that breaks the "1 drawIndirect for
everything" promise.

The escape hatch available today: pack scene textures into a few
`texture_2d_array<f32>` resources. Per-RO state in the drawHeader
adds `(arrayLayerIdx, uvScale, uvBias)`. The shader samples
`textureSample(atlas, sampler, vec3(uv * scale + bias, layer))`.

**The bargain — spirit-aligned with the rest of the heap path:**

Pay upfront / async cost (atlas packing, residency management) for
runtime uniformity that makes everything downstream collapse to one
drawIndirect. Same trade as the GPU prefix-sum: bookkeeping cost
moves to the periphery so the hot path stays branch-free and
trivial.

**Binning:**

- One `texture_2d_array` per (size class, format) combo.
  Power-of-two size classes: 256², 512², 1024², 2048², 4096². Most
  scenes need ≤ 5 distinct combos.
- ROs whose textures land in the same combo share a bucket.
- Texture variation drops out of the bucket key entirely; bucket
  key becomes `(uber-shader-family, pipelineState, atlasGroup)`
  where `atlasGroup` covers all RO atlases the bucket references.

**Costs:**

- **Atlas packing**: incremental layer assignment. Pack on demand,
  evict LRU when full. Standard texture-cache pain, well-understood
  problem (UE5 virtual textures, Sebastian Aaltonen's talks). Not
  novel; just real.
- **Resolution mismatch**: small textures pack into the next-larger
  size class. Wastes some memory; same proportional waste as any
  pow2 allocator.
- **MIP regen**: `generateMipmaps` across array layers requires a
  custom compute or per-layer render pass. Doable; one-time per
  layer-write.
- **Sub-region sampling**: shader does `fract(uv * scale + bias)`
  to handle wrapping/repeat manually since the array layer is the
  whole texture. One extra ALU per sample.

**What's gained:**

- The entire scene's textured ROs collapse to one drawIndirect per
  (size class, format). Typical scenes: 1–3 drawcalls covering all
  materials.
- Scene heterogeneity stops scaling encode time. Cost is paid in
  atlas churn (bounded, async, off-the-hot-path).

**Where this is the right trade:**

For finite-and-known texture sets (GIS feature symbols, marker
icons, vehicle skins, photogrammetry tile thumbnails) — exactly
the kind of mid-scale heterogeneity the heap path targets. Atlas
fits naturally; the working set is bounded.

For unbounded streaming (procedural materials, infinite zoom) —
this is where you'd want true bindless or virtual textures. Not
available in WebGPU 1.0; revisit when the spec catches up.

**WebGPU 1.0+ migration path:**

Bindless texture support, when it lands, replaces the atlas
machinery without changing the user-facing API. The drawHeader
already carries an "image reference" (today: layer index, future:
bindless handle) so the migration is internal. Atlas residency
management stays useful as a fallback for older browsers.

This is the right escape hatch *today* AND a clean migration path
to the eventual bindless future. Spirit-aligned with the rest of
the architecture: pay the price in management complexity at the
periphery, gain unconditional uniformity in the hot path.
