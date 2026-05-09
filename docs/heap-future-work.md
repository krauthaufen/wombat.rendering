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
  Mixed effects, mixed trafos, mixed textures (with §6 + §9),
  per-RO instancing (with §10) — a 1000-tree forest with a shared
  mesh + 1000 distinct transforms collapses to one bucket and one
  drawIndirect. Hundreds to tens of thousands of these, all
  collapsing to ~3 drawIndirect calls.

The success metric isn't "render a billion points faster than your
hand-rolled C++." It's "make the tail effectively free so users
stop architecting around the small-stuff overhead."

`isHeapEligible` evolves from a defensive screen ("reject obvious
unsupported wedges") into a positive declaration ("this RO opts
into the fast path because it fits"). Anything that fails the
predicate sails through the legacy renderer untouched.

## 1. Tile-bounded binary search ✅ SHIPPED

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

## 5b. Value-equality for constant avals (pool-level) ✅ SHIPPED (scope reduced)

**Shipped scope:**

- **`__memo` constant-source bypass** (in `wombat.adaptive`):
  when every aval-typed input to a memoized combinator is
  `isConstant === true`, the trie is skipped entirely — `compute()`
  runs directly. Constants don't mark, so caching the derived aval
  serves only identity-sharing for downstream consumers; the
  trie's reactive job is moot. The realistic 95% case for
  memoization is reactive avals, which still flow through
  normally. ~15 LOC in `src/plugin/runtime.ts`. Covers `aval`,
  `aset`, `alist`, `amap` uniformly via duck-typed `isConstant`.

- **IndexPool value-dedup** (heapScene.ts): when an incoming aval
  has `isConstant === true`, the pool also keys by an
  ArrayBuffer-tuple `(buffer-id, byteOffset, byteLength)`. Two
  distinct constant avals wrapping the same `Uint32Array` view
  collapse to one allocation. Hashing kilobytes of indices on
  every acquire would be wasteful; the tuple test catches the
  realistic "one ArrayBuffer shared across many aval wrappers"
  pattern, which is the only one that matters for the heap path.
  Refcount semantics fixed: per-aval acquire count is now tracked
  separately from the entry's total refcount so multiple aliasing
  avals can share one entry without leaking byAval bindings.

- **AtlasPool value-dedup** (atlasPool.ts): same gating; the value
  key is the inner resource reference (`GPUTexture` for
  `kind:"gpu"`, `HostTextureSource` for `kind:"host"`) rather than
  an ArrayBuffer tuple, since `ImageBitmap` /
  `HTMLVideoElement` don't expose pixel buffers but are reference-
  comparable. Catches the typical "N ROs all pointing at one
  decoded PNG" pattern. AtlasEntry now carries `aliases:
  aval<ITexture>[]` so final release clears every aliased
  byAval binding. `repack` throws if called on a deduped entry —
  shouldn't happen via the reactivity loop (constants don't mark)
  but guards against manual misuse.

**Skipped scope:**

- **UniformPool** stays identity-only. Payloads are tiny (mat4 = 64 B,
  vec4 = 16 B); even N copies of identical constant uniforms across
  N ROs cost a few MB and don't break batching (bucket key is
  effect/pipelineState/textures, not pool refs). Hashing bytes per
  acquire isn't worth the cost when the constant-source bypass
  already prevents the upstream memo trie from filling with
  no-op entries.

**Tests:**

- `tests/plugin/transformed/memoization.test.ts` — three new
  bypass tests for aval (constant source → distinct derived,
  reactive source → still memoizes) plus aset variants.
- `tests/heap-atlas-dedup.test.ts` — two AtlasPool tests:
  same-host-source dedup, different-source disambiguation, plus
  release semantics across both aliases.

**Files:**

- `wombat.adaptive/src/plugin/runtime.ts` — bypass logic.
- `wombat.rendering/packages/rendering/src/runtime/heapScene.ts` —
  IndexPool with `byValueKey` map + per-aval acquire counter.
- `wombat.rendering/packages/rendering/src/runtime/textureAtlas/atlasPool.ts` —
  AtlasPool with `entriesByValueKey` + `aliases` per entry.

---

(Original plan kept below for historical reference.)

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

## 5c. memoMap — pure-function dedup at the aval layer ✅ SHIPPED (as hidden internal)

(Lives in `wombat.adaptive`, not the rendering package — but in
the same architectural thread as §5b. Listed here because the
heap path is the loudest victim of the identity gotcha it fixes.)

**Shipped state**: as a hidden internal API at
`@aardworx/wombat.adaptive/internal`. NOT a public method on
`aval`. The implementation moved to a single `__memo(keys, compute)`
runtime + 17 per-combinator helpers (`memoAvalMap`, `memoAsetFilter`,
etc.) covering all combinators of all four collection types
(aval/aset/alist/amap × map/bind/filter/collect/choose + zipN).

Users should NOT call these directly. The intended consumer is the
build-time transform plugin (§5d) which lowers user `.map(closure)`
calls into the appropriate `__memo([...keys], () => original())`
shape. Public collection interfaces stay untouched.

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

**Solution as shipped: hidden runtime substrate:**

```ts
// Internal API (plugin-only):
import { __memo, TAG_AVAL_MAP } from "@aardworx/wombat.adaptive/internal";
__memo([TAG_AVAL_MAP, "h:bodyHash", source, ...closureDeps], () => source.map(fn))
```

The `__memo` function uses a shared **`MemoTrie`** (generic
weak-keyed cache trie at `src/core/memoTrie.ts`): each level is a
`WeakMap<object, MemoTrie>`, leaf is a `WeakRef<object>`. Lookup
walks the path; insert nests as needed; null/dead level falls
through to a miss. Every key in the path is held weakly — source
aval, function, deps. The derived aval is held via WeakRef. Any
component dies → entry naturally collected. No FinalizationRegistry,
no manual cleanup.

The 17 per-combinator helpers (`memoAvalMap(av, f)`,
`memoAsetFilter(set, p)`, etc.) are convenience wrappers; the
plugin can also emit `__memo(...)` directly for arity flexibility
(n-ary zips, multiple closure deps).

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

## 5d. Build-time combinator transform ✅ SHIPPED

Vite plugin at `@aardworx/wombat.adaptive/plugin`. Detects user
calls to adaptive combinators (in 3 call shapes — method,
free-function, namespace) across all 17 combinator helpers, and
rewrites them at compile time to memoizing equivalents using the
hidden `__memo` runtime substrate from §5c.

**What it catches:**

| Combinator | Method `x.m(...)` | Free `m(...)` | Namespace `K.m(...)` |
|---|---|---|---|
| `aval × {map, bind, zipN}` | ✓ | ✓ | ✓ |
| `aset × {map, bind, filter, collect, choose, mapA, filterA, chooseA}` | ✓ | ✓ | ✓ |
| `alist × {map, bind, filter, collect, choose, mapi, filteri, choosei, collecti, mapA, filterA, chooseA, mapAi, filterAi, chooseAi}` | ✓ | ✓ | ✓ |
| `amap × {map, bind, filter, choose, mapA, filterA, chooseA}` | ✓ | ✓ | ✓ |

**What it does, per call site:**

1. Statically infers the source's collection kind from import
   context (no TypeChecker — pure syntactic analysis tracking
   imported `cval`/`cset`/`clist`/`cmap` constructors and namespace
   imports). Skips the call if kind is ambiguous.
2. Walks the callback body collecting free-variable references.
   Classifies each: callback-local (param / body-decl) → ignored;
   module-stable (import / known namespace / builtin global) →
   ignored (covered by code hash); inner-scope capture → goes into
   the deps array.
3. Computes deterministic FNV-1a 32-bit hash of the callback's
   trimmed source text.
4. Rewrites the call to:
   ```ts
   __memo(
     [opTag, "h:hash", ...sources, ...closureDeps],
     () => /* original call */,
   )
   ```
   The fallback closure preserves the exact original call shape;
   the plugin doesn't need to know each helper's signature.
5. Injects the `__memo` + tag imports at the top of the file.

**Cache key shape decision:** The emit deliberately OMITS the
function reference (`fn`) from the cache key path. Reasoning:
inline lambdas re-allocate per call site invocation; if `fn` were
a key, fresh lambdas would always miss. The body hash is the
stable function-identity proxy across distinct lambda allocations.

**Runtime cost per call site:**

- Hash is a string constant (zero runtime work to compute).
- Cache lookup is one MemoTrie walk through the key path
  (typically 3-5 levels for `[tag, hash, source, ...deps]`).
- Hit returns the cached aval directly; miss runs the original
  call once and caches.

**Known limitations:**

- **Closure-dep member-access chains** (`props.r`) reduced to
  the base identifier (`props`). Cache keys on `props` reference
  rather than `props.r` value — conservatively correct (props
  marks → cache invalidated) but slightly broader than ideal.
- **No idempotence guard.** Vite's single-transform-per-file
  pipeline prevents re-rewriting in practice; if the plugin ever
  runs twice on the same source, nested `__memo` calls would
  result. Not a real concern with standard Vite usage.
- **Class instances without `getHashCode()`/`equals()`** fall
  through to reference identity in the runtime. Aardvark
  value-types (V3f / M44f / V3i / etc.) do implement the
  duck-type and intern correctly; arbitrary user classes don't.
  Workaround: implement the protocol on classes whose instances
  are commonly captured-but-structurally-shared.

**Previously-known limitations now fixed:**

- ✓ Method calls on non-Identifier receivers
  (`cval(1).map(...)`) now resolved via recursive
  `inferKindFromExpression` on the receiver.
- ✓ `import * as X` namespace imports recognized for both
  subpath modules (`import * as ASet from ".../aset"` →
  `ASet.map(...)`) and the bare module
  (`import * as W from "@aardworx/wombat.adaptive"` →
  `W.AVal.map(...)`).
- ✓ `*A` aval-callback variants (`mapA`, `chooseA`, `filterA`,
  `mapAi`, `chooseAi`, `filterAi`) and `*i` indexed-callback
  variants (`mapi`, `choosei`, `filteri`, `collecti`) covered
  across all relevant collection kinds.
- ✓ **Body hash is whitespace-insensitive**. Hashing now runs
  on the AST-printed form (TypeScript printer with
  `removeComments: true`, fixed newline/indent), so
  `t=>t*2` and `t => t * 2` and a multiline-formatted
  equivalent all produce the same hash. Comments inside the
  callback are stripped before hashing too.
- ✓ **Plain-object closure deps** with primitive leaves
  (`{r:1, g:0, b:0}` and `[10, 20]` literals) intern by
  structural value via a runtime `SIMPLE_INTERN` map. Plain
  objects with `Object.prototype` and arrays with primitive (or
  recursively simple) leaves, depth ≤ 4, are serialized to a
  deterministic key string and assigned a stable opaque handle.
  Two structurally-equal captures from distinct call sites
  collapse to one cache entry. Symbols, functions, typed
  arrays, and exotic objects bail to reference identity.

**Reference implementations leaned on:**

- **React Compiler** (formerly React Forget) — same shape for
  `useMemo` / `useCallback`. Algorithm well-documented.
- **Solid.js reactivity transform** — similar AST-rewrite for
  reactive primitives.
- **Million.js** — memoization at JSX level.

**Tests:** 21 AST-shape tests at `tests/plugin/transform.test.ts`
verifying tag selection, hash literal presence, fallback-closure
preservation, closure-deps capture, type-ambiguous-skip, and
import-injection shape. Plus 35 behavioral tests at
`tests/plugin/transformed/` that apply the plugin via vitest
config and assert real memoization reference-equality + value
correctness end-to-end — including 5 hashable-types tests
(`hashable.test.ts`) covering structural-equality dedup of V3-shaped
value types, plain-object reference-identity fallback, same-instance
reuse, and an adversarial forced-hashCode-collision case proving
collision-safety. The behavioural suite caught a real runtime bug
(primitive cache keys crashing the WeakMap path) that the AST-shape
tests missed.

**Files:**

- `src/plugin/index.ts` — Vite plugin entry exposing
  `adaptiveMemoPlugin()`.
- `src/plugin/transform.ts` — pure AST transform (~811 LOC):
  import tracker → local-binding kind inference → call-site
  detector → closure-deps walker → FNV-1a body hash → __memo
  rebuilder → import injector.
- `src/plugin/runtime.ts` — key-interning shim. MemoTrie needs
  object keys (WeakMap). The runtime collapses the body-hash string
  + all primitive closure-deps into ONE type-tagged interned
  `{ k: string }` per unique tuple. Value-typed objects with the
  Aardvark `getHashCode()` + `equals()` duck-type (V3f / M44f /
  V3i / etc.) are interned via a **hash-bucket-with-equals** scheme:
  `Map<hashCode, Array<{value, key}>>` — on hash collision, the
  bucket is walked calling `equals()` to find a match, so distinct
  values always get distinct opaque handles regardless of hashCode
  collisions. Plain objects (no hashable protocol) keep reference
  identity through the WeakMap path.

**Value-typed dedup correctness note:** content-hashing the cache
key (FNV of the value bytes) was rejected because a 1-in-2³² hash
collision would surface as a *wrong cache hit* — silent wrong
results, not just a missed dedup. The hash-bucket scheme uses
`getHashCode()` only as a bucket selector and `equals()` as the
sole equality primitive, so collisions degrade to bucket walks
without ever returning a wrong handle.

**Layered story (status of all four phases):**

1. **§5b** — value-equality dedup at pool level. ✅ Shipped
   (reduced scope: IndexPool + AtlasPool only; UniformPool
   skipped as not worth the per-acquire hash cost). Plus
   constant-source bypass in `__memo` so the upstream trie
   doesn't fill with no-op entries.
2. **§5c** — runtime `__memo` substrate. ✅ Shipped as hidden
   internal at `@aardworx/wombat.adaptive/internal`.
3. **§5d** — build-time transform. ✅ Shipped at
   `@aardworx/wombat.adaptive/plugin`. Subsumes 5c's manual
   surface — users never reach for `__memo` directly.

The identity footgun is now gone end-to-end:
- Constant avals: `__memo` skips the trie; pool-level dedup
  collapses the realistic "N ROs share an ImageBitmap / index
  array" patterns into single allocations.
- Reactive `.map(closure)` avals: memo trie shares derived avals
  by `(tag, body-hash, source, deps)`.
- Reactive `.map(closure-with-state)`: trie disambiguates by
  closure-dep identity — correct distinct entries.
- Closure deps with structural value (V3f / M44f / V3i; plain
  `{r,g,b}` literals; `[10, 20]` arrays): runtime intern by
  structure. Two equal-but-distinct captures collapse to one
  cache entry.
- Body hash insensitive to whitespace and comments — codebases
  with mixed formatting still dedup correctly.
- Plugin call-shape coverage: method form on any receiver
  (including `cval(1).map(...)` chains), free-function form,
  named-namespace form, `import * as X` namespace form (subpath
  AND bare-module compound), plus `*A` aval-callback and `*i`
  indexed-callback variants across all four collection kinds.

## 6. Uber-shader families with shared header pool

Currently the bucket key is `(effect, pipelineState, textureSet,
perInstanceAttrSet)` — two ROs with different effects always fall
into different buckets even if everything else matches. Encode = N
buckets = N drawIndirect calls.

Goal: collapse multiple effects into one bucket / one drawIndirect.
Earlier framing was "uber-shader unification" — merge effects into
one shader by unifying their schemas. Cleaner approach: don't unify
schemas at all. Each effect keeps its native drawHeader layout. The
machinery that gets shared is the storage and the dispatch, not the
data format.

### v1 PoC scope (agreed 2026-05-09)

The proof-of-concept implements the strong form: **bucket key
reduces to `(familyId, pipelineState)`**. `effect`, `textureSet`,
and `perInstanceAttrSet` all fold inside the family via layoutId
dispatch. Concretely:

1. **Adaptive collection from the RO set.** `buildHeapScene` walks
   the RenderTree leaves, collects every distinct `Effect`, builds
   the family. No user-supplied family list.

2. **Anonymous Varying0..N packing.** Each per-effect VS gets
   rewritten at family-build time: named varying writes
   (`out.WorldPositions = …`) become slot writes (`varyings[k] = …`)
   with a slot map computed once for the whole family. Family VsOut
   is `array<vec4<f32>, N>` where N covers the largest single
   effect's varying budget. Per-effect FS reads similarly rewritten.
   No struct field clashes, no wasted struct slots; semantics-free.

3. **drawHeader schema = union of all per-effect schemas.** Each RO
   populates only its effect's slots; layoutId selects what's live.
   New mandatory `layoutId: u32` field threaded through the
   drawHeader at a fixed offset.

4. **Atlas-route everything.** `textureSet` axis disappears: every
   heap-eligible RO routes through AtlasPool. Effects that don't
   sample a texture get a 1×1 white sub-rect auto-bound; their FS
   doesn't read it (DCE drops the dead sample if any). Cost is
   one extra reserved sub-rect per scene.

5. **perInstanceAttrSet inside the family.** The drawTable record
   already carries per-record `instanceCount`; non-instanced effects
   coexist with instanced ones in the same bucket — layoutId picks
   which path the VS dispatch takes. Family's wrapper VS reads
   `instance_index` AND `instId` (from the megacall prelude); each
   per-effect helper reads whichever it needs.

6. **Static rebuild.** Family compiled once at scene-build. Effect
   set is fixed for the scene's lifetime in v1.

7. **Lifecycle**: Eager compile, single render pipeline per
   `(familyId, pipelineState)`. Synchronous family-build.

### v2 punts (deferred)

- **Reactive rebuild on RO add/remove** — recompile family when the
  effect set changes. Likely cheap (compile is sub-100ms), but
  needs a swap mechanism that doesn't drop frames mid-rebuild.
- **Async opt-in family compile** — for very large families,
  compile in the background and keep the unmerged buckets running
  until the merged pipeline is ready.
- **Multi-pipeline per family shader** — one WGSL module driving
  multiple GPU pipelines distinguished by pipelineState (currently
  one pipeline per `(familyId, pipelineState)` pair, so each
  pipelineState today recompiles the family WGSL).
- **Trace-based opt-in** — the original "merge only when bucket
  count crosses a threshold" framing. v1 always merges; the
  pessimistic case (very few ROs, large effect family) pays a
  compile cost we don't recover. Acceptable for the heap path's
  target workload (lots of small ROs sharing little).

### Original framing (preserved for context)

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

## 9. Texture atlasing ✅ SHIPPED (different architecture than originally sketched)

**Note:** the architecture below is preserved as the original
plan; the actual ship landed at a *third* design after two
real-world architectural discoveries during integration. See
**`heap-textures-plan.md`** for the up-to-date design.

**Final shipped architecture (summary):**

- **N independent `GPUTexture` per format**, NOT a
  `texture_2d_array`. No grow-copy on adding pages.
- Bound via **N consecutive single-texture BGL bindings** per
  format (linear + srgb), addressed in WGSL via a `switch pageRef`
  ladder. Replaces the failed attempt at `binding_array<texture_2d<f32>, N>`
  (not core in WebGPU 1.0).
- **1.5×1 mip pyramid embedded per sub-rect** (Iliffe layout).
  Software mip filter in shader; hardware bilinear within each
  mip; LOD computed from screen-space derivatives.
- **Per-RO sampler state in drawHeader bits** (wrap modes + filter
  flags packed into formatBits u32). Shader applies wrap modes
  before atlas-coord transform.
- **AtlasPool with refcount + texture aval reactivity**: sprite
  swap / theme change just works (release old sub-rect + acquire
  new + update drawHeader + bump page-set version).
- **Verified end-to-end on real GPU**, including iPhone Safari.
- Tests: 25/25 mock-GPU + 2/2 real-GPU integration.

The original plan below is kept for historical reference and
for the bin-by-format-and-size discussion that's still relevant
(only one bin per format shipped in v1).

---

(Original plan from before discovery of `binding_array`'s
non-portability and `texture_2d_array`'s grow-copy cost):

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

## 10. Per-RO instancing ✅ SHIPPED

The heap path used to hard-reject ROs with `instanceAttributes` and
soft-reject `instanceCount > 1` — both went via the legacy per-RO
renderer. Today: an RO with a shared mesh + N per-instance
attribute buffers + `instanceCount = N` (e.g. a 1000-tree forest)
collapses to **one bucket, one record, one drawIndirect** in
megacall mode. Total emit per record: `indexCount * instanceCount`.

**What changed:**

- **Eligibility** (`heapEligibility.ts`): `instanceAttributes` are
  now accepted as long as every entry passes the same tight-stride
  / stride-0-broadcast rule applied to `vertexAttributes`.
  `instanceCount` may be any positive value. `firstInstance != 0`
  remains rejected.
- **DrawHeader schema** (`heapEffect.ts`): per-instance attributes
  share the existing attribute arena (no new arena introduced)
  and emit one `(refIntoArena, stride)` u32 ref per attribute,
  parallel to per-vertex attribute fields. `BucketLayout`
  records `perInstanceAttributes: ReadonlySet<string>` so the IR
  rewriter picks the per-instance index for those reads.
- **DrawTable record** grew by 4 bytes: now `(firstEmit, drawIdx,
  indexStart, indexCount, instanceCount)` = 20 bytes / 5 u32. The
  GPU prefix-sum reads `indexCount * instanceCount` per record
  instead of just `indexCount`.
- **VS prelude** splits the per-record local offset into
  `instId = local / indexCount`, `vid = indices[indexStart + local
  % indexCount]`. The header-selector identifier is `heap_drawIdx`
  so the user-facing `instance_index` builtin carries the in-RO
  instance index (= instId), not the per-RO selector.

Megacall is now the only path. The earlier non-megacall mode (one
`drawIndexed` per slot with bit-split `firstInstance` encoding) was
removed — its only advantage was avoiding the prefix-sum compute
pass, but it capped per-RO instances at `2^20 - 1` and hard-broke at
`localSlot >= 4096`. Megacall scales to billions of emits per bucket
(VRAM is the bound long before any u32 ceiling).

**Backward compat:** an RO with `instanceCount = 1` and no
per-instance attributes evaluates `instId = 0` and reads the same
header entries as before. Existing 93 rendering tests stay green.

**Why arena reuse over a new "instance arena":** per-instance
buffers don't have an access pattern WGSL distinguishes from
per-vertex — both are random-indexed reads via a `(refIntoArena,
length)` pair. The pool's aval-keyed sharing already handles the
"1000 ROs share one positions buffer" case for vertex attributes;
instance attributes flow through the same code path with no
duplication. Per-RO instance counts up to ~10⁶ fit comfortably
within today's arena growth.

**MDI migration path:** WebGPU's eventual `multi-draw-indirect`
extension replaces the binary-search-fold trick (one drawIndirect
that internally iterates `numRecords` parameter sets) with one
native MDI call. The arena layout, drawHeader schema, and
prefix-sum kernel don't change — only the encode-side switches
from "compute scan to one indirect call covering totalEmit
vertices" to "skip the scan and issue MDI". The IR's per-instance
attribute reads survive unchanged because `__heap_drawIdx` and
`instId` map directly to MDI's per-draw `drawId` builtin and the
hardware's instance-index, respectively.

**v1 limitations** worth surfacing for future tightening:

- `spec.instanceCount` is read once at addDraw time. Reactive
  updates to instance count require a re-add (CPU side) — the
  drawTable record's `instanceCount` field is wired up to be GPU-
  visible reactive, but the CPU diff path doesn't yet listen.
- The effect's IR-level attribute step-mode is inferred from the
  user's `instanceAttributes` map at adapter time; the schema
  itself is step-mode-agnostic. Two ROs with the same effect but
  one routes an attribute as per-vertex and the other as per-
  instance produce distinct buckets (different shaders).

## 11. Heap-megacall + wombat.shader composition ✅ SHIPPED

Composing VS stages — `effect(modelVS, instanceOffsetVS, clipVS, lambertFS)` —
is the wombat.shader-native way to add a small modifier (per-instance
offset, displacement, fade, …) without duplicating the trafo or lambert
pieces. The wombat.shader optimizer fuses fused-stage helpers + a
wrapper @vertex fn; it's how `DefaultSurfaces.trafo + simpleLighting`
already works in wombat.dom.

Both halves of the heap-megacall + composition incompatibility are now
fixed:

**VS side ✅ shipped.** `applyMegacallToEmittedVs` used to declare the
shared megacall values (`heap_drawIdx`, `instId`, `vid`) as `let` locals
inside the @vertex fn body. With composition, helper functions reference
those names but don't have them in scope. Initially fixed by declaring
all three as `var<private>` at module scope (Chromium/Tint accepts this);
later replaced with **parameter threading** for cross-parser portability
(Safari/WebKit rejects helper-fn reads of module-scope private vars in
some configurations). The current shape: wrapper @vertex fn declares the
three as `let` locals in the search prelude, then a post-emit text pass
(`threadMegacallParamsThroughHelpers` in `heapEffectIR.ts`) scans every
`fn _<name>(...)` body for references to the three identifiers, appends
them as `u32` parameters to the helper signature, and rewrites every
call site in the wrapper to pass the locals. No `var<private>` for these
three values in the final WGSL — portable across Chromium/Tint, Safari/
WebKit, and any conforming WGSL implementation.

**FS side ✅ shipped (wombat.shader fix).** wombat.shader's per-stage
WGSL emit now tree-shakes by entry-reachability. The new
`pruneToStage(module, stage)` pass (see
`wombat.shader/packages/shader/src/passes/pruneToStage.ts`, applied in
`compile.ts`'s `emitAll` before each per-stage emit) walks the call
graph from each stage's Entry and drops every Function ValueDef that
isn't transitively reachable. Other-stage Entries are dropped too.
Bindings, type defs, and module-scope decls are passed through (they're
declarations, not function references).

With both fixes: `effect(modelVS, instanceOffsetVS, clipVS, lambertFS)`
composes through the heap path with no FS-side WGSL `unresolved value`
errors. The heap-demo's `instancedSurface` now uses proper composition
(see `examples/heap-demo/src/effects.ts`); no hand-merged
`trafoInstancedVS` workaround.

See `~/claude/wombat-shader-composition.md` for the broader composition
mechanism + design rationale.

