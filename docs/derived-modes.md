# Derived modes

A design sketch for letting **pipeline state** — cull mode, blend mode,
depth compare, depth bias, etc. — be a function of uniform values (and
the SG-declared default), evaluated on the GPU each frame and used to
partition the draw stream at encode time.

This is the discrete-output sibling of §7 (derived uniforms). The §7
kernel computes continuous per-RO values (mat4s, vec3s) into drawHeader
slots. Derived modes reuse the same kernel shape, but the per-RO output
is a **structural hash of a pipelineState descriptor**, consumed by a
counting-sort + segmented-encode pass rather than by the shader.

## The user-facing shape

```ts
// SG-declared default flows in as the `declared` parameter.
// Rule can pass it through, refine it, or override it.
derivedMode("cull", (u, declared) =>
  u.WindingFlipped ? flipCull(declared) : declared
);

// Pure data-driven override (declared ignored).
derivedMode("blend", (u, _) => u.Premultiplied ? "premul-over" : "straight-over");

// Continuous-domain field — depthBias is i32 baked into the pipeline,
// but in practice scenes realize a small set of values.
derivedMode("depthBias", (u, _) =>
  ({ depthBias: u.Layer * -8, depthBiasSlopeScale: 0, depthBiasClamp: 0 })
);
```

A binding for pipeline state on the SG is **either a static value** (a
literal `"back"` / `"premul-over"` / a `DepthBiasTriplet`) **or a rule**
— same value-or-rule duality as derived uniforms.

Multiple rules compose: each names the axis it overrides. The rest of
the pipeline state comes from the SG defaults. The runtime fuses all
axes into one effective pipelineState descriptor per RO.

## Why not enumerate axes?

The first instinct is to treat each mode axis as enumerable and emit one
drawIndirect per combo. That works for the discrete axes (cull ∈ {none,
front, back}, blend has ~10 useful presets, depth compare has 8 values,
front-face has 2). It collapses for the **continuous baked fields**:

- `depthBias: GPUDepthBias` — i32, 2³² possible values
- `depthBiasSlopeScale: f32`, `depthBiasClamp: f32` — continuous

(Note: WebGPU pipeline state has very few dynamic fields —
`setBlendConstant(color)` and `setStencilReference(ref)` are the lot.
Everything else, including depth bias, is baked into the pipeline.)

You can't enumerate the depth-bias triplet. But you don't need to —
**realized** depth-bias values per scene are usually 2–5 ("world", "decal",
"gizmo overlay", "shadow"), even though the possible set is unbounded.

So the rule's output is a **whole pipelineState struct**, and the runtime
caches GPU pipelines by **structural hash** of that struct. The cache is
keyed on what's *observed*, not what's *possible*.

## Runtime flow

1. **§7-shaped compute pass** evaluates every RO's mode rules. Output
   per RO is a packed `pipelineStateHash: u64` (or a small struct
   spilling into a side buffer for the depth-bias triplet, hashed at
   write time). The SG-declared defaults are inputs alongside uniforms.
2. **Counting-sort by hash** in a second compute pass. Output: K
   contiguous segments in the drawTable, one per distinct hash observed
   this frame, plus a `(hash, firstRecord, count)` table.
3. **Encode** walks the segment table:
   - Look up `hash → GPURenderPipeline` in the CPU-side pipeline cache.
   - Hit → `setPipeline(pl); drawIndirect(...)` for this segment.
   - Miss → fire `createRenderPipelineAsync(...)` keyed by the descriptor;
     for *this* frame, fall back to the SG-declared static pipeline for
     those records (or skip them, if the rule disabled draw). Next frame
     the cache is warm.

K = distinct *realized* combos per frame, bounded in practice by the
rule's effective output domain.

## Async + pre-warm

`createRenderPipelineAsync` is the right primitive — pipeline compiles
run on background threads and don't block encode. The user API exposes
a pre-warm hook:

```ts
// At scene build, declare expected combos so the cache is warm by
// first frame.
preWarmPipelineCache(scene, [
  { cull: "back",  blend: "opaque",     depthBias: 0 },
  { cull: "back",  blend: "premul-over", depthBias: 0 },
  { cull: "none",  blend: "opaque",     depthBias: -8 },  // decal layer
  // ...
]);
```

Unexpected combos take one frame of lag to materialize. Acceptable for
debug toggles and rare transitions; worth surfacing in the API docs so
hot paths get declared explicitly.

## Composition with other heap pieces

- **§7 derived uniforms**: same compute machinery. Mode rules can
  consume uniforms but the partition pass is independent of the
  drawHeader writes — both can run in the same dispatch if profile
  shows it helps, or stay separate for simpler bookkeeping.
- **§6 family merge**: **dead** for now (per the 2026-05-09
  measurement). Derived modes don't depend on it. The bucket key
  drops pipelineState entirely and keeps `(effect [, texture set])`;
  pipelineState becomes a per-RO derived attribute, not a bucket key.
- **§10 instancing**: orthogonal — each record still encodes its
  `instanceCount`; the partition pass sorts records, not instances.
- **Atlasing (§9)**: orthogonal. Texture-set bucketing is unchanged.

## Open questions

- **Hashing scheme.** A 64-bit FNV of the normalized descriptor is
  enough for the realized set sizes we expect (K < 100); CPU side
  keeps the canonical descriptor in a `Map<bigint, descriptor>` so
  hash collisions can't yield wrong pipelines (compare descriptor on
  hit). Acceptable cost.
- **Where the SG-declared default lives.** Options: a dedicated
  per-RO "declared mode" slot in the drawHeader (read like any
  uniform), or a per-effect constant baked into the rule's compiled
  kernel. The former is more flexible (inheritance, per-RO overrides
  flow naturally); cost is small.
- **Disable/skip semantics.** A rule can return `"skip"` to drop an
  RO from this frame (useful for uniform-driven culling). Encode
  treats a skip-segment as a no-op. Simpler than a separate culling
  mechanism.
- **Stencil reference / blend constant.** These are *dynamic* state
  in WebGPU — they should NOT contribute to the pipeline cache key.
  If they're uniform-driven, set them per-segment via
  `setStencilReference` / `setBlendConstant` *between* drawIndirects.
  Probably worth a separate per-RO slot the encoder reads at
  segment-emit time.
- **Interaction with chain-flatten (§7 v2).** Mode rules don't
  recurse into other mode rules — the output is a discrete
  descriptor, not a value other rules can consume. Simpler than §7's
  rule-chain story.

## Costs

- **GPU**: one extra compute dispatch per frame (mode evaluation +
  counting-sort). Counting-sort over K buckets is two passes
  (histogram + scatter); cheap at any realistic record count.
- **CPU**: encode goes from one `drawIndirect` per bucket to K. K is
  small. Pipeline cache lookup is a `Map.get` per segment.
- **VRAM**: one drawIndirect args buffer slot per (bucket × K), grown
  on demand. Negligible.
- **Compile**: one `createRenderPipelineAsync` per first-observed
  combo. Pre-warm hides this for declared combos; lazy-materialize
  with one-frame lag for the rest.

## Why this is the right shape

Game engines hard-code mode = state-block lookup tables. FRP libraries
can't reach pipeline state at all. Scientific viz toolkits expose modes
as static configuration. The wombat angle — *"pipeline state is a
function of uniforms, evaluated on the GPU, partitioning the draw stream
at encode time"* — falls out naturally once the §7 kernel exists, and
turns "modes" from a bucket-key axis (which fragments encoding) into a
data-driven axis (which collapses encoding to K small drawIndirects
sharing one bucket, one arena, one drawHeader pool).

Together with §7, this completes the "everything is a function of
uniforms" framing: continuous quantities (mat4s, vec3s) go to drawHeader
slots; discrete pipeline state goes to segmentation. One compute pass
upstream of encode handles both.
