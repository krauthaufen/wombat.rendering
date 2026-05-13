# Derived modes

A design for letting **pipeline state** — cull mode, blend mode, depth
compare, topology, etc. — be a function of uniform values (and the
SG-declared default), with the full set of possible pipeline
configurations **enumerated and pre-warmed at scene build** so the
runtime never sees a cache miss.

Companion to §7 (derived uniforms). §7 evaluates continuous per-RO
values on the GPU into drawHeader slots. Derived modes is the
**discrete-output sibling**: per-RO output is a small mode key, used
to bucket records into pre-bound pipeline slots at encode.

## The user-facing shape

```ts
// SG-declared default flows in as the `declared` parameter.
// Rule can pass it through, refine it, or override it.
derivedMode("cull", (u, declared) =>
  sign(u.ModelTrafo.upperLeft3x3.determinant()) < 0
    ? flipCull(declared)
    : declared
);

derivedMode("blend", (u, _) =>
  u.Premultiplied ? "premul-over" : "straight-over"
);

// Pure override — `declared` ignored.
derivedMode("cull", (u, _) => u.Side === 0 ? "back" : "front");
```

A binding on a pipeline-state axis is **either a static value** (a
literal `"back"`, a blend preset, a depth-compare value) **or a rule** —
mirroring derived uniforms' value-or-rule duality. The SG default flows
into the rule as `declared`; identity rules (no derived rule bound)
pass the declared value through unchanged.

## What can and cannot be derived

**Derivable axes** — fields with small, enumerable value domains:

- `topology` (5 values)
- `stripIndexFormat` (2)
- `frontFace` (2)
- `cullMode` (3)
- `unclippedDepth` (2)
- `depthWriteEnabled` (2)
- `depthCompare` (8)
- color blend (as a preset index — opaque, alpha-over, premul-over,
  additive, multiply, …; typically 5–10 per scene)
- color writeMask (16)
- stencil ops / compare (each 8) when stencil is enabled

**NOT derivable per-RO** — the continuous-domain fields are deliberately
carved out:

- `depthBias`, `depthBiasSlopeScale`, `depthBiasClamp` — continuous,
  baked into the pipeline. Use static per-effect or per-SG-scope
  bindings instead ("decal effect uses bias=-8"). This is how engines
  already author depth bias; making it per-RO uniform-driven was never
  the real need.
- `blendConstant` — continuous, but **dynamic** (set via
  `setBlendConstant` per render pass). Set it at pass level, not per
  RO. If you need it varying mid-pass, do so via SG scope boundaries.
- `stencilReference` — also dynamic; same story.
- Render-pass-constant fields (color/depth attachment formats, sample
  count, multisample mask) — fixed by the render target, not per RO.

The carve-out is load-bearing. Without continuous fields in play, the
realized pipeline configuration fits in a u32 modeKey (~24 bits for a
typical single-attachment scene), and the full *possible* set of
configurations is finite and enumerable.

## Static enumeration at scene build

The compile pipeline knows the full pipeline domain before the first
frame, via two cooperating analyses:

**1. SG-side: declared-mode collection.**

The existing PipelineState derivation during SG traversal gathers the
set of declared values per axis, per effect:

```
declaredModes : Map<EffectId, Map<Axis, Set<ModeValue>>>
```

A single `<Sg PipelineState={{ cull: "back" }}>` contributes
`{ cull: {"back"} }` to whatever effects live under it. Branching SGs
naturally produce sets > 1.

**2. Rule-side: conservative output-domain analysis.**

Each `derivedMode(axis, rule)` IR is statically analyzed. For each
declared input value, the analyzer computes a conservative
over-approximation of the rule's output set:

- Identity rule → `outputs = declared`
- Constant rule → `{ literal }`
- Conditional with literal returns (`u.X ? "back" : declared`) →
  `{"back"} ∪ declared`
- Switch over a uniform with known finite domain → `⋃ arms`
- Anything not provably finite → analyzer reports "open"; **compile
  fails with a clear message** unless the user supplies an explicit
  `domain: ModeValue[]` hint on the marker.

**3. Cross-product.**

For each effect, take the cartesian product of all axis output domains.
The union across effects is the **pipeline domain** — the complete set
of `GPURenderPipelineDescriptor`s the runtime can ever realize.

Realistic scene sizes:
- cull: ≤ 3, depth (write × compare): ≤ 4, blend preset: ≤ 5,
  writeMask: ≤ 2.
- Per-effect cross-product: 5–20 typical, ≤ ~100 for the most
  feature-rich scene.

A configurable cap (default ~256) fails build with an authoring
diagnostic if the domain explodes.

## Pre-warm and `scene.ready()`

For each descriptor in the pipeline domain:

```ts
device.createRenderPipelineAsync(descriptor)
```

`scene.ready()` returns a Promise resolving when all are linked. The
browser-internal WGSL compile cache makes repeat sessions near-instant;
first session pays the compile cost, hidden behind loading.

After `await scene.ready()`, every pipeline that *can* be requested at
runtime exists in the cache. **Frame 0 is correct.**

## Runtime: fixed slots, no readback

Pipeline domain enumeration assigns each `(descriptor)` a stable **slot
index** in a per-bucket indirect-args buffer. The slot table also lives
on the GPU as a `modeKey → slot` lookup buffer.

**Per-frame, per bucket:**

```
1. mode-eval kernel:
     per-RO modeKey from rules + uniform values + declared
2. partition kernel:
     each thread looks up modeKey → slot index
     atomic-add to (indexCount, instanceCount) of slot's indirect args
     record goes into the slot's segment of the sorted drawTable
3. encode:
     for each slot in scene.pipelineSlots:
       setPipeline(slot.pipeline)
       setBindGroup(0, bucket.bindGroup)
       drawIndirect(bucket.indirectArgs, slot.indirectOffset)
```

**Empty slots draw zero.** If no records realize a given mode this
frame, that slot's `(indexCount, instanceCount)` is zero and the GPU
skips it. Cost per empty slot: one setPipeline + one drawIndirect
reading 0 — negligible at P_total ≤ ~50.

**No GPU→CPU readback.** No `mapAsync`. No "any new state?" digest.
The runtime hot path never communicates with the CPU about modes.

**Dirty-gating** (optional optimization for steady state): if no
mode-rule input avals were marked and no records added/removed, skip
the mode-eval + partition kernels entirely, reuse last frame's slot
contents. Static scenes pay zero per-frame.

## Runtime mutations

When an RO is added at runtime (or a rule changes):

1. Run the analyzer incrementally on just the new subtree → new
   declared values, new output domains.
2. Diff against the existing pipeline domain → set of new descriptors.
3. For each: `device.createRenderPipeline(descriptor)` **synchronously**.
   JS call returns instantly. Assign new slot indices, resize the
   bucket's indirect-args buffer + slot lookup table.
4. Encode the RO into its bucket immediately.

**The frame this happens on is long** — the GPU stalls on first use of
each newly-created pipeline while the driver compiles. Tens of ms cold;
sub-ms with warm browser WGSL cache. Subsequent frames are normal.
Pixels are correct that frame; just the frame is slow.

This is the **only** stall the design ever takes, and it only fires
when truly-unseen state is introduced after scene build. Apps that want
to avoid it use `preWarmPipelineCache(scene, descriptors[])` during
loading.

The build-time vite plugin warns on the obvious anti-pattern: a
`derivedMode(...)` call inside a render callback / frame-loop closure
constructs a new rule per frame and would defeat enumeration. Catch it
at compile time.

## Why this works on WebGPU specifically

- WebGPU has no pipeline serialization (`GPUPipelineCache` is a v2
  proposal); the browser's internal WGSL compile cache makes repeat
  sessions fast but the user code can't query it. Static enumeration
  + async pre-warm is the WebGPU-1.0-shaped solution.
- WebGPU's dynamic-state set is tiny (`setBlendConstant`,
  `setStencilReference`, `setViewport`, `setScissorRect`). Everything
  else bakes into the pipeline. Hence the carve-out of continuous
  baked fields from the derivable set: enumeration would fail
  otherwise.
- `createRenderPipeline` (sync) returns a JS pipeline object instantly;
  driver compile happens in the background and first submission
  stalls the GPU queue if needed. That's what makes the
  runtime-mutation path's stall behave as a frame-time spike rather
  than a CPU-side wait.

## Why this beats today's bucketing

Today the bucket key includes pipelineState, hashed by aval
**identity**, not value (`psIdOf` in `heapScene.ts:2088`). Realistic
consequences:

- 20k ROs each constructed with `cval("back")` for cullMode → 20k
  distinct cval identities → **20k buckets**, each with its own
  arena, drawTable, pipeline. All bitwise-identical state.
- Reactively flipping `cullCval.value = "front"` does NOT rebucket
  (identity unchanged) → the RO renders **with the old cullMode,
  silently wrong**.
- The only correct way to change cullMode today is to construct a
  fresh PipelineState object (new aval identities → new bucket key),
  which churns the bucket bookkeeping.

Derived modes drops pipelineState from the bucket key entirely.
Bucket key becomes `(effect, textureSet)`. The realized cullMode
**value** determines pipeline at encode via the modeKey → slot
mapping. 20k cvals all valued "back" collapse to one segment, one
pipeline, one bucket. Flipping a value flows through the mode-eval
kernel and lands in a different segment next frame.

## Composition with other heap pieces

- **§7 derived uniforms**: shares the rule-IR machinery (DerivedExpr,
  proxy leaves, build-time leafTypes hint via UniformScope). The
  mode-eval kernel is essentially a §7-shaped pass with discrete
  output types.
- **§6 family-merge**: dead (per the 2026-05-09 measurement). Derived
  modes don't depend on it. PipelineState dropping out of the bucket
  key is independent.
- **§10 instancing**: orthogonal — partition is over records, each
  carrying its own instanceCount.
- **§9 atlasing**: orthogonal — textureSet stays in the bucket key.

## Open / deferred

- **Stencil**, when used, contributes per-face compare + ops + masks to
  the modeKey. Mostly off in typical scenes; design-supports-it but
  the analyzer's stencil-on case may want tightening if a scene
  actually uses uniform-driven stencil state.
- **GPU-side rule chaining**: a mode rule consuming a §7-derived
  uniform's value (rather than a raw uniform). Doable — same kernel
  schedules both — but adds analyzer complexity. Punt to v2.
- **Persistent pipeline manifest across sessions**: still useful even
  with full enumeration, because the browser's WGSL compile cache
  doesn't cover the link step. Optional add-on; logs realized
  descriptors to IndexedDB, replays on next session boot during
  pre-warm.
