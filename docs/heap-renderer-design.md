# Heap renderer design — notes

Personal scratchpad on the wombat.rendering heap-scene
runtime as it converges to a render-object pipeline. Captures the
"why" so it survives a context flip.

## Where this lands

A render-object renderer where:

- The user emits `(effect, inputs, indices, textures?)` "things". Some
  call them render-objects, render-tasks, draws — same idea.
- The runtime introspects each effect (linked + DCE'd via wombat.shader's
  `compiled.interface`) and figures out what it needs.
- Sharing emerges from aval / object identity. Ten thousand things
  pointing at the same `cval<Trafo3d>` for `ViewProjTrafo` collapse to
  one heap allocation. Same for `Float32Array` positions. No "frequency"
  declarations on the user side.

Nothing is hardcoded by name in the runtime. Adding a new uniform or
attribute is a spec entry on the user side; adding a new WGSL type is
one entry in `packerForWgslType`.

## Aval identity is the right granularity

Two different `cval`s carrying the same value are still two cvals.
That's correct: identity means "equal for all future." Content
hashing for fusion is a separate problem (auto-instancing /
auto-effect-merging) layered on top.

## Inline reads, no helper library

The per-bucket WGSL prelude is just:

```wgsl
@group(0) @binding(0) var<storage, read> heapU32:    array<u32>;
@group(0) @binding(1) var<storage, read> headersU32: array<u32>;
@group(0) @binding(2) var<storage, read> heapF32:    array<f32>;
@group(0) @binding(3) var<storage, read> heapV4f:    array<vec4<f32>>;

struct VsOut { /* per-effect fields from schema.varyings */ };
```

The rewriter substitutes every `_w_uniform.X` and `<param>.<attr>` read
inline (with a `let _h_X = ...` preamble at the top of each entry
function). No `loadMat4` / `loadVec3` / `loadVec3Attr` helper fns to drift.

## Memory model

One global `GPUBuffer` (the "arena") backs **all** non-header data:
uniform values, vertex attributes, anything else avals carry. Multiple
typed views (`array<u32>`, `array<f32>`, `array<vec4<f32>>`) of the
same physical buffer — emscripten-style `HEAPF32`/`HEAPU32`/etc.

Per-bucket DrawHeader buffer (`array<u32>`): one u32 ref per
schema-declared uniform/attribute. Stride and field offsets baked
into the per-bucket WGSL via `let` preamble; runtime knows them
from the layout.

Indices live in their own `INDEX`-usage `GPUBuffer` (WebGPU forces
this).

## Where instancing / multidraw fit

User-side optimizations are **not** redundant. They're pre-aggregated
information; the runtime should pass them through, not throw away.

Three submission shapes the runtime supports under the same
machinery:

### (1) "N independent things"

```
N entries in the cset
→ N DrawHeader slots
→ N drawIndexed(_, 1, _, _, slot) calls
```

The pool collapses shared inputs (one Positions allocation, one
ViewProjTrafo allocation, etc.). CPU encode: O(N) draw calls.

This is the **default fallback** — works without the user knowing
what's shared. Right for SG runtime output where structure is
emergent, not declared.

### (2) "I know this is one mesh with 19K transforms" — IMPLEMENTED

```
1 RO with `instances: { count: 19000, values: { ModelTrafo: Trafo3d[] } }`
+ ModelTrafo packed as 19000 × 64 B in the arena
→ 1 DrawHeader slot (single-slot bucket)
→ 1 drawIndexed(indexCount, 19000, firstIndex, 0, 0)
```

User pre-aggregates → runtime stores the per-instance values as a
packed array allocation in the arena (aval-keyed, refcounted, same
pool as everything else). Shader reads `_h_ModelTrafo` via
`instanceLoadExpr(_, _, "iidx")` where `iidx = instance_index`.

CPU encode: O(1) per instanced RO. Same arena, same bind-group
*layout*; different bucket because the WGSL is regenerated with
per-instance load expressions for the names the user marked.

**Bucketing decision:** each instanced RO gets its OWN bucket
(single slot, slot=0 baked into WGSL as `drawIdx = 0u`). Multiple
instanced ROs of the same effect cost one extra pipeline + bind
group each — cheap. Sharing across instanced ROs is shape-(3) work,
not shape-(2).

**`firstInstance` convention:**
- Non-instanced buckets: `firstInstance = slot`, `instanceCount = 1`.
  Shader reads `drawIdx = instance_index`. (the original trick)
- Instanced buckets: `firstInstance = 0`, `instanceCount = N`.
  Shader reads `iidx = instance_index`, with `drawIdx = 0u` baked.

This avoids any `firstInstance`-bit-packing scheme — cleaner WGSL
on both paths.

### (3) Multidraw indirect

```
Many draws, same effect, same bucket
→ build indirect buffer (CPU or compute)
→ 1 drawIndexedIndirect per bucket
```

Buckets already share pipeline + bind group; only the per-draw
slot index varies. Pack the bucket's `drawSlots[]` into an indirect
buffer and emit one drawIndexedIndirect per bucket. CPU encode
becomes O(buckets), not O(draws).

This is an **opportunistic runtime optimization** — invisible to
the user, kicks in when N is large.

## Auto-instancing

Detecting that two of the user's "independent things" actually
share geometry (and could collapse to one instanced draw) is the
auto-instancing problem. It's a CONTENT-HASH-based fuse, distinct
from the identity-based sharing the heap path already does.

Same effect.id + same indices.id + same per-vertex attribute aval.ids
+ different per-instance uniform aval.ids = candidate for
auto-instancing.

This is a separate pass over the bucket's draws, applied at frame
time or on add. Not in the heap path's MVP; layered on top later.

## Trace-based effect merging

Different effects with the same input/output schema — auto-merge
shaders so the runtime issues one pipeline instead of N. Requires
varying-layout reconciliation (V0..VN: vec4<f32> generic slots, each
effect picks slots, merger renames its outputs).

User's preference: trace-based, not user-declared. Same theme as
auto-instancing — the runtime detects fusable structure.

Bucketing key today: `effect.id|texture-id`. Auto-merge would create
a fused effect from N input effects, replace each draw's effect with
the fused one, re-bucket. Trace-based means the merge cost is paid
on the per-frame trace, not per draw.

## What the runtime preserves

The heap renderer is non-destructive of user-supplied structure:

- Pre-aggregated instances → instanced draw call (don't re-decompose
  to N draws).
- Pre-aggregated multidraw → indirect call (don't re-decompose).
- Independent draws → individual draws + arena-shared inputs (don't
  pretend they're instanced).

The user picks the shape that matches what they know. The runtime
makes each cheap.

## Open follow-ups

- **Spec collapse for indices**: indices currently a separate field;
  could move into `inputs` with a well-known role tag. Minor cleanup.
- **Schema-driven texture/sampler bindings**: still wedged via
  hardcoded binding 4/5; should come from `iface.samplers` /
  `iface.textures` post-DCE.
- **Multi-chunk seal at 32 MB**: the arena's pow2 grow handles up
  to `maxBufferSize` (≥256 MB on most adapters); above that, seal
  and chunk.
- **`drawIndexedIndirect` path**: shape (3) above. Compute-built
  indirect buffer + per-bucket dispatch.
- **Auto-instancing pass**: content-hash fuse over the bucket's
  draws.
- **Trace-based effect merging**: user's longer-term goal.

## Shape of the WGSL it emits today

For a draw using lambert (modelTrafo + color + viewProj + light) with
positions+normals attribs, the rewritten VS is roughly:

```wgsl
@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) drawIdx: u32) -> VsOut {
  let _h_ModelTrafoRef = headersU32[drawIdx * 6u + 0u];
  let _h_ModelTrafo = mat4x4<f32>(/* 4 vec4 reads from heapV4f */);
  let _h_ColorRef = headersU32[drawIdx * 6u + 1u];
  let _h_Color = heapV4f[(_h_ColorRef + 16u) / 16u];
  let _h_ViewProjTrafoRef = headersU32[drawIdx * 6u + 2u];
  let _h_ViewProjTrafo = mat4x4<f32>(/* 4 vec4 reads */);
  let _h_LightLocationRef = headersU32[drawIdx * 6u + 3u];
  let _h_LightLocation = vec3<f32>(/* 3 f32 reads */);
  let _h_PositionsRef = headersU32[drawIdx * 6u + 4u];
  let _h_Positions = vec4<f32>(/* loadVec3 + assemble */, 1.0);
  let _h_NormalsRef = headersU32[drawIdx * 6u + 5u];
  let _h_Normals = vec3<f32>(/* loadVec3 */);

  /* original DSL body, with _w_uniform.X → _h_X and v.X → _h_X */
}
```

The `let` preamble is generated from the bucket's layout. Effects
that don't reference a particular schema field don't get a let for it.
