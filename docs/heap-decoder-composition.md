# Heap-decoder VS — composition-based heap rewrite

Replaces the current `Effect.substitute` + `applyFamilyMemberShape` heap path
with a design that uses wombat.shader's own composition primitives. The
heap rewrite becomes a *prepended* vertex stage instead of a mid-pipeline
IR-string surgery, which eliminates the desync bugs the current path has
when the user effect declares inputs the substitute pass elides but
`extractFusedEntry` later re-references in the wrapper init.

## Motivation

Current heap path:

```
user.Effect
  → substituteInputsInStage(...)         // rewrites ReadInput("Input"/"Uniform") nodes
  → effect.compile({ ... })              // runs extractFusedEntry → wrapper VS + helpers
  → applyFamilyMemberShape(emittedWgsl)  // string-mangles @vertex signature
```

The substitute pass walks the user's *body* and rewrites every
`ReadInput("Input", X)` it sees. But `extractFusedEntry` later inserts a
*new* wrapper-init body that writes `state.X = ReadInput("Input", X)` for
every declared `EntryParameter` — these reads weren't visible at substitute
time, so they survive into the emitted WGSL as `in.X` references against
a `@vertex` parameter list that `applyFamilyMemberShape` has already stripped.

Result: WGSL with `s.X = in.X` where both `s` and `in` are unresolved values.
Heap-demo doesn't hit it because its hand-built effects declare only inputs
the body actually reads; `DefaultSurfaces.trafo` declares
`DiffuseColorUTangents`/`DiffuseColorVTangents`/`DiffuseColorCoordinates`
that `simpleLighting` never consumes — exactly the case the wrapper-init
generates phantom reads for.

## Approach: compose, don't rewrite

Synthesise a fresh **heap-decoder vertex stage** that:

1. Takes only megacall builtins (`heap_drawIdx`, `instId`, `vid`) as inputs.
2. Loads every attribute and uniform the downstream effect reads, from the
   heap arenas / drawHeader.
3. Surfaces them on the inter-stage carrier under the names the downstream
   effect already uses.

Then build the final effect as:

```ts
const finalEffect = effect(heapDecoderVs, ...userEffect.stages);
```

wombat.shader's existing stage-fusion + DCE handle the rest:

- Inter-stage carrier inputs naturally match the downstream `EntryParameter`s.
- DCE drops decoder outputs that no downstream stage reads (so a uniform
  the user declares-but-doesn't-read costs nothing).
- `extractFusedEntry` merges the decoder with the user's VS into a single
  wrapper + helpers without seeing any heap-specific structure.
- `applyFamilyMemberShape` goes away entirely — the family wrapper just
  calls the fused VS like any other helper.

No mid-pipeline rewrites, no string-mangling, one composition.

## Two prep transforms in wombat.shader

### 1. Uniform → Input rename pass

In the user effect's module, rewrite every `ReadInput("Uniform", X)` to
`ReadInput("Input", X)` and remove `Uniform` decls for the renamed names.
After this transform, the only distinction left is whether the *decoder*
emits a value or a ref for that name — uniform-vs-attribute is no longer
visible in the user's IR.

Lives in `wombat.shader/passes/uniformsToInputs.ts`. ~30 lines.

```ts
export function uniformsToInputs(m: Module, names: ReadonlySet<string>): Module;
```

Names parameter so the caller (heap synthesis) can scope the rename to
exactly the per-RO uniforms; ambient uniforms the runtime owns directly
(if any) stay as Uniform.

### 2. ReadHeap intrinsic

New IR node, parallel to `ReadInput`:

```ts
interface ReadHeap {
  kind: "ReadHeap";
  ref: Expr;       // u32 expression — the heap offset / arena index
  arena: HeapArena; // "u32" | "f32" | "v4f" | "headers"
  type: Type;      // result type
  span?: Span;
}
```

WGSL emit: a small helper-fn library keyed by `(arena, type)` — same code
as the current `loadUniformByRef` helpers, just promoted from text
templates to IR-emit. Carrier slot is a `u32` (4 bytes); the load happens
at the use site.

DCE-friendly: if a `ReadHeap(ref, ...)` is dead, both it and the
upstream `ReadInput("Input", "XRef")` that produced `ref` get dropped.

## Per-name carrier policy: value vs ref

| Type | Slots-as-value | Slots-as-ref | Carrier kind |
|------|----------------|--------------|--------------|
| `u32`/`f32`/`i32`/`bool` | 1 | 1 | **value** |
| `vec2`/`vec3`/`vec4<T>` | 1 | 1 | **value** |
| `mat2x2<f32>` | 1 | 1 | **value** |
| `mat3x3<f32>` | 3 | 1 | **ref** |
| `mat4x4<f32>` | 4 | 1 | **ref** |
| `array<T,N>` | many | 1 | **ref** |

Rule: anything that fits in one WGSL varying slot (≤ 16 bytes) goes by
value — by-ref would use the same slot count and add an FS-side load.
Anything bigger goes by ref.

Implementation:

```ts
function carrierKind(t: Type): "value" | "ref" {
  switch (t.kind) {
    case "Float": case "Int": case "Bool": return "value";
    case "Vector": return "value"; // vec2/3/4 all fit
    case "Matrix": return (t.rows * t.cols * 4) <= 16 ? "value" : "ref";
    case "Array":  return "ref";
    default:       return "ref";
  }
}
```

Refinement v2 — **ref packing**: 4 `u32` refs fit into one `vec4<u32>`
varying slot. A later optimisation can bin refs by group of 4 and emit
a single packed varying per group. v1 ships with one slot per ref;
revisit if real workloads push slot count.

## Decoder synthesis

Input: a post-rename user effect + the bucket layout (drawHeader field
list + arena bindings).

Output: an IR `EntryDef` for the decoder VS.

Algorithm:

1. Collect `(name, type)` pairs for every Input read by any stage in
   `userEffect`. (Post-rename, this includes former uniforms.)
2. Classify each pair via `carrierKind(type)`.
3. For each `value` pair: emit a `WriteOutput(name, heapLoadExpr(name, type))`.
4. For each `ref` pair: emit a `WriteOutput(nameRef, drawHeader.XRef)`
   where `nameRef = mangleRefSuffix(name)`. Rewrite every downstream
   `ReadInput("Input", name)` of type T to `ReadHeap(ReadInput("Input", nameRef), arenaFor(T), T)`.

`heapLoadExpr(name, type)` is the same expression the current substitute
mapping produces. It hits per-vertex arena, per-instance arena, or
drawHeader depending on whether the name is a vertex attribute, an
instance attribute, or a uniform — same classification the current path
already makes via `perInstanceAttributes` / `perInstanceUniforms` sets.

## Family-merge becomes structural

A family is N decoders + N user VSes + N user FSes, all composed. The
existing extractFusedEntry merges them into one wrapper per stage exactly
as it does today for non-heap effects. The family-VS dispatch's
`switch (layoutId)` calls into per-layout helpers that are *just*
well-formed function-form WGSL — no `@vertex` mangling required.

`heapShaderFamily.ts` shrinks to:
- Schema collection (which effects fuse, drawHeader union, per-effect
  layoutId assignment) — unchanged.
- WGSL wrapper synthesis — much simpler: the per-effect helpers are
  vanilla functions extracted by wombat.shader; the wrapper just
  dispatches.
- `applyFamilyMemberShape` deleted.

## What goes away

- `Effect.substitute({ vertex: { inputs, uniforms } })` — heap path
  stops using it. Other callers (auto-instancing) keep using it.
- `applyFamilyMemberShape` — deleted.
- The custom string post-processing in `applyMegacallToEmittedVs` — the
  megacall search prelude moves into the decoder body as plain IR.
- `threadMegacallParamsThroughHelpers` — same; helpers that need
  `heap_drawIdx`/`instId`/`vid` get them as decoder outputs through the
  carrier, not as threaded params.

## What stays

- Heap layout (arenas, drawHeader, megacall, drawTable, prefix-sum) —
  GPU-side data structures unchanged.
- DrawHeader field collection per effect — unchanged.
- Per-RO uniform/attribute upload via the UniformPool — unchanged.
- Atlas binding-array path for textures — unchanged (textures aren't
  routed through the carrier; they remain bind-group resources).

## Implementation steps

Each lands independently; the heap path can keep working during the
transition.

1. **wombat.shader: `uniformsToInputs` pass.** New file in
   `packages/shader/src/passes/`, exported through `passes/index.ts`.
   ~30 lines + unit test.
2. **wombat.shader: `ReadHeap` IR node + WGSL emit.** New `Expr` variant
   + `emitExpr` case + `loadUniformByRef`-style helper-fn library
   keyed by arena/type. ~80 lines + emit test.
3. **wombat.rendering: decoder synthesis.** New file
   `runtime/heapDecoder.ts`: given a post-rename user Module + bucket
   layout, returns the decoder `EntryDef`. ~150 lines.
4. **wombat.rendering: integrate.** In `compileHeapEffect`, replace
   `Effect.substitute(...)` + `applyFamilyMemberShape` chain with
   `uniformsToInputs` → `decoder synth` → `effect(decoder, ...stages)`.
5. **wombat.rendering: simplify `heapShaderFamily.ts`.** The
   per-effect helpers are now vanilla; remove the family-member-shape
   special-casing, reduce family-VS wrapper to a straight `switch` over
   `layoutId`.
6. **Delete dead code.** `applyFamilyMemberShape`,
   `applyMegacallToEmittedVs`, `threadMegacallParamsThroughHelpers` —
   if/when no other consumer.

Step 1+2 are pure wombat.shader and can ship as 0.5.4 / 0.6.0. Steps 3-6
land as a wombat.rendering 0.10.0.

## Testing

- Bring `DefaultSurfaces.trafo + simpleLighting` back into a real-GPU
  test in `wombat.rendering/tests-browser/` — it should render. This is
  the failing case from heap-demo-sg.
- All existing heap-demo paths must keep rendering (hand-built effects
  cover surface/instanced/tinted/pulsing/wobbling/textured paths).
- Family-merge regression test: same scene with `?merge=1` and
  `?merge=0` should produce the same pixels.
- DCE coverage: an effect that declares but doesn't read
  `DiffuseColorUTangents` must not emit a heap-load for it (decoder
  output is dead → dropped). Check by inspecting emitted WGSL.

## Open questions

- **Picking shaders.** The pickId chain currently injects a separate
  fragment + uniform read. After the rename pass, picking's
  `ReadInput("Uniform", "PickId")` also gets routed through the decoder.
  Should work identically; verify in tests.
- **Cross-VS-stage uniform reads.** If both a user VS and FS read the
  same uniform, the decoder writes it once (value carrier) and both
  stages observe it. Same code path; no special case needed. But verify
  the inter-stage carrier survives across multiple VS operands in the
  fused entry.
- **`renderTo`'s inner scene.** `renderTo` builds a fresh task. Same
  flow as the outer scene; nothing special required.
