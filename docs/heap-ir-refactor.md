# Heap rewriter — IR-based refactor plan

The current `heapEffect.ts` rewriter operates on raw WGSL strings via
regex substitution. This works for the happy path but is brittle:

- Adding `_h_FooRef` varyings means scanning bodies, splicing struct
  fields, injecting `out._h_FooRef = …;` after `var out: VsOut;`.
- Per-instance threading needs a parallel `_iidx` varying.
- DCE'd FS bodies (`@fragment fn fs() -> …`) break the parameter
  regex; trivial constant-FS shaders fail to compile.
- Source vs schema mismatches (V3-tight vs V4-tight for vec4) need
  runtime-stride lookups in the WGSL itself.

Each new feature compounds the fragility. The principled fix is to
operate at the **IR level**, the same surface wombat.shader already
uses for cross-stage linking, DCE, and uniform reduction.

## What the IR gives us for free

- **`ReadInput("Uniform", name)`** is a first-class IR node. Replacing
  it with an arbitrary expression (a heap-load) is a one-line `Stmt`
  transformation via `passes/substitute.ts:substInput`.
- **Cross-stage linking** already threads VS outputs to FS inputs via
  matched names. If we make per-uniform `_h_<name>Ref` a synthetic
  VS output, the FS reads it as a synthetic input — no regex.
- **DCE** drops unused decoded-uniform lets, unused varyings, unused
  attribute reads.
- **`composeStages`** sequentially composes same-stage entries. We
  emit a "heap reader" prologue stage, compose with the user's VS,
  let the existing inliner / linker do the rest.
- **`reverseMatrixOps`, `reduceUniforms`, `linkFragmentOutputs`,
  `pruneCrossStage`** all keep working on top of our injected loads.

## What we'd build in heapEffect.ts

```ts
// 1. Build the user's effect normally; get its Module + ProgramInterface.
const compiled = effect.compile({ target: "ir", fragmentOutputLayout });

// 2. Derive a layout (drawHeader stride, per-field offsets) from the
//    interface, exactly as today.
const layout = buildBucketLayout(compiled.interface, …);

// 3. Substitute every uniform read in every entry with a heap-load.
const heapLoaded = substituteUniforms(compiled.module, (u) => {
  // Per-uniform IR expression that:
  //   - Computes the alloc ref:
  //       drawIdx = ReadInput("Builtin", "instance_index")
  //       refIdent = headersU32[drawIdx * stride + offset]
  //   - Reads the typed value from heapV4f / heapF32.
  return makeHeapLoadIR(u, layout);
});

// 4. Substitute every attribute read in the VS with a heap-load
//    (vid + cyclic count from header, V3/V4 picked from header stride).
const decoded = substituteAttributes(heapLoaded, (a) => {
  return makeAttributeLoadIR(a, layout);
});

// 5. Optionally inject a small "heap prologue" via composeStages —
//    handy if we want to share decoded values across multiple VS
//    entries that touch the same input. For v1 this is unnecessary;
//    substitution alone is enough.

// 6. Add the four heap storage buffers as IR `UniformDecl`s on
//    `@group(0) @binding(0..3)`. Add texture/sampler bindings from
//    the schema's surviving entries at @binding(4+).

// 7. Emit WGSL via the existing emit pass.
```

## What needs to be public in wombat.shader

To do this without forking the package:

- `passes/substitute.ts` — already public. `substInput` does the work.
- A higher-level helper would be nicer:
  ```ts
  export function substituteUniforms(
    m: Module,
    f: (info: UniformFieldInfo) => Expr,
  ): Module;
  export function substituteAttributes(
    m: Module,
    f: (info: AttributeInfo) => Expr,
    onlyStage?: Stage,
  ): Module;
  ```
  These would walk every entry, every statement, and call `substInput`
  with the right `(scope, name)` pair, returning the rewritten Module.

- IR builders for our heap loads. Either:
  - **Inline IR construction** in heapEffect (verbose; type imports
    from wombat.shader/ir).
  - **A small "heap intrinsic" library** — wombat.shader exposes
    `irLoadStorageU32(buffer, index): Expr`, `irLoadStorageF32(...)`,
    `irConstructVec(…): Expr`, etc. We'd call these to build the
    expressions wombat.shader expects.

- A way to **add storage-buffer bindings** to a Module post-construction.
  Today these are declared in the user's source. We'd add four
  synthetic `UniformDecl`s for `heapU32 / headersU32 / heapF32 /
  heapV4f`. If `wombat.shader` doesn't currently support adding
  storage bindings to a finalized Module, that's a new public surface.

- A way to **add a synthetic VS output / FS input pair** for
  cross-stage threaded refs (`_h_FooRef: u32 @interpolate(flat)`).
  Could be a helper `addInterstageVarying(module, name, type, interp)`.

## What the heap path stops doing

After the refactor, `heapEffect.ts` has zero regex, zero string
splicing, zero `stripTextureSamplerDecls`, zero `_w_uniform.X`
sniffing. The rewriter becomes:

```
compile → substitute uniforms → substitute attributes → emit
```

Every existing pass (DCE, CSE, fold-constants, type legalisation,
cross-stage linking, fragment-output linking, matrix-op reversal)
runs on the substituted IR and produces correct, optimised WGSL.

## What this fixes immediately

- Trivial-FS shaders (constant return) compile correctly. The DCE
  trims unused VsOut fields and the emitter produces a valid
  `@fragment fn fs() -> @location(0) vec4<f32>` even with zero
  inputs.
- FS uniform reads work uniformly. The "thread-via-VsOut" pattern
  becomes a synthetic varying the linker handles like any other.
- Per-instance reads in FS work the same way (just with `iidx`
  threaded as another flat varying).
- Source/schema stride mismatches are resolved at the IR-construction
  level: `makeAttributeLoadIR(a, layout)` decides whether to read 3
  or 4 floats (or do a runtime stride pick) from the alloc header.
  No WGSL regex.

## Migration order

1. **Land `substituteUniforms` / `substituteAttributes`** as wombat.
   shader public utilities (or write thin wrappers locally over
   `substInput`).
2. **Replace `rewriteForLayout` with an IR pipeline** — substitute
   uniforms first (simpler), keep attribute substitution as the
   second step so we can stage incremental changes.
3. **Drop `stripTextureSamplerDecls`, `augmentPreludeForFsUniforms`,
   `buildVsPreamble`, `buildFsPreamble`, `injectVaryingWrites`** —
   they all become dead code.
4. **Verify pixel-equivalent output** vs the regex path for the demo
   effects (lambert, wave, flat, surface).
5. **Drop the regex rewriter entirely.**

## Out of scope for this refactor

- Native-buffer ingest (still legacy fallback via `isHeapEligible`).
- Multi-binding texture API (still single pair v1).
- Auto-instancing (separate later work).
- Trace-based effect merging (separate later work).
