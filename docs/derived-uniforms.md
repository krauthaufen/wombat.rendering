# Derived uniforms (§7)

A uniform binding on a RenderObject is **either a value** (an `aval` / constant)
**or a rule** (a `DerivedRule` produced by `derivedUniform(...)`). A rule is a
pure function of *other* uniforms; the heap renderer computes it on the GPU in
one compute pass before each frame (the "§7" pre-pass) and writes the result
into the drawHeader, so the surface/fragment shaders just read it like any other
uniform.

```ts
import { derivedUniform } from "@aardworx/wombat.rendering/runtime";

// "TintBGR is Tint with its RGB channels swapped"
const tintBGR = derivedUniform((u) => u.Tint.swizzle("zyx"));

// MVP — same IR/hash as the built-in `ModelViewProjTrafo` recipe.
const mvp = derivedUniform((u) => u.ProjTrafo.mul(u.ViewTrafo).mul(u.ModelTrafo));

// Normal matrix (mat3, from a mat4 inverse-transpose):
const N = derivedUniform((u) => u.ModelTrafo.inverse().transpose().upperLeft3x3());
```

**On by default.** `enableDerivedUniforms: false` opts out. The 13 standard
trafo recipes (`ModelTrafo`, `ModelTrafoInv`, `NormalMatrix`,
`ModelViewTrafo`, `ModelViewProjTrafo`, `ViewProjTrafo`, …) are pre-shipped as
`STANDARD_DERIVED_RULES`; the heap renderer applies them automatically when an
RO declares those uniforms in its shader and binds the base trafos
(`ModelTrafo`/`ViewTrafo`/`ProjTrafo` as `aval<Trafo3d>`).

## Where rules slot in

Anywhere a uniform binding goes — the type stays `aval | DerivedRule`:

- `RenderObject.uniforms` / heap `spec.inputs[name]` (in
  `@aardworx/wombat.rendering`): a `DerivedRule` value is recognised by
  `isDerivedRule` and routed to the §7 dispatcher.
- `<Sg Uniform={…}>` / `Sg.uniform({...})` (in `@aardworx/wombat.dom`): rules
  pass through `uniformBag` and `splitTexturesFromUniforms` raw; the SG-to-heap
  path delivers them to `spec.inputs` where the heap renderer picks them up.

## The `(u) => …` DSL

`u.<Name>` is a leaf — it reads the uniform named `<Name>` on the RO.

- **Type.** Default `mat4x4<f32>` (the trafo case — covers most rules).
  - With the wombat-shader-vite plugin (`v0.3.2+`), the leaf's type is read
    automatically from the file's `declare module
    "@aardworx/wombat.shader/uniforms" { interface UniformScope { … } }`
    augmentation — so `u.Tint` is `vec4<f32>` if you declared `Tint: V4f`.
  - Without the plugin, use `u.<Name>.as("vec4" | "vec3" | "vec2" | "f32" |
    "u32" | "i32" | "mat3" | "mat4")` to re-type a leaf.
- `u.<Name>.inverse()` on a constituent trafo reads its stored backward half —
  free (no inverse computed at runtime).

`DerivedExpr` (the thing every `u.<Name>` and operation returns) has the full
WGSL builtin set as methods:

- `.add / .sub / .mul / .neg`
- `.transpose / .inverse / .upperLeft3x3 / .transformOrigin / .transformDir`
- `.swizzle("xyz")`, plus `.x / .y / .z / .w` getters
- `.sin .cos .tan .abs .floor .ceil .fract .sqrt .exp .log .exp2 .log2 .sign
   .normalize`
- `.min / .max / .pow / .atan2 / .step / .clamp / .mix`
- `.dot / .cross / .length / .distance / .reflect`
- `.call(name, resultType, ...others)` — escape hatch for any other WGSL
  builtin

The rule's output WGSL type follows from the ops (so `.upperLeft3x3()` yields
a `mat3` rule, `.swizzle("xyz")` of a vec4 yields vec3, etc.).

## What rules can't do

Rules **read uniforms only** — no textures, no samplers, no storage buffers,
no storage textures. The §7 compute kernel binds only the constituents heap,
the main heap (where uniforms are packed), the records buffer, and a count
uniform — there are no resource bindings at all. The build-time vite marker
rejects a `Sampler*`/`Texture*`/`Storage` leaf with a clear diagnostic; the
runtime `DerivedExpr` API simply has no methods to construct one.

A rule's body can't take a matrix `inverse()` either — WGSL has no `inverse`
builtin and the codegen doesn't synthesise one. The exception is
`Inverse(constituent-trafo-leaf)`, which is special-cased to read the
trafo's pre-computed backward half from the constituents storage.

## Two precision tiers

The codegen recognises the standard trafo shapes and emits **df32-precise**
arms for them (collapse, N-matmul chain, the normal-matrix transpose). Every
other rule lowers via a single-precision generic arm — the IR's expression
body is printed straight to WGSL (over leaves rewritten to function
parameters), with type-parametrised `load_<T>` / `store_<T>` helpers for the
types in play (f32, i32, u32, vecN<f32>, mat3, mat4).

## Build-time vs runtime

Both end up producing the same `DerivedRule`. The build-time
`@aardworx/wombat.shader-vite` plugin recognises `derivedUniform((u) => …)`
calls, reads each `u.<Name>` leaf's type from `UniformScope`, and appends a
hint map as the call's 2nd arg — so the runtime proxy mints correctly-typed
leaves with no `.as(...)` ceremony. With the plugin off (e.g. a unit test
without the vite pipeline), `u.<Name>` defaults to `mat4`; non-trafo leaves
need `.as("vec4")` / etc.

## What does §7 cost?

- **CPU:** O(changed avals) per frame. The constituents heap is keyed by aval
  identity, so a shared `ViewTrafo` aval has one (fwd, inv) slot pair on the
  GPU regardless of how many ROs read it. "View moved" → one `Set.add`, one
  `getValue`, one 64-float pack, one ~256-byte `writeBuffer`. **No CPU
  fan-out** to the records.
- **GPU:** one compute dispatch per frame over `recordCount` threads (1
  thread per (RO × derived uniform) pair). Each thread does its arm's math
  (mat·mat over df32 for the recipes, plain WGSL ops for generic rules) and
  writes the result. No dirty filter in v0 — every record runs every frame
  anything is dirty. At low millions of records this is sub-millisecond;
  past ~100k records, the future `dispatchWorkgroupsIndirect`-over-a-GPU-
  compacted-dirty-list path (deferred) is the next step.

See `derived-uniforms-extensible.md` for the design rationale, shape
classifier, slot-tag encoding, dispatcher recompile triggers, and the
deferred per-record dirty path.
