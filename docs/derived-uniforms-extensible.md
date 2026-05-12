# Extensible derived uniforms (§7 v2)

Replaces the hardcoded recipe table in `packages/rendering/src/runtime/derivedUniforms/` with a content-keyed rule registry. A uniform binding is now **either a value (an `aval`/constant) or a rule** (`DerivedRule`): rules are pure IR fragments over other uniforms; per RO they're flattened (chained references substituted away), registered with content-hash dedup, and each input leaf resolved to a tagged slot handle; the dispatcher codegen-emits one compute kernel that switches on `rule_id` and recompiles when the rule set (or record stride) changes.

**Status (as built).** Steps 1–11 of the plan below are done; `heap-demo?derived=1` runs through this path and matches `?derived=0`. The codegen has two tiers: **df32-precise** arms for the recognized trafo shapes (collapse / N-matmul chain / normal-matrix), and a **single-precision generic** path for any other IR (rewrite leaves → params, print the body, type-parametrised `load_<T>`/`store_<T>` for f32/i32/u32/vecN<f32>/mat3/mat4). The records buffer is **fixed-stride** (`2 + maxArity` u32s, grows + recompiles if a higher-arity rule appears) — not the variable-length+offsets design sketched in older revisions of this doc. There are **no dependency levels** — chains are flattened (see "Chain flattening"). There is **no "globals" concept** — whether a uniform is "global" is a per-RO fact (see the slot-tag table). df32 is *not* a first-class IR type; it lives in the constituent storage + the recognized trafo arms only. The author-facing `derivedUniform((u) => …)` is a runtime tracing builder (mat4 trafo leaves); a build-time vite marker with full type inference is still TODO.

---

## What stays from current §7

- The scene-wide records buffer model — **one dispatch per frame**, no dependency levels (chains flattened).
- The `_debug.{validateHeap, simulateDraws, probeBinarySearch, checkTriangleCoherence}` infrastructure.
- The constituent slots heap (`ConstituentSlots` — per-aval `Trafo3d` fwd/bwd storage in df32).
- The integration with `heapScene.addDraw` / `removeDraw` (the per-RO `registerRoDerivations` / `deregisterRoDerivations` shape; the per-frame `pullDirty`/`uploadDirty`/`encode`).

## What changes

- `RecipeId` enum + the static `RECIPES` table → **content-hash–keyed `DerivedUniformRegistry`** over flattened IR; the 13 recipes are ported to `DerivedRule` constants (`STANDARD_DERIVED_RULES`).
- The hand-written WGSL uber-kernel → **codegen-emitted** (`buildUberKernel`), recompiled on `registry.version` / record-stride change.
- A uniform binding was implicitly "always a value (an aval)" → now **value | `DerivedRule`** everywhere a uniform goes; `isDerivedRule(v)` routes rules to §7.
- `RoDerivedRequest.requiredNames` (a list of recipe names) → **`{ rules: Map<string, DerivedRule>, trafoAvals, hostUniformOffset, outputOffset, drawHeaderBaseByte }`**.
- The fixed-`MAX_INPUTS` `Record` struct → **fixed-stride packed records** at `2 + maxArity` u32s (grows + recompiles, rare).

---

## API surface

```ts
interface DerivedRule<T = unknown> {
  readonly __derivedRule: true;          // brand — `isDerivedRule` recognises a rule among uniform values
  readonly outputType: Type;             // `=== ir.type`; drives the storer + drawHeader byte budget
  readonly ir: Expr;                     // pure IR; every leaf input is `ReadInput("Uniform", name)` (or `Inverse(…)` of one)
  readonly hash: string;                 // structural content hash of `ir` — the registry dedup key
}

function isDerivedRule(x: unknown): x is DerivedRule;

// Author-facing builder (runtime tracing). `u.<Name>` is a mat4 trafo leaf; the
// DerivedExpr methods compose, and the result type follows from the ops.
function derivedUniform<T>(build: (u: DerivedScope) => DerivedExpr): DerivedRule<T>;

// Lower level: build a rule directly from hand-constructed IR.
function ruleFromIR<T>(ir: Expr, outputType?: Type): DerivedRule<T>;
```

`DerivedExpr` methods: `.mul` (matrix·matrix / matrix·vector by operand types), `.add` / `.sub` / `.neg`, `.inverse` (of a constituent trafo: reads its stored backward half — free), `.transpose`, `.upperLeft3x3` (→ mat3), `.transformOrigin` (matrix translation as vec3), `.swizzle("xyz")`. Examples:

```ts
const ModelViewTrafo     = derivedUniform(u => u.ViewTrafo.mul(u.ModelTrafo));
const ModelViewProjTrafo = derivedUniform(u => u.ProjTrafo.mul(u.ViewTrafo).mul(u.ModelTrafo));
const ModelViewTrafoInv  = derivedUniform(u => u.ModelTrafo.inverse().mul(u.ViewTrafo.inverse()));
const NormalMatrix       = derivedUniform(u => u.ModelTrafo.inverse().transpose().upperLeft3x3());   // → mat3
const CameraInWorld      = derivedUniform(u => u.ViewTrafo.inverse().transformOrigin());             // → vec3
```

Non-mat4-leaf rules (reading a `vec4` host uniform, a scalar, etc.) — and rules whose leaves are non-trafo host uniforms — aren't expressible through the runtime builder yet (it only mints mat4 leaves) and the codegen's host-uniform-leaf path isn't wired; use `ruleFromIR` with hand-built IR, or wait for the build-time vite marker (which will read leaf types from the program, à la `vertex(...)` / `fragment(...)`).

**Where rules slot in.** Anywhere a uniform value goes: the heap `spec.inputs[name]` (`aval<unknown> | DerivedRule`); a drawHeader field whose binding is a `DerivedRule` is produced by the §7 dispatcher instead of packed. *(Wiring the same into wombat.dom's `<Sg uniforms={…}>` / `TraversalState` / `IUniformProvider` is TODO — the value type there is still aval-only.)* A rule's leaves are resolved per RO: a matrix-typed leaf bound as an `aval<Trafo3d>` ⇒ a df32 constituent slot; anything else ⇒ a host uniform on that RO (host-leaf resolution not wired in v0).

---

## Registry

```ts
class DerivedUniformRegistry {
  private byHash = new Map<string, RuleEntry>();
  private nextId = 0;
  /** Bumps on every register(...) that finds a NEW hash. The
   *  dispatcher recompiles when its `lastVersion !== registry.version`. */
  version = 0;

  register(rule: DerivedRule): u32 {
    const cached = this.byHash.get(rule.hash);
    if (cached !== undefined) {
      cached.refcount++;
      return cached.id;
    }
    const entry: RuleEntry = {
      id: this.nextId++,
      ir: rule.ir,
      outputType: rule.outputType,
      inputs: extractInputs(rule.ir),  // [(name, wgslType), …]
      wgslFn: emitWgslFn(rule.ir, this.nextId - 1),
      refcount: 1,
    };
    this.byHash.set(rule.hash, entry);
    this.version++;
    return entry.id;
  }

  release(rule: DerivedRule): void {
    const entry = this.byHash.get(rule.hash);
    if (entry === undefined) return;
    entry.refcount--;
    // v0: never sweep. v1: sweep at next recompile-trigger.
  }

  emitKernel(): string { /* see Codegen below */ }
}

interface RuleEntry {
  id: u32;
  ir: IRFragment;
  outputType: WgslType;
  inputs: { name: string; wgslType: WgslType }[];
  wgslFn: string;     // `fn rule_<id>(in0: T0, in1: T1, …) -> Tout { … }`
  refcount: number;
}
```

**Dedup is content-based.** The `rule.hash` covers the IR's structure + named ReadInputs (`name + type`, NOT slot identity). Two `derivedUniform(u => u.View.mul(u.Model))` calls produced from any source files collapse to one entry.

**ID assignment** is monotonic. A removed-then-re-added rule gets a fresh id (we never reuse). Switch-arm sparsity is harmless to Tint.

---

## Chain flattening (no levels, ever)

A rule whose closure mentions another derived name — `derivedUniform(u => u.ViewProj.mul(u.Model))` where `ViewProj` is itself a `DerivedRule` — is **flattened at registration time**: the producer's IR is substituted in place of the `ReadInput("ViewProj")` node (its own `ReadInput`s renamed into the consumer's namespace), recursively, until the consumer's IR reads only *non-derived* inputs (constituents / host uniforms). The flattened IR is then CSE'd and content-hashed → that hash is the registry key.

Consequences:
- **Every record is independent.** No `RecordLevels`, no `maxLevel`, no inter-dispatch barrier, no `010` "reads another rule's output" tag. One dispatch per frame, full stop.
- **Diamonds don't blow up at runtime.** If `A` feeds both `B` and `C` and both feed `D`, `D`'s flattened IR contains `A` twice — but CSE on the flattened form collapses it to one computation in the emitted `fn rule_k`. Compile-time IR is a bit larger; the kernel isn't.
- **Producers that the shader *also* reads stay materialised.** If the vertex shader binds `ViewProj` *and* `MVP` derives from it, `ViewProj` keeps its own record (it's a name in the drawHeader the shader loads); `MVP` just doesn't *depend* on that record — it recomputes the product. Recompute on the GPU is cheap; the dependency isn't worth its complexity here.
- **Cross-RO dedup still works.** Two ROs that flatten `MVP` the same way share one `RuleEntry`. An RO where `ViewProj` is host-supplied (not derived) flattens differently (the `ReadInput("ViewProj")` stays, resolved to a host slot) and gets its own entry — correct, and rare.
- **Cycles** = a flattening that never terminates → registration error, detected by tracking the substitution stack.

`flatten(ir, ro.derivedUniforms)`:
```
walk ir; for each ReadInput(name, ty):
  if name in derivedUniforms: replace node with flatten(derivedUniforms[name].ir, …) under a fresh α-rename;  push name on stack, pop after; error if name already on stack
  else: leave it
then CSE the result.
```

Run once per `(ro, name)` at addRO; memoise on `derivedUniforms[name].hash` since the result depends only on the producer's IR and which names are derived (the *set* of derived names on the RO, captured in the stack walk — in practice the standard set, so the memo hits almost always).

The "expensive shared producer, compute once" alternative (keep levels, opt in per rule) is **explicitly out of scope** — revisit only if a profile shows a real flattened-recompute cost, which for mat·mat it won't.

---

## Records buffer (per scene)

**As built:** one scene-wide `RecordData: array<u32>` at a **fixed stride** of `2 + maxArity` u32s (`maxArity` = the largest input count of any registered rule; min stride 5 covers the 13 trafo recipes). Each record is `[ rule_id, out_slot, in_slot[0], …, in_slot[maxArity-1] ]` with the unused tail slots zero. No offsets array — the kernel does `let base = gid.x * STRIDE;`. If a higher-arity rule registers, the stride grows, `RecordData` is re-packed once, and the kernel recompiles (the registry-version bump triggers it anyway) — rare. Swap-remove keeps the array dense (move the tail record into the hole, patch its owner's index set).

> Older revisions of this doc described a variable-length blob + a `RecordOffsets` index (so a rule could read N inputs without a global `MAX_INPUTS`). The fixed-growable-stride form gets the same "no MAX_INPUTS" property — the stride is exactly `2 + maxArity` — with no offsets indirection and a trivial swap-remove, at the cost of a one-time re-pack + recompile when `maxArity` grows (which for the trafo recipes never happens). The sections below describe the records semantics; mentally substitute `gid.x * STRIDE` for `RecordOffsets[gid.x]`.

**Slot tagging** — `out_slot` and every `in_slot` is a tagged 32-bit handle; top 3 bits select the binding, low 29 bits are the payload:

| tag | meaning | payload |
|----|---------|---------|
| `000` | constituent slot — a `Trafo3d` half (`View.fwd`, `Model.bwd`, …) in df32 storage | slot index into `Constituents` |
| `001` | a host uniform — its data byte offset in the main heap. **This is whatever value reached this RO**: the scene-graph traversal already collapsed any subtree overrides, so there is nothing "global" to special-case. A uniform set once at the sg root and never overridden is the *same `aval`* on every RO ⇒ the UniformPool interns it to one heap slot; the only per-RO cost is the 4-byte ref. Same for trafos on the constituent side — `ConstituentSlots` keys by aval identity, so a shared `ViewTrafo` aval is one df32 slot pair shared by every RO that reads it. | byte offset into `MainHeap` |
| `010`–`111` | reserved (future: per-instance attribute arena, indirect double-precision pair, opt-in "shared producer" link, …) | — |

(No "reads another rule's output" tag — chains are flattened away at registration, so every input is a constituent or a host uniform. And **no "global" tag**: "is this uniform global?" is a per-RO question, not a property of the name — the sg can override `LightLocation`, `ViewportSize`, anything, anywhere — so it's resolved per RO like everything else. Conceptually *every* uniform (`ViewTrafo` included) lives per-RO and just happens to collapse to one shared slot via aval interning.)

The loader for input `i` is chosen by **two** things: the tag (which binding) and `rule.inputs[i].wgslType` (how many components, what bit-layout, df32 or not). Codegen emits `loadAs_<wgslType>(tag, payload)` per arm. Likewise the storer is `storeAs_<outputType>(out_slot_tag, out_payload, value)`.

**Per-RO addRO**:
```ts
for (const [name, rule] of ro.derivedUniforms) {
  const flat    = flatten(rule.ir, ro.derivedUniforms);          // see "Chain flattening" — reads only non-derived inputs
  const ruleId  = registry.register({ ir: flat, outputType: rule.outputType, hash: hashIR(flat) });
  const outSlot = taggedHandle(TAG_HOST_HEAP, drawHeaderByteOf(ro, name));
  const inputs  = inputsOf(flat).map(({ name: inName, wgslType }) =>
    resolveSource(ro, inName, wgslType)   // → tagged handle, see resolveSource below
  );
  records.add(ro, { ruleId, outSlot, inputs });
}
```

`resolveSource(ro, name, wgslType)` (post-flatten, so `name` is never another derived uniform):
1. `name` resolves to a `Trafo3d` aval (`Model`, `View`, `Proj`, …) → tag `000`, allocate/find the constituent slot for that aval (the existing per-aval interning), payload = slot index.
2. `ro.uniforms` (the RO's resolved uniform set — sg overrides already collapsed) has `name` → tag `001`, payload = its data byte offset in the main heap. This covers everything else, "auto-uniforms" included; there is no separate global path.
3. otherwise → error: the rule references an input this RO can't supply.

Two ROs with the same rule but different `View` avals get records pointing to different constituent slots; same `View` aval ⇒ same slot (interned).

**Per-RO removeDraw**:
```ts
for (const [name, rule] of ro.derivedUniforms) registry.release(/* the flattened entry, keyed by hashIR(flatten(rule.ir, …)) — memoised */);
records.removeAllForRo(ro);   // existing swap-remove machinery, now operating on the packed blob + offsets
```

The swap-remove on a variable-length blob: keep a per-RO list of `(recordIndex, byteSpan)`; on remove, the standard tail-swap moves the last record's bytes into the hole and patches its `RecordOffsets` entry. Slightly more bookkeeping than fixed-stride, but the same O(removed) cost.

---

## Codegen

The kernel is emitted from the registry every time `version` changes — **one dispatch per frame** (chains are flattened, so there are no levels). Everything below — arity, input types, output type, loaders, storers — is generated from `RuleEntry`; nothing is hand-written per rule.

```wgsl
@group(0) @binding(0) var<storage, read>       Constituents:  array<u32>;   // df32 trafo halves, raw words
@group(0) @binding(1) var<storage, read_write> MainHeap:      array<u32>;   // drawHeaders, raw words (host uniforms + rule outputs)
@group(0) @binding(2) var<storage, read>       RecordData:    array<u32>;   // packed records (fixed stride)
@group(0) @binding(3) var<uniform>             Dispatch:      DispatchUbo;   // { count: u32 }

// ---- generic typed loaders/storers, emitted once for each (wgslType) actually used ----
// each decodes the 3-bit tag (constituent / host-heap) and reads
// the right number of words with the right bit layout. df32 forms read 2× words and reassemble.
fn load_mat4x4_f32(h: u32) -> mat4x4<f32> { /* tag-switch → words → mat4 (df32 → collapse to f32) */ }
fn load_df32_mat4(h: u32) -> Df32Mat4    { /* tag-switch → 2× words → {hi, lo} */ }
fn load_vec3_f32(h: u32)  -> vec3<f32>   { /* … */ }
fn load_u32(h: u32)       -> u32         { /* … */ }
// …only the (type) combos some registered rule needs get emitted.
fn store_mat3x3_f32(h: u32, v: mat3x3<f32>) { /* tag → byte offset in MainHeap → write 9 (padded) words */ }
fn store_df32_mat4(h: u32, v: Df32Mat4)    { /* … 32 words … */ }
fn store_bool(h: u32, v: bool)              { /* … 1 word, 0/1 … */ }

// ---- one fn per registered rule (closure body lowered from IR; df32 ops use the df32 helper lib) ----
fn rule_0(a: Df32Mat4, b: Df32Mat4) -> Df32Mat4 { return df32_mat4_mul(a, b); }                 // ModelView
fn rule_1(a: mat4x4<f32>) -> mat3x3<f32> { return upper_left_3x3(transpose(invert4(a))); }       // NormalMatrix
fn rule_2(p: Df32Mat4, v: Df32Mat4, m: Df32Mat4) -> Df32Mat4 { return df32_mat4_mul(df32_mat4_mul(p, v), m); }  // MVP, inline triple
fn rule_7(c: vec3<f32>) -> bool { return c.z > 0.0; }                                            // HasLighting
fn rule_9(res: u32) -> f32 { return 1.0 / f32(res); }                                            // TexelSize

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= Dispatch.count) { return; }
  let i = gid.x;
  let off      = RecordOffsets[i];
  let rule_id  = RecordData[off];
  let out_slot = RecordData[off + 1u];
  switch rule_id {
    case 0u: { let a = load_df32_mat4(RecordData[off+2u]); let b = load_df32_mat4(RecordData[off+3u]);
               store_df32_mat4(out_slot, rule_0(a, b)); }
    case 1u: { let a = load_mat4x4_f32(RecordData[off+2u]);
               store_mat3x3_f32(out_slot, rule_1(a)); }
    case 2u: { let p = load_df32_mat4(RecordData[off+2u]); let v = load_df32_mat4(RecordData[off+3u]); let m = load_df32_mat4(RecordData[off+4u]);
               store_df32_mat4(out_slot, rule_2(p, v, m)); }
    case 7u: { let c = load_vec3_f32(RecordData[off+2u]);
               store_bool(out_slot, rule_7(c)); }
    case 9u: { let r = load_u32(RecordData[off+2u]);
               store_f32(out_slot, rule_9(r)); }
    default: { return; }
  }
}
```

**Per-arm codegen** — for `RuleEntry e` with id `k`:
1. for each `e.inputs[i]`: emit `let a{i} = load_{wgslSym(e.inputs[i].wgslType)}(RecordData[off + ${2+i}u]);`
2. emit `store_{wgslSym(e.outputType)}(out_slot, rule_${k}(a0, a1, …));`
3. ensure the `load_*` / `store_*` for every type used got emitted (a `Set<WgslType>` collected across the registry).

**The IR lowering for `rule_k`** is the existing IR → WGSL emitter, with one addition: when a value's type is a df32 form, arithmetic lowers to the df32 helper library (`df32_add`, `df32_mul`, `df32_mat4_mul`, `df32_to_f32`, …). That library is emitted into the kernel prelude whenever any rule touches df32. This is exactly the math the current §7 hand-codes for trafos — now it's a reusable helper set the codegen calls, not 13 copies.

**drawHeader byte budget.** `name`'s slot in the RO's drawHeader must be sized for `outputType` (a df32 mat4 needs 32 words, a `bool` needs 1, a `mat3x3<f32>` needs 12 with std430 padding). `buildBucketLayout` already knows each uniform's type; it just needs to accept derived-uniform names with their `outputType` as part of the schema. No change to how the *vertex/fragment* shader reads them — they read `name` from the drawHeader the same way they read any host uniform.

**No globals binding.** There's nothing "global" to bind: `u.ViewportSize`, `u.LightLocation`, the camera matrices — these are all just uniforms that reached the RO (the sg traversal can override any of them, anywhere), so they're resolved per RO like everything else (tag `001`, their byte offset in the main heap). A uniform that genuinely is the same for every RO is the same `aval` everywhere ⇒ the UniformPool already interns it to one heap slot; the only per-RO cost is the 4-byte ref. Trafos get the same treatment on the constituent side via `ConstituentSlots`'s aval-identity keying.

---

## Dispatcher (one fat pipeline, one dispatch, recompile on version change)

Chains are flattened at registration (see "Chain flattening"), so every record is independent — one `dispatchWorkgroups` covers the whole scene, no barriers.

```ts
class DerivedUniformsDispatcher {
  private compiledVersion = -1;
  private pipeline: GPUComputePipeline | undefined;

  encode(enc: GPUCommandEncoder): void {
    if (this.records.recordCount === 0) return;
    if (this.registry.version !== this.compiledVersion) {
      this.recompile();   // v1 sweeps refcount=0 entries first
    }
    this.records.flush();
    const pass = enc.beginComputePass({ label: "derivedUniforms.uber" });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroup());
    pass.dispatchWorkgroups(Math.ceil(this.records.recordCount / 64));
    pass.end();
  }

  private recompile(): void {
    const wgsl = this.registry.emitKernel();
    const module = this.device.createShaderModule({ code: wgsl });
    this.pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
      compute: { module, entryPoint: "main" },
    });
    this.cachedBg = undefined;   // bind group rebuilt next bindGroup() call
    this.compiledVersion = this.registry.version;
  }
}
```

Recompile triggers:
- New unique (flattened) rule registered.
- (v1) Sweep at recompile time drops dead `RuleEntry`s and regenerates WGSL.

Recompile cost: one `createShaderModule` + one `createComputePipeline`. Tens of ms in the worst case. Happens at scene-build time and on subsequent unique-rule additions (rare). Hot path is one dispatch per frame, no recompile.

---

## Lifecycle of removed rules

**v0**: Never remove. `RuleEntry` accumulates monotonically. Records pointing at a rule are removed via the existing swap-remove machinery; the rule's switch arm becomes empty (no records dispatched at it). Memory cost: bytes per rule.

**v1**: Refcount-tracked, lazy GC at recompile time.
- `register(rule)`: refcount++.
- `release(rule)`: refcount--. **Don't recompile here.**
- Next `recompile()` (triggered by some new register): sweep entries with `refcount === 0`, drop them, regenerate WGSL without them, then compile.
- Rule reappearing later: registers fresh, gets new ID, triggers recompile.

ID gaps from sweeps are harmless. Tint compiles sparse switches identically. We never reuse IDs.

**v2**: Aggressive GC (recompile every time a refcount hits 0). Only worth it if v1 still bloats. Probably never needed.

Ship v0. Add v1 when someone profiles the kernel size and finds it embarrassing.

---

## Dirty propagation

**v0 (now): recompute everything every frame.** The kernel runs every record every frame. With ~mat·mat per record and the records buffer in the low millions, GPU cost is sub-millisecond and simpler is better. Ship this.

**End state: proper per-value tracking.** The granularity that matters is per-input-aval, not per-object and not per-semantic-name (a moving camera dirties `View`/`Proj` → most rules anyway, so coarse masks barely help). §7 already tracks aval dirtiness on the input side (`dirtyAvals` set + `pullDirty`); the pieces to add:

- **Reverse index** `constituentSlot → [recordIndex…]` (and the same for host-uniform sources), maintained on `addRO` / `removeDraw`. A record subscribes to every slot its flattened rule reads.
- **Compaction kernel** (GPU-side, runs each frame *before* the uber-kernel): walk the dirty-slot list → for each, walk its records → `atomicAdd` a counter and append the record index to a `DirtyRecords: array<u32>` buffer. A trailing 1-thread step writes `(ceil(count/64), 1u, 1u)` into a small `indirectArgs` buffer (`usage: INDIRECT | STORAGE`).
- **Indirect dispatch**: `pass.dispatchWorkgroupsIndirect(indirectArgs, 0)` runs the uber-kernel; `main` indexes through `RecordOffsets[DirtyRecords[gid.x]]` instead of `RecordOffsets[gid.x]`, guarded by `gid.x >= count` read from the same counter buffer. No CPU round-trip, no fixed upper bound on dirty count.
- Fully-static frame ⇒ dirty count 0 ⇒ `dispatchWorkgroupsIndirect` with `(0,1,1)` ⇒ nothing runs.

The records buffer layout doesn't change between v0 and the end state — the dirty path is purely additive (reverse index + two extra buffers + the compaction kernel + swap `dispatchWorkgroups` for `dispatchWorkgroupsIndirect`). Build v0; add the dirty path when a profile shows the uber-dispatch's memory traffic mattering (roughly: north of ~100k records that actually change rarely).

---

## What goes where (file layout)

```
packages/rendering/src/runtime/derivedUniforms/
  index.ts             ← public exports
  rule.ts              ← `DerivedRule` type + `derivedUniform(...)` marker
  flatten.ts           ← `flatten(ir, derivedNames)`, `hashIR`, CSE, cycle detection (memoised)
  registry.ts          ← `DerivedUniformRegistry`, content-hash dedup over flattened IR
  codegen.ts           ← `emitKernel(registry)`, type-parametrised loaders/storers, df32 prelude
  records.ts           ← fixed-growable-stride `RecordsBuffer` (packed u32 blob, per-RO index sets, tail-swap)
  slots.ts             ← `ConstituentSlots` (existing, df32 trafo storage)
  dispatch.ts          ← `DerivedUniformsDispatcher`, recompile-on-version, single dispatch
  sceneIntegration.ts  ← `DerivedUniformsScene`, `registerRoDerivations`
                          (existing surface, swap RecipeId for flatten+register)
```

The current `recipes.ts` and the hardcoded `uberKernel.wgsl.ts` go away.

---

## Migration plan

1. Define `DerivedRule` + `derivedUniform` (stub IR — accept hand-built fragments first; defer the inline-marker plugin work). `WgslType` already exists in wombat.shader; reuse it.
2. `flatten(ir, derivedNames)` + `hashIR` + CSE on the flattened form; memoised on producer hash. Cycle detection. This is what makes "no levels" true; do it early so everything downstream only ever sees flattened rules.
3. Build `DerivedUniformRegistry` with content-hash dedup over flattened IR (`extractInputs`, `emitWgslFn`).
4. Write `codegen.ts`:
   - **generic typed loaders/storers**, generated per `WgslType` used across the registry — handle the source tags (constituent / host-heap) and all of: f32/f16/i32/u32/bool, vecN, matNxM<f32>, df32 forms (df32 reads/writes 2× words and reassembles). One emitter parametrised by type; not a fixed menu.
   - df32 helper library (`df32_add/mul/mat4_mul/to_f32/…`) emitted into the prelude when any rule uses df32 — the math the current §7 hand-codes, factored out.
   - per-rule `fn rule_k(...)` from the IR emitter (with df32-aware lowering).
   - the `main` loop: read `RecordOffsets`/`RecordData`, switch over rule ids, one dispatch.
5. `records.ts`: fixed-stride (`2 + maxArity`) packed `RecordData` u32 blob, per-RO index sets, tail-swap removal, stride growth bumps a layout version. No levels.
6. `dispatch.ts`: recompile-on-version, one `dispatchWorkgroups` per frame.
7. `sceneIntegration.registerRoDerivations` takes `HashMap<string, DerivedRule>`; flattens each rule against the RO's derived-name set, registers the flattened entry, implements `resolveSource` (constituent → host); bumps the constituent interning for `Trafo3d` inputs as today.
8. `buildBucketLayout` / `HeapEffectSchema`: size derived-uniform drawHeader slots for their `outputType` (df32 mat4 = 32 words, bool = 1, …) — currently moot, the arena slots are `PACKER_MAT4`-sized which fits mat4 and mat3. Vertex/fragment shaders read them unchanged.
9. In `heapScene.addDraw`, iterate the RO's derived rules instead of a hardcoded name set; wire `hostUniformOffset` against the RO's already-packed uniform refs.
10. Port the existing 13 hardcoded recipes to standalone `DerivedRule` constants (`STANDARD_DERIVED_RULES`); heapScene installs them. *(done)*
11. Wire wombat-shader-vite plugin to recognize `derivedUniform(...)` markers and emit IR (last; can ship without — hand-built IR fragments work today).

Status: steps 1–10 done — the standard trafo recipes run end-to-end through the new path (`heap-demo?derived=1` matches `?derived=0`), with the codegen restricted to the recipe shapes (collapse / N-matmul chain / normal-matrix). Widening the codegen to arbitrary IR (the `expr()`-lowered path) and step 11 are the remaining work; there were never any levels to retrofit.
