# Extensible derived uniforms (§7 v2)

Replace the hardcoded recipe table in `packages/rendering/src/runtime/derivedUniforms/` with a per-RO, content-keyed rule registry. Each `RenderObject` carries an arbitrary map `name → DerivedRule`; rules are inline-marker closures over other uniforms; the dispatcher generates one fat compute kernel that switches on `rule_id` and recompiles when the rule set changes.

This document is the design only — no code yet.

**End state: arbitrary input count, arbitrary WGSL input/output types, arbitrary chaining.** A `DerivedRule` may read any number of inputs of any type (scalars, vectors, matrices, integer/bool, df32 doubles, the constituent `Trafo3d` halves) and produce any WGSL type. Rules may consume the outputs of other rules to any depth. The phased plan below ships the common trafo derivations first, but every data-structure choice here is sized for the general case — no fixed `MAX_INPUTS`, no "mat4-only" loaders, no "one dispatch, hope the ordering works out" for chains. Double precision is a first-class input/output type, not a special case hand-coded into one kernel.

---

## What stays from current §7

- The scene-wide records buffer model (one dispatch per dependency level — usually one level; see "Chained rules").
- The `_debug.{validateHeap, simulateDraws, probeBinarySearch, checkTriangleCoherence}` infrastructure.
- The constituent slots heap (per-aval `Trafo3d` storage in df32).
- The integration with `heapScene.addDraw` / `removeDraw`.

## What changes

- `RecipeId` enum and the static `RECIPES` table → **runtime rule registry** keyed by IR-hash.
- The fat hand-written WGSL kernel with 13 hardcoded arms → **codegen-emitted kernel**, recompiled on registry version bump. Codegen handles any rule arity and any WGSL input/output type, df32 included.
- `RoDerivedRequest.requiredNames` (a list of recipe names) → **`HashMap<string, DerivedRule>`** carried directly on the `RenderObject`.
- Fixed-width `Record` struct with `MAX_INPUTS` → **variable-length packed records** + a `recordOffsets` index, so a rule can read N inputs.
- Single dispatch → **one dispatch per dependency level**, so a rule may read another rule's output (the common case stays a single level).

---

## API surface

```ts
interface DerivedRule<T = unknown> {
  /** Output WGSL type — any of: f32 f16 i32 u32 bool, vecN of those,
   *  matNxM<f32>, the df32 "double" forms (df32, df32 vecN, df32 matNxM).
   *  Drives the storer codegen and the drawHeader byte budget for `name`. */
  readonly outputType: WgslType;
  /** IR fragment. The IR's `ReadInput("Uniform", <name>, <type>)` nodes
   *  enumerate this rule's inputs — ANY count, ANY of the types above.
   *  Sources are resolved per-RO at registration time. */
  readonly ir: IRFragment;
  /** Stable content hash of `ir` (post-CSE/DCE canonical form). Covers
   *  structure + the (name, type) of every ReadInput, not slot identity. */
  readonly hash: string;
}

function derivedUniform<T>(closure: (u: UniformScope) => T): DerivedRule<T>;
```

`derivedUniform(...)` is an inline-marker (analogous to `vertex(...)` / `compute(...)`) that the wombat-shader-vite plugin lowers at build time. Each captured `u.<Name>` becomes a `ReadInput` node; the closure body becomes a function body that returns the IR expression. Whatever types the closure's arithmetic implies (including df32 — `u.View` is a `Trafo3d`, i.e. a df32 mat4) flow through to `inputs[i].wgslType` / `outputType`.

Examples — note the spread of arities and types:
```ts
const ModelView    = derivedUniform(u => u.View.mul(u.Model));                 // 2× df32 mat4 → df32 mat4
const ModelViewProj = derivedUniform(u => u.Proj.mul(u.View).mul(u.Model));     // 3× df32 mat4 → df32 mat4 (or one inline triple-product rule)
const NormalMatrix = derivedUniform(u => u.Model.inverse().transpose().upperLeft3x3()); // 1× mat4 → mat3x3<f32>
const CameraInWorld = derivedUniform(u => u.View.inverse().transformPos(vec3(0.0))); // 1× df32 mat4 → vec3<f32>
const HasLighting  = derivedUniform(u => u.LightLocation.z > 0.0);              // 1× vec3<f32> → bool
const TexelSize    = derivedUniform(u => 1.0 / f32(u.AtlasResolution));         // 1× u32 → f32
const Tinted       = derivedUniform(u => u.BaseColor * u.TintFactor);           // vec4<f32> × f32 → vec4<f32>  (host-supplied inputs, not trafos)
```

`UniformScope` is structurally typed so any `u.<Name>` is legal; the marker emits a `ReadInput(name, inferredType)` and the type checker on the recipe side validates the closure. No registry of "known" uniform names — sources are resolved per-RO (see Records).

`RenderObject` gains:
```ts
interface RenderObject {
  // ... existing fields ...
  readonly derivedUniforms?: HashMap<string, DerivedRule<unknown>>;
}
```

The heap renderer cross-references the effect's required-uniforms list against `ro.derivedUniforms` — names present here are produced by the §7 dispatcher; names absent are served by the standard `spec.inputs` path. Same routing logic that exists in current §7, just the source is dynamic per-RO instead of a fixed name set.

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

A rule whose closure mentions another derived name — `derivedUniform(u => u.ViewProj.mul(u.Model))` where `ViewProj` is itself a `DerivedRule` — is **flattened at registration time**: the producer's IR is substituted in place of the `ReadInput("ViewProj")` node (its own `ReadInput`s renamed into the consumer's namespace), recursively, until the consumer's IR reads only *non-derived* inputs (constituents / host uniforms / globals). The flattened IR is then CSE'd and content-hashed → that hash is the registry key.

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

A rule has arbitrary arity (and every record is independent — see "Chain flattening"), so records are **variable length, packed into a flat `u32` blob** with a parallel offsets index:

```
RecordOffsets : array<u32>          // recordOffsets[i] = start of record i in RecordData (one entry per derived-uniform instance)
RecordData    : array<u32>          // packed: [ rule_id, out_slot, in_slot[0], in_slot[1], … ]  — arity known from rule_id
```

A single record is `2 + arity(rule_id)` u32s. The kernel for invocation `gid.x` does:
```wgsl
let off     = RecordOffsets[gid.x];
let rule_id = RecordData[off];
let out_slot = RecordData[off + 1u];
// arity is baked into the switch arm for rule_id; it reads RecordData[off + 2u .. off + 2u + arity]
```

(If `array<u32>` indexing chains hurt, the alternative is sorting records by `rule_id` and emitting one fixed-stride sub-dispatch per rule — codegen knows every rule's stride. Start with the offsets blob; it's simpler and the indirection is one extra load.)

**Slot tagging** — `out_slot` and every `in_slot` is a tagged 32-bit handle; top 3 bits select the binding, low 29 bits are the payload (byte offset, slot index, or sub-record index):

| tag | meaning | payload |
|----|---------|---------|
| `000` | constituent slot — a `Trafo3d` half (`View.fwd`, `Model.bwd`, …) in df32 storage | slot index into `Constituents` |
| `001` | drawHeader byte offset of a **host-supplied** uniform on the same RO (`u.BaseColor`, `u.AtlasResolution`, …) | byte offset into `MainHeap` |
| `010` | a **shared/global** uniform (camera, viewport, time…) in the global uniform buffer | byte offset into `Globals` |
| `011`–`111` | reserved (future: per-instance attribute arena, indirect double-precision pair, opt-in "shared producer" link, …) | — |

(There is no "reads another rule's output" tag — chains are flattened away at registration, so every input is a constituent / host uniform / global.)

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
1. `name` resolves to a `Trafo3d` aval (`View`, `Model`, `Proj`, `ViewProj`, …) → tag `000`, allocate/find the constituent slot for that aval (the existing per-aval interning), payload = slot index.
2. `ro.uniforms.tryFind(name)` is one of the global auto-uniforms (camera/viewport/time/…) → tag `010`, payload = field byte offset in `GlobalsUbo`.
3. `ro.uniforms.tryFind(name)` is a host-supplied constant/value → tag `001`, payload = its drawHeader byte offset.
4. otherwise → error: rule references an input the RO can't supply.

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
@group(0) @binding(2) var<uniform>             Globals:       GlobalsUbo;    // camera/viewport/time/…
@group(0) @binding(3) var<storage, read>       RecordOffsets: array<u32>;
@group(0) @binding(4) var<storage, read>       RecordData:    array<u32>;
@group(0) @binding(5) var<uniform>             Dispatch:      DispatchUbo;   // { count: u32 }

// ---- generic typed loaders/storers, emitted once for each (wgslType) actually used ----
// each decodes the 3-bit tag (constituent / host-heap / rule-output-heap / globals) and reads
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

**Globals binding.** Shared uniforms (camera location, viewport, frame time, …) live in one UBO the kernel binds read-only. A rule that reads `u.ViewportSize` gets a tag-`011` handle whose payload is the field's byte offset in `GlobalsUbo`. (These are the same auto-uniforms `TraversalState` already exposes; the heap renderer just needs to mirror them into a GPU UBO, which it largely does already for the per-frame block.)

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

- **Reverse index** `constituentSlot → [recordIndex…]` (and the same for host-uniform / globals sources), maintained on `addRO` / `removeDraw`. A record subscribes to every slot its flattened rule reads.
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
  records.ts           ← variable-length `RecordsBuffer` (packed blob + offsets, per-RO spans, tail-swap)
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
   - **generic typed loaders/storers**, generated per `WgslType` used across the registry — handle all three source tags (constituent / host-heap / globals) and all of: f32/f16/i32/u32/bool, vecN, matNxM<f32>, df32 forms (df32 reads/writes 2× words and reassembles). One emitter parametrised by type; not a fixed menu.
   - df32 helper library (`df32_add/mul/mat4_mul/to_f32/…`) emitted into the prelude when any rule uses df32 — the math the current §7 hand-codes, factored out.
   - per-rule `fn rule_k(...)` from the IR emitter (with df32-aware lowering).
   - the `main` loop: read `RecordOffsets`/`RecordData`, switch over rule ids, one dispatch.
5. Variable-length `records.ts`: packed `RecordData` u32 blob + `RecordOffsets`, per-RO span tracking, tail-swap removal. No levels.
6. `dispatch.ts`: recompile-on-version, one `dispatchWorkgroups` per frame.
7. `sceneIntegration.registerRoDerivations` takes `HashMap<string, DerivedRule>`; flattens each rule against the RO's derived-name set, registers the flattened entry, implements `resolveSource` (constituent → globals → host); bumps the constituent interning for `Trafo3d` inputs as today.
8. `buildBucketLayout` / `HeapEffectSchema`: accept derived-uniform names with their `outputType` so the drawHeader reserves the right byte budget (df32 mat4 = 32 words, bool = 1, …). Vertex/fragment shaders read them unchanged.
9. In `heapScene.addDraw`, iterate `ro.derivedUniforms` instead of a hardcoded name set.
10. Mirror the shared auto-uniforms into a GPU `GlobalsUbo` the kernel binds (mostly exists as the per-frame block); wire tag-`010` resolution.
11. Port the existing 13 hardcoded recipes to standalone `DerivedRule` constants exported alongside `derivedUniform` for back-compat. Eventually demos define their own.
12. Wire wombat-shader-vite plugin to recognize `derivedUniform(...)` markers and emit IR (last; can ship without if the user accepts hand-built fragments).

Each step is independently testable. A useful intermediate milestone: steps 1–9 with the loader/storer emitter restricted to df32-mat4 + f32-mat4 + f32-mat3 types proves the architecture end-to-end on the standard recipes; widening the type set in step 4 is then purely additive (no structural change), arity is unbounded from step 5, and there were never any levels to retrofit. Steps 10–12 are reach and polish.
