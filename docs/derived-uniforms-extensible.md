# Extensible derived uniforms (§7 v2)

Replace the hardcoded recipe table in `packages/rendering/src/runtime/derivedUniforms/` with a per-RO, content-keyed rule registry. Each `RenderObject` carries an arbitrary map `name → DerivedRule`; rules are inline-marker closures over other uniforms; the dispatcher generates one fat compute kernel that switches on `rule_id` and recompiles when the rule set changes.

This document is the design only — no code yet. Skip double-precision in v0; current §7 hand-codes df32 inside the kernel and that's fine to keep for the standard trafo derivations, but new rules use plain f32 math until a follow-up.

---

## What stays from current §7

- The single-dispatch-per-frame, scene-wide records buffer model.
- The `_debug.{validateHeap, simulateDraws, probeBinarySearch, checkTriangleCoherence}` infrastructure.
- The constituent slots heap (per-aval `Trafo3d` storage in df32).
- The integration with `heapScene.addDraw` / `removeDraw`.

## What changes

- `RecipeId` enum and the static `RECIPES` table → **runtime rule registry** keyed by IR-hash.
- The fat hand-written WGSL kernel with 13 hardcoded arms → **codegen-emitted kernel**, recompiled on registry version bump.
- `RoDerivedRequest.requiredNames` (a list of recipe names) → **`HashMap<string, DerivedRule>`** carried directly on the `RenderObject`.

---

## API surface

```ts
interface DerivedRule<T = unknown> {
  /** Output WGSL type (mat4x4<f32>, mat3x3<f32>, vec4<f32>, …). */
  readonly outputType: WgslType;
  /** IR fragment. The IR's `ReadInput("Uniform", <name>, <type>)` nodes
   *  enumerate this rule's inputs by name + type. Sources are resolved
   *  per-RO at registration time. */
  readonly ir: IRFragment;
  /** Stable content hash of `ir` (post-CSE/DCE canonical form). */
  readonly hash: string;
}

function derivedUniform<T>(closure: (u: UniformScope) => T): DerivedRule<T>;
```

`derivedUniform(...)` is an inline-marker (analogous to `vertex(...)` / `compute(...)`) that the wombat-shader-vite plugin lowers at build time. Each captured `u.<Name>` becomes a `ReadInput` node; the closure body becomes a function body that returns the IR expression.

Examples:
```ts
const ModelView = derivedUniform(u => u.View.mul(u.Model));
const NormalMatrix = derivedUniform(u => u.Model.inverse().transpose().upperLeft3x3());
const HasLighting = derivedUniform(u => u.LightLocation.z > 0.0);
```

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

## Records buffer (per scene)

Layout unchanged from current §7:
```
struct Record {
  rule_id:  u32,
  in_slot:  array<u32, MAX_INPUTS>,  // top bits tag the source
  out_byte: u32,
}
```

`MAX_INPUTS` = 3 covers `mat·mat·mat`. Triple-products handled either inline (one rule reads 3 inputs) or as a chain (two rules with one intermediate slot); the rule's IR decides.

**Slot tagging** (top 2 bits of each `in_slot`):
- `00` — constituent slot (per-aval `Trafo3d` half: `View.fwd`, `Model.bwd`, …, df32 storage).
- `01` — drawHeader byte offset of a host-supplied uniform on the same RO (e.g. user-bound `Tint: vec4<f32>`).
- `10` — output of another rule on the same RO (chained derivation; output already written to that RO's drawHeader).
- `11` — reserved.

The kernel decodes the tag and reads through the appropriate binding.

**Per-RO addRO**:
```ts
for (const [name, rule] of ro.derivedUniforms) {
  const ruleId  = registry.register(rule);
  const outByte = drawHeaderOffsetOf(ro, name);
  const inputs  = rule.inputs.map(({ name: inName, wgslType: ty }) =>
    resolveSource(ro, inName, ty)  // → tagged slot ID
  );
  records.add({ rule_id: ruleId, in0..2: inputs, out_byte: outByte });
}
```

`resolveSource` looks up the input name against `ro.uniforms` (host-supplied) first, then `ro.derivedUniforms` (chained), then falls back to the constituent layer for `Trafo3d` avals. Two ROs with the same rule but different `View` avals get records pointing to different constituent slots.

**Per-RO removeDraw**:
```ts
for (const [name, rule] of ro.derivedUniforms) {
  registry.release(rule);
}
records.removeAllForRo(ro);   // existing swap-remove machinery
```

---

## Codegen

The kernel is emitted from the registry every time `version` changes:

```wgsl
@group(0) @binding(0) var<storage, read>       Constituents:  array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> MainHeap:      array<f32>;
@group(0) @binding(2) var<storage, read>       Records:       array<Record>;
@group(0) @binding(3) var<uniform>             Count:         CountUniform;

// One emitted function per registered rule.
fn rule_0(in0: mat4x4<f32>, in1: mat4x4<f32>) -> mat4x4<f32> { return in0 * in1; }
fn rule_1(in0: mat4x4<f32>) -> mat3x3<f32> { /* upperLeft3x3(transpose(inverse(in0))) */ }
fn rule_42(in0: vec3<f32>) -> f32 { return select(0.0, 1.0, in0.z > 0.0); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= Count.count) { return; }
  let r = Records[gid.x];
  switch r.rule_id {
    case 0u: {
      let a = loadMat4(r.in_slot[0]);
      let b = loadMat4(r.in_slot[1]);
      writeMat4(r.out_byte, rule_0(a, b));
    }
    case 1u: {
      let a = loadMat4(r.in_slot[0]);
      writeMat3(r.out_byte, rule_1(a));
    }
    case 42u: {
      let a = loadVec3(r.in_slot[0]);
      writeF32(r.out_byte, rule_42(a));
    }
    default: { return; }
  }
}
```

`loadMat4` / `loadMat3` / `loadVec*` / `loadF32` decode the slot tag and read from the right binding. Type info from `RuleEntry.inputs[i].wgslType` and `outputType` drives which loader/storer to invoke per arm.

The arm body is generated, NOT hand-written — for each rule, codegen walks `inputs[i].wgslType` and emits the right loader call, then the typed function call, then the storer.

**Loader emission examples**:
- `mat4x4<f32>` from constituent: same df32 collapse pattern that today's `run_collapse_mat4` uses.
- `mat4x4<f32>` from drawHeader: `loadMat4FromHeap(out_byte_offset_decoded)` — already exists in heap renderer's IR loaders.
- `vec3<f32>` from constituent: not currently a thing (constituents are df32 mat4); add later.

For v0 we can punt non-mat4 inputs and require all input types to be `mat4x4<f32>`. Covers the standard derivations.

---

## Dispatcher (one fat pipeline, recompile on version change)

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
- New unique rule registered.
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

## Dirty propagation (deferred)

For v0, the kernel runs **every record every frame**. With ~mat·mat per record and the records buffer in the low millions, GPU cost is fine and simpler is better.

For a future v1, the existing §7 dirty-tracking machinery (constituent `dirtyAvals` set + `pullDirty`) extends naturally:
- Each input source's "dirty" propagates to every record that reads it.
- Per-frame: walk dirty inputs → mark dirty record set → dispatch ranged over dirty records (e.g. via indirect dispatch with atomic compaction, or sub-pass dispatch over a contiguous sorted-dirty subset).

Don't optimise this until profiling demands it.

---

## What goes where (file layout)

```
packages/rendering/src/runtime/derivedUniforms/
  index.ts             ← public exports
  rule.ts              ← `DerivedRule` type + `derivedUniform(...)` marker
  registry.ts          ← `DerivedUniformRegistry`, IR-hash dedup
  codegen.ts           ← `emitKernel(registry)`, loaders, storers
  records.ts           ← `RecordsBuffer` (existing, minor cleanup)
  slots.ts             ← `ConstituentSlots` (existing, df32 trafo storage)
  dispatch.ts          ← `DerivedUniformsDispatcher`, recompile-on-version
  sceneIntegration.ts  ← `DerivedUniformsScene`, `registerRoDerivations`
                          (existing surface, swap RecipeId for rule lookup)
```

The current `recipes.ts` and the hardcoded `uberKernel.wgsl.ts` go away.

---

## Migration plan

1. Define `DerivedRule` + `derivedUniform` (stub IR — accept hand-built fragments first; defer the inline-marker plugin work).
2. Build `DerivedUniformRegistry` with hash-based dedup.
3. Write `codegen.ts` with loader/storer helpers for the type combos the standard rules need (mat4 in, mat4 out; mat4 in, mat3 out).
4. Replace `dispatch.ts`'s static-pipeline construction with recompile-on-version.
5. Update `sceneIntegration.registerRoDerivations` to take `HashMap<string, DerivedRule>` instead of `requiredNames: string[]`.
6. In `heapScene.addDraw`, instead of looking up names in a hardcoded set, iterate `ro.derivedUniforms`.
7. Port the existing 13 hardcoded recipes to standalone `DerivedRule` constants exported alongside `derivedUniform` for back-compat (so existing demos keep working). Eventually demos define their own.
8. Wire wombat-shader-vite plugin to recognize `derivedUniform(...)` markers and emit IR (last; can ship without if the user accepts hand-built fragments for v0).

Each step is independently testable. Steps 1–6 land §7 v2 with the standard recipes ported as DerivedRules. Step 7 lets users write their own. Step 8 is the polish.
