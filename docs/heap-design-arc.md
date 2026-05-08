# How we got here

A narrative of the design path that produced wombat's heap renderer.
The end state — an ECS embedded in a reactive scene graph, evaluated
on GPU through a shader DSL — sounds bigger than it is. The valuable
thing isn't the destination; it's that every step is *forced* by the
constraint left over from the previous one. Once you walk the chain,
the architecture stops looking inventive and starts looking
inevitable.

Companion to `heap-future-work.md` — that doc is the punch-list of
threads worth pulling. This one is the rationale for *why those
threads exist at all.*

## Step 0 — the starting constraints

Three constraints, none negotiable:

1. **Lots of small things.** Scientific viz, design review, GIS, CAD
   viewers, parametric modeling — the canonical workloads put
   thousands to tens of thousands of small ROs in front of the user.
   Markers, gizmos, labels, vehicle props, GIS feature symbols.
   Mostly homogeneous, occasionally heterogeneous, usually
   stable-with-sparse-mutation.
2. **Reactive scene composition.** Wombat is a port of Aardvark's
   pull-based functional reactivity. A scene is a tree of avals;
   user interaction marks some, the renderer pulls what changed.
   Imperative scene maintenance is not on the table — the API is
   declarative all the way down.
3. **Arbitrary custom shaders.** Researchers write Loop-Blinn
   curve renderers, custom volumetric kernels, novel projection
   variants. The shader system is a programming language, not a
   slot-configuration UI. Every RO can in principle have a
   bespoke vertex/fragment stage.

These three constraints are why standard answers don't fit. Game
engines (UE, Unity) optimize for #1 but not #2 or #3 — their scenes
are design-time-built, their shaders are slot configurations.
Scientific-viz toolkits (VTK, ParaView, deck.gl) often have #1 and
#2 but not #3. FRP libraries have #2 but no GPU story. Nobody had
all three.

So we had to build it. Each step from here is what the previous
step's leftover problem demanded.

## Step 1 — encode time scales with N

The naive renderer issues one `pass.drawIndexed` per RO. At 10K ROs
that's 10K JS-side calls every frame, plus per-RO bind-group +
pipeline switches. On WebGPU specifically, each call crosses the
JS→C++→Dawn→native bridge — visible CPU overhead even for tiny
draws. At 10K ROs we measured ~180ms encode time. Unusable.

The leftover problem: **JS-driven encode scales with RO count.**

## Step 2 — WebGPU 1.0 has no multi-draw

The native answer is `glMultiDrawElementsIndirect` (or Vulkan's
equivalent) — one CPU call dispatches N draws on the GPU side.
Aardvark uses this on native. WebGPU 1.0 doesn't have it. Won't
have it for years (we estimated 2–4 years given Apple/Mozilla pace
on these features).

Chromium has an experimental flag (`chromium-experimental-multi-
draw-indirect`), but it requires `--enable-dawn-features=
allow_unsafe_apis`. Not shippable. Not portable. We tested it; the
perf gain was measurable but small, and the architectural cost of
relying on an unsafe-flag-only feature is not.

The leftover problem: **we need MDI's behavior without MDI.**

## Step 3 — megacall

Observation: WebGPU lets us write the shader. We don't actually
need fixed-function MDI — we can do its work in the vertex shader.
Replace N drawIndexed calls with **one** `pass.draw(totalEmit)`
where `totalEmit` is the sum of all draws' index counts. The
vertex shader gets a linear `vertex_index` from 0 to totalEmit-1,
and *figures out* which logical draw it belongs to via a binary
search over a per-bucket "drawTable" of `(firstEmit, drawIdx,
indexStart, indexCount)` records.

```wgsl
@vertex fn vs(@builtin(vertex_index) emitIdx: u32) -> VsOut {
  // binary search drawTable for the slot whose firstEmit <= emitIdx
  var lo = 0u; var hi = numRecords - 1u;
  loop { ... }
  let drawIdx = drawTable[lo * 4u + 1u];
  let vid = indexStorage[drawTable[lo * 4u + 2u] + (emitIdx - drawTable[lo * 4u])];
  // ... rest of VS proceeds normally with drawIdx + vid ...
}
```

One drawcall per bucket. Encode time drops from O(N) to O(buckets).
At 10K ROs across 3 buckets, encode goes sub-millisecond.

The leftover problem: **drawTable's `firstEmit` field is a prefix
sum of indexCount. Who computes it?**

## Step 4 — CPU prefix sum is fine, until it isn't

First implementation: CPU maintains the drawTable in a Uint32Array.
Each `addDraw` appends a record with the running sum. Each frame,
writeBuffer the dirty range to the GPU.

Works. But: when the user mutates a RO's geometry (rare but real),
or when records are added/removed in bulk, the CPU has to re-sum
and re-upload. For static scenes with sparse churn, fine. For
"camera grabs the scene and orbits" — also fine, because firstEmit
doesn't change with camera. Camera-only changes don't touch the
drawTable.

So far so good. But:

The leftover problem: **What about the *uniforms* that DO change
on camera move? At 10K ROs, the ModelView matrix per RO needs
recomputation when the camera moves. CPU work scales with N
again.**

This is the classic geodetic-precision case. ModelView = View ×
Model, both potentially huge in absolute coordinates, computed in
double precision so the camera-relative offsets cancel cleanly.
50K ROs × per-frame matrix multiply + writeBuffer = back to
unusable.

## Step 5 — GPU compute is already running, use it for more

The prefix-sum step had to happen on GPU eventually anyway (when
add/remove fires we want to avoid CPU readback latency for the
indirect-draw count). So we put it on GPU: a three-pass Blelloch
scan that consumes drawTable[*].indexCount and writes the indirect
args buffer directly. CPU never reads totalEmit; `pass.drawIndirect`
takes the count from the GPU.

Now we have a compute pass running on dirty buckets. Once that
infrastructure exists, the question becomes: **what else can we
compute there?**

The N-matrix-update problem becomes the obvious next inhabitant.
Per RO, store ModelTrafo as df32 (two-float double-precision,
compensated arithmetic). Per scene, store ViewTrafo as df32. A
compute kernel reads both, multiplies in df32, truncates to f32,
writes ModelView for the shader to use.

Camera move:
- One cval mark (`viewTrafo`).
- Adaptive system propagates ONE mark to a per-class dirty flag.
- Compute pass dispatches one thread per RO, computes ModelView,
  writes to arena.
- Render pass reads the arena slot like any other uniform.

CPU work: setting a boolean. GPU work: 50K matrix multiplies in
parallel, ~1ms.

The leftover problem: **handwriting compute kernels is bespoke
work for every new derivation. The user writing a shader can't
also be expected to write the compute kernel that supplies its
inputs.**

## Step 6 — let the user declare derivations in the shader DSL

We already have wombat.shader: a TypeScript-shaped DSL for vertex
and fragment stages, compiled to WGSL via an IR pipeline. Extend
it. Same DSL, new context:

```ts
deriveUniform("ModelView", (u) => u.ViewTrafo.mul(u.ModelTrafo));
```

The compiler:
- Detects this is a derivation, not a stage.
- Statically analyzes the lambda's reads (`u.ViewTrafo`,
  `u.ModelTrafo`) → emits the dependency set.
- Emits a `@compute @workgroup_size(...)` entry.
- Plumbs it into the runtime's per-class dirty machinery.

The user never wrote a compute pass. They wrote a function. The
runtime figured out everything else.

Scene-graph inheritance kicks in:
- A `Camera` SG node publishes `ViewTrafo` to its descendants.
- A `Geodetic` modifier publishes `precision: df32` — the compiler
  picks compensated-arithmetic codegen automatically downstream.
- A `Trafo` SG node publishes `ModelTrafo`, composing with parent.
- Leaves get the derivation resolved through the inheritance chain.

The user describes the scene; the runtime makes it so.

The leftover problem: **once you can declare derivations like this,
you realize there's nothing uniform-specific about the mechanism.
It works for any per-entity computation.**

## Step 7 — uniforms are just one kind of component

The pattern is general. Anything that's "computed per RO from other
per-RO state plus some scene-global inputs" fits the same shape:

- **Bounding box** from positions + ModelTrafo.
- **Sort key** from camera position + ModelTrafo.
- **LOD level** from screen-space bbox.
- **Animation pose** from skeleton + time.
- **Hover state** from cursor ray + bbox.
- **Custom user data** of any shape.

Each is a derivation. Each gets a dirty list. Each compiles to a
compute kernel. Each composes through the SG.

```ts
deriveComponent("BBox",     (e) => bboxOf(e.Positions, e.ModelTrafo));
deriveComponent("SortKey",  (e) => (e.ViewTrafo.mul(e.ModelTrafo)).origin.z);
deriveComponent("Hovered",  (e) => rayHits(camera.cursorRay, e.BBox));
```

SG modifiers become *system* declarations:

```ts
WithSorted(by = "SortKey", order = "back-to-front") { children }
WithFrustumCulled(camera) { children }
WithHoverable(onChange = handler) { children }
```

We have:

- **Entities**: RO slots, indexed by `drawIdx`.
- **Components**: arena allocations indexed by `drawIdx`, each typed.
- **Systems**: compute kernels, dependency-tracked, sparse-dispatched.
- **Schema**: derived from the SG's modifier composition.
- **Reactivity**: marks propagate through the dirty-list mesh.

That's an ECS. Built bottom-up, in a place we didn't expect to
arrive.

## What it means

The two programming models people argue about — scene graph
("intuitive authoring") vs. ECS ("efficient evaluation") — were
never opposed; they're answering different questions. SG describes
*what the user wants to express*. ECS describes *how the runtime
should store and process it*. With wombat.shader as the DSL bridge,
the same description compiles into both:

- The user reads/writes the SG (the authoring model).
- The runtime materializes it as components + systems (the
  evaluation model).
- The shader DSL describes both stages and derivations in one
  language.

This is the architecture. It came out of trying to draw 10K boxes
on iPhone Safari without exceeding 16ms per frame.

## Why nobody else built this

Each step in isolation is uninteresting. The combination only
emerges if you start from a specific set of constraints:

- **Game-engine teams** don't follow this chain. They don't start
  from "reactive scene composition" because their scenes are
  design-time. They don't start from "arbitrary custom shaders"
  because their shaders are slot configurations. So they never
  reach the point where a shader DSL can carry derivation logic.
- **FRP libraries** don't follow it because they don't have a GPU
  compute pass. Step 5 is unavailable to them. Their reactive
  graphs evaluate on CPU; "50K matrix updates per camera move" is
  fundamentally a non-starter.
- **Scientific-viz toolkits** don't follow it because they don't
  have arbitrary shader code. The derivation has nowhere to live
  if every render path is built-in.

The chain requires all three constraints to be live at once. That's
rare. It's also exactly the wombat / Aardvark domain. The
architecture isn't novel because we were clever; it's novel because
nobody else was working on the right problem.

## What's actually shipped

To be honest about how far we are along the chain:

- **Steps 1–5**: shipped. Megacall + GPU prefix-sum + drawIndirect
  + GPU pacing + identity-driven sharing. 50K objects on iPhone at
  44 fps, GPU-bound, 0.3ms encode time.
- **Steps 6–7**: design only (this doc + heap-future-work.md §7,
  §7a). Manual derivation kernels are next; the DSL extension is
  after that; the full ECS framing falls out of those.

The implementation backbone is in place. The user-facing
declarations require wombat.shader to grow a new emit path. That's
a real task but not a research task — it's a port of patterns
already proven elsewhere (UE's material derivation, Aardvark's
existing derived-uniform system) into the new context.

## Honest trade-offs

The architecture is not free:

- **Bulk synchronous mutations are slower than ECS-array updates.**
  Setting 20K transforms via cval writes is real work — likely
  5–20ms on desktop. ECS would do it in well under a millisecond
  via array memcpy. For continuous bulk-mutation workloads
  (real-time particle sims, N-body), this is the wrong tool.
- **The aval bookkeeping has fixed overhead per dependency.** Mark
  propagation is cheap but not free. Static or sparse-mutation
  scenes pay almost nothing; mutation-heavy scenes pay
  proportionally to the mutation rate.
- **Compile times scale with shader complexity.** The uber-shader
  family approach trades pipeline-cache count for shader-compile
  time. For interactive editing of shader code, this is fine; for
  cold-start of complex apps, it's a real cost.

Where this is the *right* architecture:

> Modelling tools, design review, scientific exploration, GIS,
> CAD viewers, IFC/BIM, parametric modeling, photogrammetry
> annotation, simulation *result* visualization (not running the
> simulation).

These workloads share a structure: lots of stable content,
*interesting* mutations are local and user-driven, bulk
re-evaluations are rare-ish events. That's the sweet spot. The
adaptive overhead lives below the cost of a single user
interaction; the GPU-driven fast path makes the static side cheap.

## Closing

The valuable thing in this whole document isn't the diagram of the
final system. It's that you can put yourself at step 0, hold the
three starting constraints in your head, and watch each step force
the next. By step 7 you've built an ECS without setting out to,
and the SG/ECS dichotomy that game-engine designers have wrestled
with for a decade quietly resolves itself.

That's the thing worth showing to people: not the destination, the
path. Once you've walked it, the architecture stops being
remarkable. It just becomes obvious.
