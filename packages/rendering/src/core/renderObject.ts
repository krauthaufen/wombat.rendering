// RenderObject — the user-facing primitive: an effect + a bag of
// named inputs + a draw call. Names are matched against the
// shader's ProgramInterface during preparation; extras are
// silently ignored, missing required bindings are an error.
//
// Every input is `aval<T>` over a *source* type (IBuffer, ITexture,
// ISampler, …). The runtime (resources package) wraps these into
// internal AdaptiveResources for upload + ref-counting; the user
// never instantiates an AdaptiveResource directly.

import type { HashMap, aval } from "@aardworx/wombat.adaptive";
import type { IBuffer } from "./buffer.js";
import type { BufferView } from "./bufferView.js";
import type { DrawCall } from "./drawCall.js";
import type { ISampler } from "./sampler.js";
import type { ITexture } from "./texture.js";
import type { PipelineState } from "./pipelineState.js";
import type { Effect } from "./shader.js";
import type { IAttributeProvider, IUniformProvider } from "./provider.js";

export interface RenderObject {
  /**
   * A wombat.shader `Effect` — produced by the `vertex(...) /
   * fragment(...) / effect(...)` markers (transformed at build
   * time by `@aardworx/wombat.shader-vite`) or hand-built via
   * `stage(module, ...)`. The runtime calls
   * `effect.compile({ target: "wgsl" })` (Effect's own
   * hole-value-keyed cache makes this free per frame).
   *
   * `Effect.id` (stable build-time hash) feeds the pipeline cache
   * key, which means renaming a captured value invalidates only
   * its specialization; everything else stays cached.
   *
   * For users with a precompiled `CompiledEffect` (e.g. from
   * `compileShaderSource(...)`), wrap it with the test helper
   * `fakeEffectFromCompiled` or call `stage(module)` to produce
   * a real Effect.
   */
  readonly effect: Effect;
  readonly pipelineState: PipelineState;

  /**
   * Vertex attributes, looked up by name. The binding layer pulls
   * shader-driven (one `tryGet(name)` per declared vertex input), so a
   * provider may compute views lazily — though in practice attribute
   * providers are map-backed. `undefined` from `tryGet` for a name the
   * shader requires is an error at prepare time.
   */
  readonly vertexAttributes: IAttributeProvider;
  /** Instance attributes, looked up by name (same contract). */
  readonly instanceAttributes?: IAttributeProvider;
  /**
   * Uniform values, looked up by name. The runtime packs the UBO from
   * the shader's declared layout — for each declared uniform it calls
   * `uniforms.tryGet(name)`. Names no shader reads are never pulled, so
   * a lazy provider (e.g. the Sg layer's auto-injected derived trafos)
   * never builds their aval chains.
   */
  readonly uniforms: IUniformProvider;
  /** name → texture source (CPU image or pre-built GPUTexture). */
  readonly textures: HashMap<string, aval<ITexture>>;
  /** name → sampler source (descriptor or pre-built GPUSampler). */
  readonly samplers: HashMap<string, aval<ISampler>>;
  /** name → storage buffer source. */
  readonly storageBuffers?: HashMap<string, aval<IBuffer>>;

  /** Index buffer for indexed draws. */
  readonly indices?: BufferView;
  readonly drawCall: aval<DrawCall>;
  /**
   * Per-axis derived-mode rules — `(u, declared) => modeValue`
   * closures evaluated against this RO's uniforms at scene-build
   * (and on reactive marks). See
   * `@aardworx/wombat.rendering/runtime` — `derivedMode(...)`.
   *
   * A rule overrides the corresponding `pipelineState.rasterizer.*`
   * (or `pipelineState.depth.*`) aval value. v1 evaluates rules
   * CPU-side; v2 (deferred) lowers them to a GPU compute kernel.
   * See `docs/derived-modes.md`.
   */
  readonly modeRules?: {
    readonly cull?:           import("../runtime/derivedModes/rule.js").DerivedModeRule<"cull">;
    readonly frontFace?:      import("../runtime/derivedModes/rule.js").DerivedModeRule<"frontFace">;
    readonly topology?:       import("../runtime/derivedModes/rule.js").DerivedModeRule<"topology">;
    readonly depthCompare?:   import("../runtime/derivedModes/rule.js").DerivedModeRule<"depthCompare">;
    readonly depthWrite?:     import("../runtime/derivedModes/rule.js").DerivedModeRule<"depthWrite">;
    readonly alphaToCoverage?: import("../runtime/derivedModes/rule.js").DerivedModeRule<"alphaToCoverage">;
  };
}
