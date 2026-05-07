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
   * name → vertex buffer view; e.g. "position", "normal", "uv".
   *
   * The set of attribute names is fixed structurally; only the
   * individual buffer values are reactive. The shader's required-input
   * set is fixed at compile; the map must contain at least those names.
   */
  readonly vertexAttributes: HashMap<string, BufferView>;
  /** name → instance buffer view; e.g. "modelMatrix", "instanceColor". */
  readonly instanceAttributes?: HashMap<string, BufferView>;
  /** name → uniform value; runtime packs into UBO based on shader layout. */
  readonly uniforms: HashMap<string, aval<unknown>>;
  /** name → texture source (CPU image or pre-built GPUTexture). */
  readonly textures: HashMap<string, aval<ITexture>>;
  /** name → sampler source (descriptor or pre-built GPUSampler). */
  readonly samplers: HashMap<string, aval<ISampler>>;
  /** name → storage buffer source. */
  readonly storageBuffers?: HashMap<string, aval<IBuffer>>;

  /** Index buffer for indexed draws. */
  readonly indices?: BufferView;
  readonly drawCall: aval<DrawCall>;
}
