// RenderTask — the compiled, runnable form of an alist<Command>.
// The runtime returns one of these from `compile`; the user calls
// `run(framebuffer, token)` to encode + submit a frame against the
// supplied framebuffer instance. The framebuffer's signature must
// match the one the task was compiled against.

import type { AdaptiveToken } from "@aardworx/wombat.adaptive";
import type { FramebufferSignature } from "./framebufferSignature.js";
import type { IFramebuffer } from "./framebuffer.js";

export interface IRenderTask {
  /**
   * The signature this task was compiled against. All pipelines
   * baked in were specialised against this signature; framebuffers
   * passed to `run`/`encode` must match.
   */
  readonly signature: FramebufferSignature;

  /**
   * Encode all commands into the GPU queue using the current
   * adaptive state read via `token`. Returns once submission has
   * been queued (does not wait for GPU completion).
   */
  run(framebuffer: IFramebuffer, token: AdaptiveToken): void;

  /** Tear down resources owned by the task. Idempotent. */
  dispose(): void;
}
