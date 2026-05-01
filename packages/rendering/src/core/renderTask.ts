// RenderTask — the compiled, runnable form of an alist<Command>.
// The runtime returns one of these from `compile`; the user calls
// `run(token)` to encode + submit a frame.

import type { AdaptiveToken } from "@aardworx/wombat.adaptive";

export interface IRenderTask {
  /**
   * Encode all commands into the GPU queue using the current
   * adaptive state read via `token`. Returns once submission has
   * been queued (does not wait for GPU completion).
   */
  run(token: AdaptiveToken): void;

  /** Tear down resources owned by the task. Idempotent. */
  dispose(): void;
}
