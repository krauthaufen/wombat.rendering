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

  /**
   * Total GPU buckets the heap path emits across this task's
   * compiled scenes. With §6 family-merge in v1 this collapses to
   * one per pipelineState per Render command. Useful for status
   * text / dev overlays.
   */
  heapBucketCount(): number;

  /** Per-frame breakdown of §7 derived-uniforms work (CPU, last frame). */
  heapDerivedTimings(): {
    pullMs: number; uploadMs: number; encodeMs: number; records: number;
  };

  /** Diagnostic: walks heap drawHeaders, drawTables, prefix sums,
   *  attribute alloc headers; checks refs/indices/instance counts and
   *  data finiteness. Useful for hunting OOB-corruption bugs. */
  validateHeap(): Promise<{
    arenaBytes: number; issues: string[];
    okRefs: number; badRefs: number;
    drawTableRows: number; drawTableErrs: number; prefixSumErrs: number;
    attrAllocsChecked: number; attrAllocsBad: number;
    tilesChecked: number; tilesBad: number;
    vidChecks: number; vidBad: number;
    indicesHash: string;
  }>;
  /** CPU draw simulator. Walks N sampled emits, runs the binary-search
   *  + index/instance recovery, and verifies every storage read lands
   *  inside the bound buffer. */
  simulateDraws(samples?: number): Promise<{
    emitsChecked: number; oob: number; issues: string[];
  }>;

  /** Tear down resources owned by the task. Idempotent. */
  dispose(): void;
}
