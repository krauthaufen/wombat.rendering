// runFrame — adaptivity-driven render loop. Mirrors Aardvark.Rendering's
// approach: an AVal wraps the user's frame callback, the frame's
// dependency set comes from whatever the callback reads via the
// adaptive `token`, and a marking callback on the wrapper schedules
// the next `requestAnimationFrame`. Idle scenes (no avals fired
// since last render) consume zero CPU per rAF.
//
// The standard time-ticking pattern (which the wombat.dom
// `<RenderControl>` follows) is:
//   1. The user's frame callback calls `task.run(token)` which
//      reads its inputs (uniforms, alists, …) via the token.
//      Those reads register as inputs to the wrapping AVal.
//   2. After `task.run`, the callback ticks a global `time` cval
//      inside a `transact(...)`.
//   3. If anything in the scene (e.g. an `OrbitController`'s
//      damping aval, a custom animation aval) reads `time`, the
//      mark cascades into the wrapper → marking callback fires →
//      next rAF scheduled. If *nothing* depends on time, the mark
//      dies in midair and the loop sleeps.
//   4. Canvas resize / explicit cval edits / scene-graph deltas
//      mark their respective avals → cascade → wake-up.
//
// Returns a `stop()` to cancel the loop and clear the subscription.

import type { aval, AdaptiveToken, IDisposable } from "@aardworx/wombat.adaptive";
import { AVal, transact } from "@aardworx/wombat.adaptive";
import type { CanvasAttachment } from "./canvas.js";

export interface RunFrameOptions {
  /** Stop after N frames. Useful for tests + finite animations. */
  readonly maxFrames?: number;
  /**
   * Hook called after each frame's eval completes — typically used
   * to tick a `time` cval (so any aval that depends on time gets
   * marked AFTER the frame, waking the next rAF). Fired in a fresh
   * transaction. Returning without ticking (`onAfterFrame: () => {}`,
   * or omitting) means the loop won't advance unless something else
   * marks an aval that the frame reads. That's the correct idle
   * behavior.
   */
  readonly onAfterFrame?: () => void;
  /**
   * Frame pacer. When provided, runFrame awaits this promise after
   * each frame's eval and BEFORE invoking `onAfterFrame` / scheduling
   * the next rAF. Caps the JS encode rate to whatever the pacer
   * resolves at — typically `() => device.queue.onSubmittedWorkDone()`
   * to keep the GPU queue at depth 1 instead of letting submit-
   * returns-immediately balloon the queue under load.
   *
   * Without a pacer, JS happily encodes 60 frames/s into a GPU that's
   * actually running at 10 fps — the compositor then drops frames
   * silently and the perceived rate has nothing to do with rAF rate.
   * Always supply a pacer for continuous-redraw scenes.
   */
  readonly pacer?: () => Promise<void>;
}

export type FrameCallback = (token: AdaptiveToken, info: FrameInfo) => void;

export interface FrameInfo {
  readonly frame: number;
  readonly timestampMs: number;
  readonly deltaMs: number;
}

export function runFrame(
  attachment: CanvasAttachment,
  frame: FrameCallback,
  opts: RunFrameOptions = {},
): { stop(): void } {
  let stopped = false;
  let frameNo = 0;
  let lastTime = performance.now();
  let rafId = 0;
  let pending = false;

  // Wrap the frame callback in an AVal so the adaptive system tracks
  // its dependency set. Each `force()` re-reads the inputs and runs
  // the frame; subsequent input marks fire the marking callback.
  const renderAval: aval<number> = AVal.custom((token) => {
    const now = performance.now();
    const delta = now - lastTime;
    lastTime = now;
    attachment.markFrame();
    try {
      frame(token, { frame: frameNo, timestampMs: now, deltaMs: delta });
    } catch (e) {
      stopped = true;
      console.error("runFrame: error in frame callback", e);
      throw e;
    }
    frameNo++;
    return frameNo;
  });

  const finalizeFrame = (): void => {
    if (stopped) return;
    if (opts.onAfterFrame !== undefined) {
      transact(opts.onAfterFrame);
    }
    if (opts.maxFrames !== undefined && frameNo >= opts.maxFrames) {
      stopped = true;
      sub.dispose();
    }
  };

  const tick = (): void => {
    if (stopped) return;
    pending = false;
    // The single force-on-render-path. Pulls `renderAval`, which
    // re-evaluates the user's frame callback (making it up-to-date
    // for the next mark). Adaptivity-driven equivalent of "run one
    // frame on this rAF."
    renderAval.force(/* allow-force */);
    // Post-frame hook in a fresh transaction. Marks fired here
    // propagate AFTER the eval, so they actually reach
    // `renderAval`'s marking callback and wake the next rAF.
    // Inside the eval the same mark would race the
    // `outOfDate=false` reset and animation would stall after one
    // frame.
    if (opts.pacer !== undefined) {
      opts.pacer().then(finalizeFrame, (e) => {
        stopped = true;
        console.error("runFrame: pacer rejected", e);
      });
    } else {
      finalizeFrame();
    }
  };

  const sub: IDisposable = (renderAval as unknown as {
    addMarkingCallback(cb: () => void): IDisposable;
  }).addMarkingCallback(() => {
    if (pending || stopped) return;
    pending = true;
    rafId = requestAnimationFrame(tick);
  });

  // Initial render. Schedules subsequent frames only when adaptive
  // marks reach `renderAval` (via the marking callback above).
  pending = true;
  rafId = requestAnimationFrame(tick);

  return {
    stop() {
      stopped = true;
      cancelAnimationFrame(rafId);
      sub.dispose();
    },
  };
}
