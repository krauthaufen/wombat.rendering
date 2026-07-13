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

// ─── Benchmark toggles ─────────────────────────────────────────────────
//
// `globalThis.__wombatUncappedRenderLoop = true` (set BEFORE the loop
// starts) schedules frames via setTimeout(0) instead of
// requestAnimationFrame — no vsync clamp; with the standard
// `onSubmittedWorkDone` pacer the loop then runs at true GPU
// completion rate. The canvas still presents whatever frame is
// latest. When uncapped, `globalThis.__wombatFrameCount` increments
// AFTER each frame's pacer resolves (i.e. GPU work done), so a
// harness can serialize edit → frame-complete → next edit and read
// honest end-to-end frame costs.
// Read at runFrame() start (NOT module load) — consumers set the flag
// from app code that runs after this module is imported.
function isUncapped(): boolean {
  return (globalThis as Record<string, unknown> & typeof globalThis)
    .__wombatUncappedRenderLoop === true;
}

function bumpFrameCount(): void {
  const g = globalThis as Record<string, unknown> & typeof globalThis;
  g.__wombatFrameCount = ((g.__wombatFrameCount as number | undefined) ?? 0) + 1;
  // one-shot completion hook: lets a harness await frame completion
  // without poll-spinning a timer/MessageChannel loop (which competes
  // with the render loop's own scheduling).
  const cb = g.__wombatOnFrame as (() => void) | undefined;
  if (cb !== undefined) {
    g.__wombatOnFrame = undefined;
    cb();
  }
}

// setTimeout(0) is useless for an uncapped loop: Chrome clamps nested
// timers to a 4 ms minimum after depth 5. MessageChannel.postMessage
// is the unthrottled macrotask primitive.
const immediateCbs = new Map<number, () => void>();
let immediateNext = 1;
const immediateChannel: MessageChannel | null =
  typeof MessageChannel !== "undefined" ? new MessageChannel() : null;
if (immediateChannel !== null) {
  immediateChannel.port1.onmessage = (e: MessageEvent): void => {
    const id = e.data as number;
    const cb = immediateCbs.get(id);
    immediateCbs.delete(id);
    cb?.();
  };
}
function setImmediate2(cb: () => void): number {
  if (immediateChannel === null) return setTimeout(cb, 0) as unknown as number;
  const id = immediateNext++;
  immediateCbs.set(id, cb);
  immediateChannel.port2.postMessage(id);
  return id;
}
function clearImmediate2(id: number): void {
  if (immediateChannel === null) { clearTimeout(id); return; }
  immediateCbs.delete(id);
}

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
  let consecutiveErrors = 0;
  const MAX_CONSECUTIVE_ERRORS = 30;
  const UNCAPPED = isUncapped();

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
      consecutiveErrors = 0;
    } catch (e) {
      // A throwing frame used to kill the loop for good. That is far too harsh
      // for a transient fault (a resource disposed mid-frame, a one-off device
      // hiccup): the page then looks alive — DOM, input and workers all keep
      // running — while nothing renders again and any animation driven from the
      // frame callback freezes. Retry instead, and only give up if the fault
      // repeats every frame (a genuinely broken scene, which would otherwise
      // spin the error path forever).
      consecutiveErrors++;
      console.error(
        `runFrame: error in frame callback (${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS})`, e,
      );
      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        stopped = true;
        console.error("runFrame: giving up after repeated frame errors — render loop stopped");
        throw e;
      }
      // Keep the loop breathing: schedule the next frame ourselves, since a
      // frame that died early may not have read the inputs it would be marked
      // by. Swallow the error rather than rethrowing — an exception escaping
      // the aval eval leaves the wrapper's dirty state ambiguous, and we have
      // already reported it.
      queueMicrotask(() => {
        if (stopped || pending) return;
        pending = true;
        rafId = schedule(tick);
      });
    }
    frameNo++;
    return frameNo;
  });

  const finalizeFrame = (): void => {
    if (stopped) return;
    // pacer (GPU completion) has resolved by now — this frame is DONE
    if (UNCAPPED) bumpFrameCount();
    if (opts.onAfterFrame !== undefined) {
      transact(opts.onAfterFrame);
    }
    if (opts.maxFrames !== undefined && frameNo >= opts.maxFrames) {
      stopped = true;
      sub.dispose();
    }
  };

  const schedule = (cb: () => void): number =>
    UNCAPPED ? setImmediate2(cb) : requestAnimationFrame(cb);
  const cancel = (h: number): void => {
    if (UNCAPPED) clearImmediate2(h);
    else cancelAnimationFrame(h);
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
        // A rejected pacer (GPU hiccup) must not brick the loop either — the
        // frame is done as far as we're concerned; carry on.
        console.error("runFrame: pacer rejected", e);
        finalizeFrame();
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
    rafId = schedule(tick);
  });

  // Initial render. Schedules subsequent frames only when adaptive
  // marks reach `renderAval` (via the marking callback above).
  pending = true;
  rafId = schedule(tick);

  return {
    stop() {
      stopped = true;
      cancel(rafId);
      sub.dispose();
    },
  };
}
