// runFrame — requestAnimationFrame-driven loop integrated with
// wombat.adaptive transactions. Each tick:
//   1. Bumps the canvas attachment's frame counter so its
//      framebuffer aval invalidates and the next read returns the
//      current swap-chain texture.
//   2. Calls the user's `frame(token)` callback. The user runs
//      their `IRenderTask` here.
//
// Returns a `stop()` to cancel the loop.

import type { AdaptiveToken } from "@aardworx/wombat.adaptive";
import { AdaptiveToken as Tok } from "@aardworx/wombat.adaptive";
import type { CanvasAttachment } from "./canvas.js";

export interface RunFrameOptions {
  /** Stop after N frames. Useful for tests + finite animations. */
  readonly maxFrames?: number;
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

  const tick = (now: number) => {
    if (stopped) return;
    const delta = now - lastTime;
    lastTime = now;
    attachment.markFrame();
    try {
      frame(Tok.top, { frame: frameNo, timestampMs: now, deltaMs: delta });
    } catch (e) {
      stopped = true;
      console.error("runFrame: error in frame callback", e);
      throw e;
    }
    frameNo++;
    if (opts.maxFrames !== undefined && frameNo >= opts.maxFrames) return;
    rafId = requestAnimationFrame(tick);
  };

  rafId = requestAnimationFrame(tick);
  return {
    stop() {
      stopped = true;
      cancelAnimationFrame(rafId);
    },
  };
}
