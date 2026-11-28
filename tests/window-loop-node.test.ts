// window/loop in Node: importing the module must not keep the process alive, and the uncapped
// loop still runs its frames. A MessagePort with an onmessage handler is an active handle, so a
// channel made at module load kept any Node script that only imports the package from exiting.

import { describe, expect, it } from "vitest";
import { cval } from "@aardworx/wombat.adaptive";
import type { CanvasAttachment } from "../packages/rendering/src/window/canvas.js";

const messagePorts = (): number =>
  process.getActiveResourcesInfo().filter((r) => r === "MessagePort").length;

describe("window/loop in Node", () => {
  it("importing it opens no MessagePort", async () => {
    const before = messagePorts();
    await import("../packages/rendering/src/window/loop.js");
    expect(messagePorts()).toBe(before);
  });

  it("an uncapped loop runs its frames through a channel made on first use", async () => {
    const before = messagePorts();
    const g = globalThis as Record<string, unknown>;
    g.__wombatUncappedRenderLoop = true;
    try {
      const { runFrame } = await import("../packages/rendering/src/window/loop.js");
      const time = cval(0);
      const frames: number[] = [];
      const attachment = { markFrame() {} } as unknown as CanvasAttachment;
      await new Promise<void>((resolve) => {
        runFrame(attachment, (token, info) => {
          time.getValue(token);
          frames.push(info.frame);
          if (frames.length === 3) resolve();
        }, { maxFrames: 3, onAfterFrame: () => { time.value = time.value + 1; } });
      });
      expect(frames).toEqual([0, 1, 2]);
      expect(messagePorts()).toBe(before + 1);
    } finally {
      delete g.__wombatUncappedRenderLoop;
      delete g.__wombatFrameCount;
    }
  });
});
