// runFrame — a frame waits for the pacer of the frame before it, whoever asks
// for it. A mark from an input (a camera drag) used to schedule the next frame
// at once, so on a slow GPU frames were encoded faster than drawn and the
// picture arrived in bursts.

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { cval, transact } from "@aardworx/wombat.adaptive";
import { runFrame } from "@aardworx/wombat.rendering/window";

const turn = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 5));

describe("runFrame pacing", () => {
  const g = globalThis as unknown as { requestAnimationFrame?: unknown; cancelAnimationFrame?: unknown };
  const saved = { raf: g.requestAnimationFrame, caf: g.cancelAnimationFrame };
  beforeEach(() => {
    g.requestAnimationFrame = (cb: () => void) => setTimeout(cb, 0);
    g.cancelAnimationFrame = (h: number) => clearTimeout(h);
  });
  afterEach(() => { g.requestAnimationFrame = saved.raf; g.cancelAnimationFrame = saved.caf; });

  it("draws an input's change only once the frame before is done on the GPU", async () => {
    const input = cval(0);
    const drawn: number[] = [];
    const release: Array<() => void> = [];
    const loop = runFrame(
      { markFrame() {} } as never,
      (token) => { drawn.push(input.getValue(token)); },
      { pacer: () => new Promise<void>((resolve) => release.push(resolve)) },
    );
    await turn();
    expect(drawn).toEqual([0]);
    // the GPU is still busy with frame 0: changes are not encoded behind it
    transact(() => { input.value = 1; });
    await turn();
    transact(() => { input.value = 2; });
    await turn();
    expect(drawn).toEqual([0]);
    // done: ONE frame, with the latest value
    release[0]!();
    await turn();
    expect(drawn).toEqual([0, 2]);
    // nothing marked since: no further frame
    release[1]!();
    await turn();
    expect(drawn).toEqual([0, 2]);
    // idle again, a change draws at once
    transact(() => { input.value = 3; });
    await turn();
    expect(drawn).toEqual([0, 2, 3]);
    loop.stop();
  });

  it("without a pacer every change draws on the next frame", async () => {
    const input = cval(0);
    const drawn: number[] = [];
    const loop = runFrame({ markFrame() {} } as never, (token) => { drawn.push(input.getValue(token)); });
    await turn();
    transact(() => { input.value = 1; });
    await turn();
    expect(drawn).toEqual([0, 1]);
    loop.stop();
  });
});
