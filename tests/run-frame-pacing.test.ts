// runFrame — at most two frames are in flight: a frame goes on once the pacer
// of the frame BEFORE it has resolved, whoever asks for the next one (the
// animation tick or a mark from an input). Waiting for the frame just
// submitted stalled in Firefox, which delivers its completion only with the
// next submit.

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

  it("lets two frames be in flight and holds a third until the oldest is done", async () => {
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
    // frame 0 is still on the GPU: ONE more frame may be encoded behind it
    transact(() => { input.value = 1; });
    await turn();
    expect(drawn).toEqual([0, 1]);
    // two in flight: further changes wait
    transact(() => { input.value = 2; });
    await turn();
    transact(() => { input.value = 3; });
    await turn();
    expect(drawn).toEqual([0, 1]);
    // frame 0 done: ONE frame, with the latest value
    release[0]!();
    await turn();
    expect(drawn).toEqual([0, 1, 3]);
    // frames 1 and 3 in flight: still held
    transact(() => { input.value = 4; });
    await turn();
    expect(drawn).toEqual([0, 1, 3]);
    release[1]!();
    await turn();
    expect(drawn).toEqual([0, 1, 3, 4]);
    // nothing marked since: no further frame
    release[2]!(); release[3]!();
    await turn();
    expect(drawn).toEqual([0, 1, 3, 4]);
    // idle again, a change draws at once
    transact(() => { input.value = 5; });
    await turn();
    expect(drawn).toEqual([0, 1, 3, 4, 5]);
    loop.stop();
  });

  it("goes on when a frame's completion is only delivered by the next submit (Firefox)", async () => {
    // every pacer promise resolves when the NEXT frame asks for its own
    const tick = cval(0);
    let frames = 0, deliver: (() => void) | undefined;
    const loop = runFrame(
      { markFrame() {} } as never,
      (token) => { tick.getValue(token); frames++; },
      {
        pacer: () => { deliver?.(); return new Promise<void>((resolve) => { deliver = resolve; }); },
        onAfterFrame: () => { tick.value = tick.value + 1; },
      },
    );
    for (let i = 0; i < 20; i++) await turn();
    loop.stop();
    expect(frames).toBeGreaterThan(8);
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
