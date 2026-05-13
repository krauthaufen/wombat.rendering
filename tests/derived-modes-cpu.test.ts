import { describe, test, expect } from "vitest";
import { AVal, cval, transact } from "@aardworx/wombat.adaptive";
import { createFramebufferSignature } from "@aardworx/wombat.rendering/resources";
import type { PipelineState, CullMode } from "@aardworx/wombat.rendering/core";
import {
  ModeKeyTracker,
  snapshotDescriptor,
  encodeModeKey,
  DEFAULT_DESCRIPTOR,
} from "@aardworx/wombat.rendering/runtime";

function makeSig() {
  return createFramebufferSignature({
    colors: { outColor: "rgba8unorm" },
    depthStencil: { format: "depth24plus", hasDepth: true, hasStencil: false },
  });
}

describe("derivedModes/modeKeyCpu", () => {
  test("snapshotDescriptor on undefined PS uses defaults", () => {
    const sig = makeSig();
    const d = snapshotDescriptor(undefined, sig);
    expect(d.topology).toBe(DEFAULT_DESCRIPTOR.topology);
    expect(d.cullMode).toBe(DEFAULT_DESCRIPTOR.cullMode);
    expect(d.attachments.length).toBe(1);
    expect(d.attachments[0]!.writeMask).toBe(0xF);
  });

  test("tracker initial modeKey reflects current aval values", () => {
    const sig = makeSig();
    const ps: PipelineState = {
      rasterizer: {
        topology: AVal.constant<GPUPrimitiveTopology>("triangle-list"),
        cullMode: cval<CullMode>("none"),
        frontFace: AVal.constant("ccw"),
      },
    };
    const tracker = new ModeKeyTracker(ps, sig, () => {});
    expect(tracker.descriptor.cullMode).toBe("none");
    expect(tracker.modeKey).toBe(encodeModeKey(tracker.descriptor));
    tracker.dispose();
  });

  test("mutating a cval fires onDirty; recompute updates the modeKey", () => {
    const sig = makeSig();
    const cullModeC = cval<CullMode>("back");
    const ps: PipelineState = {
      rasterizer: {
        topology: AVal.constant<GPUPrimitiveTopology>("triangle-list"),
        cullMode: cullModeC,
        frontFace: AVal.constant("ccw"),
      },
    };
    let dirtyCount = 0;
    const tracker = new ModeKeyTracker(ps, sig, () => { dirtyCount++; });
    const k0 = tracker.modeKey;

    transact(() => { cullModeC.value = "front"; });
    expect(dirtyCount).toBeGreaterThan(0);

    const changed = tracker.recompute();
    expect(changed).toBe(true);
    expect(tracker.modeKey).not.toBe(k0);
    expect(tracker.descriptor.cullMode).toBe("front");
    tracker.dispose();
  });

  test("recompute returns false when no input actually changed", () => {
    const sig = makeSig();
    const cullModeC = cval<CullMode>("back");
    const ps: PipelineState = {
      rasterizer: {
        topology: AVal.constant<GPUPrimitiveTopology>("triangle-list"),
        cullMode: cullModeC,
        frontFace: AVal.constant("ccw"),
      },
    };
    const tracker = new ModeKeyTracker(ps, sig, () => {});
    expect(tracker.recompute()).toBe(false);
    // Mutate to same value -> still no key change.
    transact(() => { cullModeC.value = "back"; });
    expect(tracker.recompute()).toBe(false);
    tracker.dispose();
  });

  test("two trackers built from distinct cvals with same value produce the same modeKey", () => {
    // The exact pathology Task 1 fixes: aval identity must not affect modeKey.
    const sig = makeSig();
    const ps1: PipelineState = {
      rasterizer: {
        topology: AVal.constant<GPUPrimitiveTopology>("triangle-list"),
        cullMode: cval<CullMode>("back"),     // distinct identity
        frontFace: AVal.constant("ccw"),
      },
    };
    const ps2: PipelineState = {
      rasterizer: {
        topology: AVal.constant<GPUPrimitiveTopology>("triangle-list"),
        cullMode: cval<CullMode>("back"),     // distinct identity, same value
        frontFace: AVal.constant("ccw"),
      },
    };
    const t1 = new ModeKeyTracker(ps1, sig, () => {});
    const t2 = new ModeKeyTracker(ps2, sig, () => {});
    expect(t1.modeKey).toBe(t2.modeKey);
    t1.dispose();
    t2.dispose();
  });

  test("dispose unsubscribes; further marks don't fire onDirty", () => {
    const sig = makeSig();
    const cullModeC = cval<CullMode>("back");
    const ps: PipelineState = {
      rasterizer: {
        topology: AVal.constant<GPUPrimitiveTopology>("triangle-list"),
        cullMode: cullModeC,
        frontFace: AVal.constant("ccw"),
      },
    };
    let dirtyCount = 0;
    const tracker = new ModeKeyTracker(ps, sig, () => { dirtyCount++; });
    tracker.dispose();
    transact(() => { cullModeC.value = "front"; });
    expect(dirtyCount).toBe(0);
  });
});
