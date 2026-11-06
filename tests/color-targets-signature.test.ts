// A classic-path pipeline must match the pass's attachment state: a leaf whose
// shader writes only some of the signature's attachments (a NoEvents leaf has
// no pick outputs, yet is drawn in the pass with the pick target) still gets a
// target for every attachment, the unwritten ones with writeMask 0. Before, it
// got only its own outputs, and WebGPU rejected the pipeline in the pass
// ("Attachment state … is not compatible"), invalidating the whole frame.

import { describe, test, expect } from "vitest";
import { HashMap } from "@aardworx/wombat.adaptive";
import { colorTargetsForSnap } from "../packages/rendering/src/resources/preparedRenderObject.js";

const sig = {
  colors: HashMap.empty<string, GPUTextureFormat>().add("Colors", "rgba8unorm").add("PickId", "rgba32float"),
  colorNames: ["Colors", "PickId"],
  sampleCount: 1,
} as never;

const iface = (outs: { name: string; location: number }[]) =>
  ({ fragmentOutputs: outs.map((o) => ({ ...o, wgslType: "vec4<f32>" })) }) as never;

describe("colorTargetsForSnap", () => {
  test("an attachment the shader does not write gets a target with nothing written", () => {
    const t = colorTargetsForSnap(iface([{ name: "Colors", location: 0 }]), sig, undefined);
    expect(t).toHaveLength(2);
    expect(t[0]).toEqual({ format: "rgba8unorm" });
    expect(t[1]).toEqual({ format: "rgba32float", writeMask: 0 });
  });

  test("written attachments keep their targets", () => {
    const t = colorTargetsForSnap(iface([{ name: "Colors", location: 0 }, { name: "PickId", location: 1 }]), sig, undefined);
    expect(t).toEqual([{ format: "rgba8unorm" }, { format: "rgba32float" }]);
  });
});
