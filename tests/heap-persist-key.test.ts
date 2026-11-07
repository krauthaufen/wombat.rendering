// The heap-shader localStorage key carries the wombat.shader version: a
// compiler change (fusion, linking, emit) must re-key the cache, not serve
// WGSL the previous compiler produced.
import { describe, expect, it } from "vitest";
import * as wombatShader from "@aardworx/wombat.shader";
import { persistKey, persistVersionNow, shaderVersionNow } from "../packages/rendering/src/runtime/heapEffect.js";

describe("heap persist key", () => {
  it("names the wombat.shader version", () => {
    const v = (wombatShader as { VERSION?: string }).VERSION ?? "unknown";
    expect(shaderVersionNow()).toBe(v);
    expect(persistVersionNow()).toContain(`-ws${v}`);
    expect(persistKey(persistVersionNow(), "ir", "abc")).toContain(`-ws${v}`);
  });

  it("changes when the compiler version changes", () => {
    const a = persistKey(`h15-aps4096-ws0.6.8`, "ir", "abc");
    const b = persistKey(`h15-aps4096-ws0.6.9`, "ir", "abc");
    expect(a).not.toBe(b);
  });
});
