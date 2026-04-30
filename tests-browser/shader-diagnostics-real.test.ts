// Shader-compile-error reporting on real GPU. Inject deliberately
// invalid WGSL via createShaderModule + installShaderDiagnostics
// and confirm we capture the compilation messages.

import { describe, expect, it } from "vitest";
import { installShaderDiagnostics } from "@aardworx/wombat.rendering-resources";
import { requestRealDevice } from "./_realGpu.js";

describe("shader diagnostics — real GPU", () => {
  it("captures and forwards getCompilationInfo() errors", async () => {
    const device = await requestRealDevice();
    try {
      const errors: string[] = [];
      const warns: string[] = [];
      const logger = {
        error: (...a: unknown[]) => errors.push(a.map(String).join(" ")),
        warn:  (...a: unknown[]) => warns.push(a.map(String).join(" ")),
      };
      const bad = `@vertex fn main() -> @builtin(position) vec4<f32> { not_a_real_function(); return vec4<f32>(); }`;
      const m = device.createShaderModule({ code: bad });
      installShaderDiagnostics(m, bad, { label: "bad", logger });

      // getCompilationInfo is async — wait for it to settle.
      await m.getCompilationInfo();
      // Give the .then() in installShaderDiagnostics a microtask to flush.
      await new Promise(r => setTimeout(r, 0));

      expect(errors.length).toBeGreaterThan(0);
      expect(errors.join("\n")).toMatch(/bad/);
    } finally {
      device.destroy();
    }
  });
});
