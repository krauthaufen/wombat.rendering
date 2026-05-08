// Browser-mode vitest config. Runs tests in headless Chromium via
// Playwright with real WebGPU enabled.
//
// Linux + NVIDIA notes:
//   - Chromium needs the WebGPU + Vulkan flags, plus the unsafe-webgpu
//     flag for headless mode. We feed them via `launch.args`.
//   - On NVIDIA the Vulkan ICD must be discoverable; the standard
//     /usr/share/vulkan/icd.d/nvidia_icd.json setup works.

import { defineConfig } from "vitest/config";

export default defineConfig({
  define: { global: "globalThis" },
  test: {
    include: ["tests-browser/**/*.test.ts"],
    browser: {
      enabled: true,
      provider: "playwright",
      name: "chromium",
      headless: true,
      providerOptions: {
        launch: {
          // Playwright's bundled chromium_headless_shell falls back
          // to SwiftShader for WebGPU. Use the system Chromium
          // (full build with the compositor) so adapter selection
          // picks the real GPU via Vulkan.
          executablePath: "/usr/bin/chromium",
          args: [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan,UseSkiaRenderer",
            "--use-vulkan=native",
            "--ignore-gpu-blocklist",
            "--enable-webgpu-developer-features",
            "--use-angle=vulkan",
          ],
        },
      },
    },
  },
});
