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
  test: {
    include: ["tests-browser/**/*.test.ts"],
    browser: {
      enabled: true,
      provider: "playwright",
      name: "chromium",
      headless: true,
      providerOptions: {
        launch: {
          args: [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan,UseSkiaRenderer",
            "--use-vulkan=native",
            "--ignore-gpu-blocklist",
            "--enable-webgpu-developer-features",
          ],
        },
      },
    },
  },
});
