// Browser-mode vitest config. Runs tests in headless Chromium via
// Playwright with real WebGPU enabled.
//
// Linux + NVIDIA notes:
//   - Chromium needs the WebGPU + Vulkan flags, plus the unsafe-webgpu
//     flag for headless mode. We feed them via `launch.args`.
//   - On NVIDIA the Vulkan ICD must be discoverable; the standard
//     /usr/share/vulkan/icd.d/nvidia_icd.json setup works.

import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { defineConfig } from "vitest/config";
import { boperators } from "@boperators/plugin-vite";
import { wombatShader } from "@aardworx/wombat.shader-vite";

const here = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  plugins: [
    boperators(),
    wombatShader({
      rootDir: here,
      tsconfigPath: resolve(here, "tests/tsconfig.json"),
    }),
  ],
  optimizeDeps: { include: ["typescript"] },
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
