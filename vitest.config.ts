import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { defineConfig } from "vitest/config";
import { boperators } from "@boperators/plugin-vite";
import { wombatShader } from "@aardworx/wombat.shader-vite";

// Two test environments. Node tests use the mock GPUDevice; browser
// tests run headless in Chromium with a real WebGPU device. Run them
// separately:
//   npm run test          # node only (fast, default for CI)
//   npm run test:browser  # browser only (real GPU)
//   npm run test:all      # both
//
// Both configs load boperators + wombat.shader-vite so test fixtures
// can author effects via inline `vertex(...) / fragment(...) /
// effect(...)` markers — same path as production code in heap-demo
// and wombat.dom. The plugins read TypeScript types via the local
// `tests/tsconfig.json` (see there for the @boperators/plugin-tsc
// program transform).

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
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    exclude: ["tests-browser/**"],
  },
});
