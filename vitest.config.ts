import { defineConfig } from "vitest/config";

// Two test environments. Node tests use the mock GPUDevice; browser
// tests run headless in Chromium with a real WebGPU device. Run them
// separately:
//   npm run test          # node only (fast, default for CI)
//   npm run test:browser  # browser only (real GPU)
//   npm run test:all      # both
//
// vitest's project mode lets us share most config but switch env per
// project. The browser project requires `@vitest/browser` + Playwright
// (already wired into devDependencies + `npx playwright install`).

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    exclude: ["tests-browser/**"],
  },
});
