// Browser-mode vitest config that runs the real-GPU tests in an already
// running Chromium reached over CDP (see tests-browser/_cdpProvider.ts)
// instead of launching a headless one. Same plugins/includes as
// vitest.browser.config.ts.
//
//   WOMBAT_CDP_URL=http://127.0.0.1:9222 npx vitest run --config vitest.browser.cdp.config.ts

import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { defineConfig, mergeConfig } from "vitest/config";
import base from "./vitest.browser.config.js";

const here = fileURLToPath(new URL(".", import.meta.url));

export default mergeConfig(base, defineConfig({
  test: {
    browser: {
      provider: resolve(here, "tests-browser/_cdpProvider.ts"),
      name: "chromium",
      headless: false,
      providerOptions: {},
    },
  },
}));
