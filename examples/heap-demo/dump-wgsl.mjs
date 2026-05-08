// Dump compiled WGSL for the demo's surface effect, so we can read
// what the DSL actually emits for the inter-stage normal path.

import { chromium } from "/home/schorsch/projects/wombat.dom/examples/hello-cube/node_modules/playwright/index.mjs";

const browser = await chromium.launch({
  executablePath: "/usr/bin/chromium",
  headless: false,
  ignoreHTTPSErrors: true,
  args: ["--no-sandbox", "--ignore-certificate-errors"],
});
const page = await browser.newContext({ ignoreHTTPSErrors: true }).then(c => c.newPage());
page.on("console", m => console.log(`[${m.type()}]`, m.text()));
page.on("pageerror", e => console.log("[pageerror]", e.message));

await page.goto("https://localhost:5180/?dump=1", { waitUntil: "networkidle", timeout: 15000 });
await page.waitForTimeout(2000);
await browser.close();
