// Headless visual check for heap-demo. Opens the dev server in
// Playwright's launched system Chromium with WebGPU enabled,
// captures console + page errors + the canvas pixels, dumps status.
//
// Usage:
//   node check.mjs               # baseline path
//   node check.mjs --heap        # ?heap=1 (experimental path)
//   node check.mjs --url <url>   # any URL

import { chromium } from "/home/schorsch/projects/wombat.dom/examples/hello-cube/node_modules/playwright/index.mjs";
import fs from "node:fs";

const args = process.argv.slice(2);
const useHeap = args.includes("--heap");
const probeClear = args.includes("--probe-clear");
const urlIdx = args.indexOf("--url");
const urlArg = urlIdx >= 0 ? args[urlIdx + 1] : undefined;
const qs = probeClear ? "?probe=clear" : useHeap ? "?heap=1" : "";
const url = urlArg ?? `https://localhost:5180/${qs}`;
const outBase = probeClear ? "probe-clear" : useHeap ? "heap" : "baseline";

// Headless Chromium can run WebGPU via SwiftShader, but the canvas
// compositor doesn't deliver swap-chain pixels in pure-headless. Run
// non-headless under xvfb-run instead — gives Chrome a real X display
// + GPU access via native Vulkan, and the swap-chain presents
// normally. Wrap the script with `xvfb-run node check.mjs`.
const headless = process.env.CHECK_HEADLESS === "1";
const browser = await chromium.launch({
  executablePath: "/usr/bin/chromium",
  headless,
  ignoreHTTPSErrors: true,
  args: [
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    ...(headless
      ? [
          "--enable-features=Vulkan",
          "--use-vulkan=swiftshader",
          "--use-webgpu-adapter=swiftshader",
        ]
      : [
          "--enable-features=Vulkan,UseSkiaRenderer",
          "--use-vulkan=native",
          "--enable-webgpu-developer-features",
          "--use-angle=vulkan",
        ]),
    "--ignore-gpu-blocklist",
    "--ignore-certificate-errors",
  ],
});
const ctx = await browser.newContext({
  viewport: { width: 1024, height: 600 },
  ignoreHTTPSErrors: true,
});
const page = await ctx.newPage();
const log = [];
page.on("console",   m => log.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", e => log.push(`[pageerror] ${e.message}`));
page.on("requestfailed", r => log.push(`[reqfail] ${r.url()} :: ${r.failure()?.errorText ?? "?"}`));

console.log("opening", url);
try {
  await page.goto(url, { waitUntil: "networkidle", timeout: 15000 });
} catch (e) {
  console.log("navigation error:", e.message);
}
await page.waitForTimeout(5000);  // let WebGPU init + several rAF ticks land

// Probe canvas dims.
const dims = await page.evaluate(() => {
  const c = document.querySelector("canvas");
  if (!c) return null;
  return {
    cssW: c.clientWidth,
    cssH: c.clientHeight,
    bsW: c.width,
    bsH: c.height,
  };
});
console.log("canvas:", JSON.stringify(dims));

// Probe rAF: count how many frames the loop has fired.
const rafCount = await page.evaluate(() => new Promise(resolve => {
  let n = 0;
  const start = performance.now();
  const tick = () => {
    n++;
    if (performance.now() - start > 500) resolve(n);
    else requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}));
console.log("rAF ticks in 500ms:", rafCount);

// Probe WebGPU adapter info.
const adapter = await page.evaluate(async () => {
  if (!navigator.gpu) return "(no navigator.gpu)";
  const a = await navigator.gpu.requestAdapter();
  if (!a) return "(no adapter)";
  return a.info ? JSON.stringify(a.info) : "(adapter ok, no info)";
});
console.log("adapter:", adapter);

// Snapshot the actual canvas pixels (bypasses the compositor path).
const canvasPng = await page.evaluate(() => {
  const c = document.querySelector("canvas");
  return c?.toDataURL("image/png") ?? null;
});
if (canvasPng) {
  const b64 = canvasPng.replace(/^data:image\/png;base64,/, "");
  fs.writeFileSync(`${outBase}-canvas.png`, Buffer.from(b64, "base64"));
} else {
  console.log("(no canvas found)");
}
await page.screenshot({ path: `${outBase}-screenshot.png`, fullPage: true });

const status = await page.locator("#status").textContent().catch(() => "(no #status)");
console.log("status:", status);
console.log("--- logs ---");
for (const l of log) console.log(l);

await browser.close();
