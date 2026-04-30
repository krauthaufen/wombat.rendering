// Headless verification: open the dev server in Playwright Chromium
// with WebGPU enabled, render a frame, save a screenshot, exit.

import { chromium } from "playwright";

const browser = await chromium.launch({
  // Use system Chromium 146 (full build, not the headless_shell one
  // bundled with Playwright — that lacks the compositor needed to
  // present WebGPU to the canvas).
  executablePath: "/usr/bin/chromium",
  headless: true,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,UseSkiaRenderer",
    "--use-vulkan=native",
    "--ignore-gpu-blocklist",
    "--enable-webgpu-developer-features",
    "--use-angle=vulkan",
  ],
});
const ctx = await browser.newContext();
const page = await ctx.newPage();
const consoleLog = [];
page.on("console", m => consoleLog.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", e => consoleLog.push(`[pageerror] ${e.message}`));

await page.goto("http://localhost:5174/", { waitUntil: "networkidle" });
await page.waitForTimeout(2000);  // let a few rAF ticks happen
// Save the canvas's raw bytes (bypasses the compositor / screenshot path).
const canvasPng = await page.evaluate(() => {
  const c = document.getElementById("gpu");
  return c.toDataURL("image/png");
});
const fs = await import("node:fs");
const b64 = canvasPng.replace(/^data:image\/png;base64,/, "");
fs.writeFileSync("canvas.png", Buffer.from(b64, "base64"));
await page.screenshot({ path: "screenshot.png" });

const status = await page.locator("#status").textContent();
console.log("status:", status);
console.log("---logs---");
for (const l of consoleLog) console.log(l);

await browser.close();
