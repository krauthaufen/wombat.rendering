#!/usr/bin/env node
// renderbench driver — spawns the vite dev server, drives the bench page in
// the persistent headed Chromium (real RTX GPU; see machine notes — headless
// WebGPU is unreliable on this box), runs heap + baked, prints the table.
//
//   node bench.mjs [--parts N] [--frames 60] [--size 1024] [--modes heap,baked]

import { spawn } from "node:child_process";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { browser } = require("/home/schorsch/.headed-chrome");

const arg = (name, dflt) => {
  const i = process.argv.indexOf(name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : dflt;
};
const PARTS = arg("--parts", null);
const PAGE_MB = arg("--pageMB", null);
const PACKED = arg("--packed", null);
const FRAMES = arg("--frames", "60");
const SIZE = arg("--size", "1024");
const MODES = arg("--modes", "heap,baked").split(",");
const PORT = 5173;

function startVite() {
  // Reclaim the port from a stale orphaned vite (a SIGTERM'd earlier run's
  // child survives its parent).
  try { spawn("fuser", ["-k", `${PORT}/tcp`]).on("error", () => {}); } catch { /* best-effort */ }
  return new Promise((resolve, reject) => {
    // detached → own process group, so kill(-pid) takes the whole npx→vite
    // tree down (kill(pid) only hits the npx wrapper; the orphaned vite child
    // keeps the port AND its stdio pipes keep our event loop alive forever).
    const p = spawn("npx", ["vite", "--port", String(PORT), "--strictPort"], {
      cwd: new URL(".", import.meta.url).pathname,
      stdio: ["ignore", "pipe", "pipe"],
      detached: true,
    });
    let ready = false;
    const onData = (d) => {
      const s = d.toString();
      if (!ready && (s.includes("Local:") || s.includes("ready in"))) {
        ready = true;
        resolve(p);
      }
    };
    p.stdout.on("data", onData);
    p.stderr.on("data", (d) => process.stderr.write(`[vite] ${d}`));
    p.on("exit", (c) => { if (!ready) reject(new Error(`vite exited ${c}`)); });
    setTimeout(() => { if (!ready) reject(new Error("vite start timeout")); }, 30000);
  });
}

async function runMode(b, mode) {
  const page = await b.newPage();
  page.on("console", (m) => {
    const t = m.text();
    if (t.startsWith("[bench]")) console.log(`  ${t.slice(8)}`);
  });
  const q = new URLSearchParams({ mode, frames: FRAMES, size: SIZE });
  if (PARTS !== null) q.set("parts", PARTS);
  if (PAGE_MB !== null) q.set("pageMB", PAGE_MB);
  if (PACKED !== null) q.set("packed", PACKED);
  await page.goto(`https://localhost:${PORT}/?${q}`, { waitUntil: "domcontentloaded" });
  // Bench end-to-end: asset load (~270 MB) + build + 210 measured frames.
  const deadline = Date.now() + 15 * 60 * 1000;
  let result;
  for (;;) {
    result = await page.evaluate(() => window.__benchResult);
    if (result !== undefined) break;
    if (Date.now() > deadline) throw new Error(`${mode}: timeout`);
    await new Promise((r) => setTimeout(r, 1000));
  }
  if (mode === "compare") {
    // The page blitted both renderings onto canvases — save them.
    const shot = new URL("./compare.png", import.meta.url).pathname;
    await page.screenshot({ path: shot, fullPage: true });
    console.log(`  screenshots → ${shot}`);
  }
  await page.close();
  if (result.error) throw new Error(`${mode}: ${result.error}`);
  return result;
}

const vite = await startVite();
console.log(`vite up on :${PORT}`);
const b = await browser();
const results = {};
try {
  for (const mode of MODES) {
    console.log(`\n── ${mode} ──`);
    results[mode] = await runMode(b, mode);
  }
} finally {
  b.disconnect();          // never close() — shared instance
  try { process.kill(-vite.pid, "SIGTERM"); } catch { vite.kill(); }
}

console.log("\n════════ renderbench (wombat) ════════");
const f = (v) => v === undefined ? "—" : v.toFixed(2).padStart(8);
for (const [mode, r] of Object.entries(results)) {
  if (mode === "compare") {
    console.log(
      `compare  covered heap ${r.coveredHeapPct.toFixed(1)}% / baked ${r.coveredBakedPct.toFixed(1)}%` +
      `  diff ${r.diffPixels} px (${r.diffPct.toFixed(3)}%)  maxΔ ${r.maxChannelDelta}  · ${r.parts} parts · ${r.pages} page(s)`,
    );
    continue;
  }
  console.log(
    `${mode.padEnd(6)} gpu ${f(r.gpuMs)} ms (min ${f(r.gpuMinMs)})  wall ${f(r.wallMs)} ms` +
    `  · ${r.parts} parts · ${(r.verts / 1e6).toFixed(2)} M verts · ${r.pages} page(s)` +
    `  · build ${(r.buildMs / 1000).toFixed(1)} s${r.timestamps ? "" : "  [no ts — wall only]"}`,
  );
}
if (results.heap && results.baked) {
  const num = results.heap.timestamps ? results.heap.gpuMs : results.heap.wallMs;
  const den = results.baked.timestamps ? results.baked.gpuMs : results.baked.wallMs;
  console.log(`\nheap / baked = ${(num / den).toFixed(2)}x   (aardvark clean-room floors: NVIDIA 1.2-1.4x, typed-partitions ~1.0x)`);
}
// Hard exit — lingering CDP sockets / pipe handles must not keep us alive.
process.exit(0);
