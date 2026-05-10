// Precision experiment: df32 mat4×mat4 on the GPU vs. naive f32 vs. f64
// reference. For each test case we compute three results, then report
// max absolute and max relative error of df32 and f32 against f64.

import { DF32_MAT4_MUL_WGSL } from "./shader.wgsl.js";
import {
  packDf32Mat4, unpackDf32Mat4, mat4MulF64, mat4MulF32,
} from "./df32.js";
import { allCases, type Case } from "./cases.js";

const out = document.getElementById("out")!;

function setHtml(html: string): void { out.innerHTML = html; }
function escapeHtml(s: string): string {
  return s.replace(/[&<>]/g, c => c === "&" ? "&amp;" : c === "<" ? "&lt;" : "&gt;");
}

async function boot(): Promise<void> {
  if (navigator.gpu === undefined) {
    setHtml(`<pre class="bad">WebGPU not available.</pre>`);
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) { setHtml(`<pre class="bad">No GPU adapter.</pre>`); return; }
  const device = await adapter.requestDevice();

  const module = device.createShaderModule({ code: DF32_MAT4_MUL_WGSL });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  // No external state — kernel primitives are self-contained.

  // ── Diagnostic: cross-buffer dedup test ─────────────────────────────
  // Compute (A[0] + B[0]) − Adup[0] where Adup[0] == A[0] but lives in
  // a separate buffer. If the compiler dedups the loads, result = 0.
  // If buffers are treated as distinct, result = B[0].
  {
    const W = /* wgsl */ `
      @group(0) @binding(0) var<storage, read>       A:    array<f32>;
      @group(0) @binding(1) var<storage, read>       B:    array<f32>;
      @group(0) @binding(2) var<storage, read>       Adup: array<f32>;
      @group(0) @binding(3) var<storage, read_write> Out:  array<f32>;
      @compute @workgroup_size(1)
      fn main() {
        let s = A[0] + B[0];
        Out[0] = s - Adup[0];
      }
    `;
    const m = device.createShaderModule({ code: W });
    const p = device.createComputePipeline({ layout: "auto", compute: { module: m, entryPoint: "main" } });
    const aBuf  = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const bBuf  = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const adBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const oBuf  = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rBuf  = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    // Use B above f32 epsilon (~1e-7) so 1.0+B doesn't truly round to 1.0.
    device.queue.writeBuffer(aBuf,  0, new Float32Array([1.0,  0, 0, 0]));
    device.queue.writeBuffer(bBuf,  0, new Float32Array([1e-4, 0, 0, 0]));
    device.queue.writeBuffer(adBuf, 0, new Float32Array([1.0,  0, 0, 0]));   // same as A, distinct buffer

    // Variant 1: same physical buffer (aBuf) bound at slots 0 AND 2.
    const bg1 = device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: bBuf } },
        { binding: 2, resource: { buffer: aBuf } },   // ← same buffer, different binding
        { binding: 3, resource: { buffer: oBuf } },
      ],
    });
    // Variant 2: distinct buffer with identical contents.
    const bg2 = device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuf  } },
        { binding: 1, resource: { buffer: bBuf  } },
        { binding: 2, resource: { buffer: adBuf } },
        { binding: 3, resource: { buffer: oBuf  } },
      ],
    });

    for (const [label, bg] of [["same-buffer-twice", bg1], ["distinct-buffer", bg2]] as const) {
      const enc = device.createCommandEncoder();
      const ps = enc.beginComputePass();
      ps.setPipeline(p); ps.setBindGroup(0, bg); ps.dispatchWorkgroups(1); ps.end();
      enc.copyBufferToBuffer(oBuf, 0, rBuf, 0, 16);
      device.queue.submit([enc.finish()]);
      await rBuf.mapAsync(GPUMapMode.READ);
      const r = new Float32Array(rBuf.getMappedRange().slice(0));
      rBuf.unmap();
      console.log(`[DF32_DIAG] ${label}: (1.0 + 1e-4) − Adup = ${r[0]} (expect ≈1e-4 if no fold; 0 if folded)`);
    }
    aBuf.destroy(); bBuf.destroy(); adBuf.destroy(); oBuf.destroy(); rBuf.destroy();
  }

  // ── Diagnostic: is fma() single-rounded on this driver? ─────────────
  // True FMA computes a*b+c with one rounding; fma(a,b,-(a*b)) gives the
  // exact rounding error of a*b. Mul+add fallback gives 0.
  {
    const W = /* wgsl */ `
      @group(0) @binding(0) var<storage, read>       In:  array<f32>;
      @group(0) @binding(1) var<storage, read_write> Out: array<f32>;
      @compute @workgroup_size(1)
      fn main() {
        let a = In[0]; let b = In[1];
        let p = a * b;
        Out[0] = p;
        Out[1] = fma(a, b, -p);  // residual ⇒ FMA preserved; 0 ⇒ folded
      }`;
    const m = device.createShaderModule({ code: W });
    const p = device.createComputePipeline({ layout: "auto", compute: { module: m, entryPoint: "main" } });
    const inBuf  = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rdBuf  = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inBuf, 0, new Float32Array([0.1, 0.3, 0, 0]));
    const bg = device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }],
    });
    const enc = device.createCommandEncoder();
    const ps = enc.beginComputePass();
    ps.setPipeline(p); ps.setBindGroup(0, bg); ps.dispatchWorkgroups(1); ps.end();
    enc.copyBufferToBuffer(outBuf, 0, rdBuf, 0, 16);
    device.queue.submit([enc.finish()]);
    await rdBuf.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(rdBuf.getMappedRange().slice(0));
    rdBuf.unmap();
    console.log(`[DF32_DIAG] fma(0.1,0.3,-(0.1*0.3)): p=${r[0]}, err=${r[1]}  ${r[1] === 0 ? "← FOLDED to mul+add" : "← FMA preserved"}`);
    inBuf.destroy(); outBuf.destroy(); rdBuf.destroy();
  }

  // ── Diagnostic: GPU two_sum primitive ───────────────────────────────
  // two_sum(1.0, 1e-9) should give s≈1.0, err≈1e-9. If err==0 the
  // driver is contracting a+b−a → b → 0.
  {
    const TS_WGSL = /* wgsl */ `
      @group(0) @binding(0) var<storage, read>       In:  array<f32>;
      @group(0) @binding(1) var<storage, read_write> Out: array<f32>;
      @group(1) @binding(0) var<storage, read> NoContract: array<f32>;
      fn opaque(x: f32) -> f32 { return x + NoContract[0]; }
      fn two_sum(a: f32, b: f32) -> vec2<f32> {
        let s  = a + b;
        let bb = opaque(s) - a;
        let err = (a - (opaque(s) - bb)) + (b - bb);
        return vec2<f32>(s, err);
      }
      @compute @workgroup_size(1)
      fn main() {
        let r = two_sum(In[0], In[1]);
        Out[0] = r.x;
        Out[1] = r.y;
      }
    `;
    const m = device.createShaderModule({ code: TS_WGSL });
    const p = device.createComputePipeline({ layout: "auto", compute: { module: m, entryPoint: "main" } });
    const inHost = new Float32Array([1.0, 1e-9]);
    const inBuf  = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rdBuf  = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inBuf, 0, inHost);
    const ncBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ncBuf, 0, new Float32Array([0, 0, 0, 0]));
    const bg = device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }],
    });
    const ncBg = device.createBindGroup({
      layout: p.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: ncBuf } }],
    });
    const enc = device.createCommandEncoder();
    const ps = enc.beginComputePass();
    ps.setPipeline(p); ps.setBindGroup(0, bg); ps.setBindGroup(1, ncBg); ps.dispatchWorkgroups(1); ps.end();
    enc.copyBufferToBuffer(outBuf, 0, rdBuf, 0, 16);
    device.queue.submit([enc.finish()]);
    await rdBuf.mapAsync(GPUMapMode.READ);
    const back = new Float32Array(rdBuf.getMappedRange().slice(0));
    rdBuf.unmap();
    console.log(`[DF32_DIAG] two_sum(1.0, 1e-9) → s=${back[0]}, err=${back[1]} (expect err≈1e-9)`);
    inBuf.destroy(); outBuf.destroy(); rdBuf.destroy(); ncBuf.destroy();
  }

  // ── Diagnostic: GPU copy roundtrip (no math) ────────────────────────
  {
    const COPY_WGSL = /* wgsl */ `
      @group(0) @binding(0) var<storage, read>       In:  array<vec2<f32>>;
      @group(0) @binding(1) var<storage, read_write> Out: array<vec2<f32>>;
      @compute @workgroup_size(1)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        Out[gid.x] = In[gid.x];
      }
    `;
    const m = device.createShaderModule({ code: COPY_WGSL });
    const p = device.createComputePipeline({ layout: "auto", compute: { module: m, entryPoint: "main" } });
    const probe = new Float64Array(16);
    for (let i = 0; i < 16; i++) probe[i] = (i + 1) * 0.123456789012345;
    const inHost = new Float32Array(32);
    const { packDf32Mat4: pk, unpackDf32Mat4: un } = await import("./df32.js");
    pk(probe, inHost, 0);
    const inBuf  = device.createBuffer({ size: 128, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rdBuf  = device.createBuffer({ size: 128, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inBuf, 0, inHost);
    const bg = device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }],
    });
    const enc = device.createCommandEncoder();
    const ps = enc.beginComputePass();
    ps.setPipeline(p); ps.setBindGroup(0, bg); ps.dispatchWorkgroups(16); ps.end();
    enc.copyBufferToBuffer(outBuf, 0, rdBuf, 0, 128);
    device.queue.submit([enc.finish()]);
    await rdBuf.mapAsync(GPUMapMode.READ);
    const back = new Float32Array(rdBuf.getMappedRange().slice(0));
    rdBuf.unmap();
    let maxRel = 0;
    const recovered = un(back, 0);
    for (let i = 0; i < 16; i++) {
      const rel = Math.abs(recovered[i]! - probe[i]!) / Math.abs(probe[i]!);
      if (rel > maxRel) maxRel = rel;
    }
    console.log(`[DF32_DIAG] GPU copy roundtrip max rel = ${maxRel.toExponential(3)}`);
    inBuf.destroy(); outBuf.destroy(); rdBuf.destroy();
  }

  // ── Diagnostic: pack / unpack roundtrip with NO compute ─────────────
  // If this is precise (~1e-15), df32 representation is fine and any
  // residual error in the cases is from the GPU multiply itself.
  {
    let maxRel = 0;
    const probe = new Float64Array(16);
    for (let i = 0; i < 16; i++) probe[i] = (i + 1) * 0.123456789012345;
    const buf = new Float32Array(32);
    const { packDf32Mat4, unpackDf32Mat4 } = await import("./df32.js");
    packDf32Mat4(probe, buf, 0);
    const back = unpackDf32Mat4(buf, 0);
    for (let i = 0; i < 16; i++) {
      const rel = Math.abs(back[i]! - probe[i]!) / Math.abs(probe[i]!);
      if (rel > maxRel) maxRel = rel;
    }
    console.log(`[DF32_DIAG] pack→unpack max rel = ${maxRel.toExponential(3)}`);
  }

  // ── Diagnostic: pure-JS f32 simulation of the kernel ────────────────
  // Runs the same algorithm with Math.fround for every op. If THIS
  // achieves f64-level precision, the algorithm is right and the gap
  // is purely WGSL-compiler. If it also stalls at f32-ulp, the
  // algorithm itself has a bug.
  {
    const { df32MatMulSim } = await import("./sim.js");
    const { allCases } = await import("./cases.js");
    const { packDf32Mat4, unpackDf32Mat4, mat4MulF64 } = await import("./df32.js");
    for (const c of allCases()) {
      let maxAbs = 0;
      for (const [A, B] of c.pairs) {
        const aHost = new Float32Array(32); packDf32Mat4(A, aHost, 0);
        const bHost = new Float32Array(32); packDf32Mat4(B, bHost, 0);
        const cHost = df32MatMulSim(aHost, bHost);
        const got = unpackDf32Mat4(cHost, 0);
        const truth = mat4MulF64(A, B);
        for (let k = 0; k < 16; k++) {
          const d = Math.abs(got[k]! - truth[k]!);
          if (d > maxAbs) maxAbs = d;
        }
      }
      console.log(`[DF32_DIAG] sim (Math.fround) ${c.name} max abs = ${maxAbs.toExponential(3)}`);
    }
  }

  const cases = allCases();
  const blocks: string[] = [];
  for (const c of cases) blocks.push(await runCase(device, pipeline, c));
  setHtml(blocks.join(""));
  // Mirror plain-text summary to console for headless capture.
  console.log("[DF32_RESULTS_BEGIN]");
  console.log(out.textContent ?? "");
  console.log("[DF32_RESULTS_END]");
}

async function runCase(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  c: Case,
): Promise<string> {
  const N = c.pairs.length;
  const matFloats = 16 * 2;          // 16 entries × (hi, lo)
  const bufBytes  = N * matFloats * 4;

  // Pack inputs.
  const aHost = new Float32Array(N * matFloats);
  const bHost = new Float32Array(N * matFloats);
  for (let i = 0; i < N; i++) {
    packDf32Mat4(c.pairs[i]![0], aHost, i * matFloats);
    packDf32Mat4(c.pairs[i]![1], bHost, i * matFloats);
  }

  const aBuf = device.createBuffer({
    size: bufBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bBuf = device.createBuffer({
    size: bufBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const cBuf = device.createBuffer({
    size: bufBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readBuf = device.createBuffer({
    size: bufBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(aBuf, 0, aHost);
  device.queue.writeBuffer(bBuf, 0, bHost);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: cBuf } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(N);
  pass.end();
  enc.copyBufferToBuffer(cBuf, 0, readBuf, 0, bufBytes);
  device.queue.submit([enc.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const cHost = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  // Raw-output peek for the first pair, first 4 entries — lets us see
  // whether lo bits are surviving the kernel at all.
  if (c.name === "sanity") {
    const samples: string[] = [];
    for (let k = 0; k < 4; k++) {
      const hi = cHost[k * 2]!;
      const lo = cHost[k * 2 + 1]!;
      samples.push(`[${k}] hi=${hi.toExponential(3)} lo=${lo.toExponential(3)}`);
    }
    console.log(`[DF32_DIAG] sanity raw GPU output: ${samples.join("  ")}`);
  }

  // Compare per pair.
  let df32MaxAbs = 0, df32MaxRel = 0;
  let f32MaxAbs  = 0, f32MaxRel  = 0;
  let df32SumRel = 0, f32SumRel  = 0, count = 0;
  let worstPair = 0, worstK = 0, worstHi = 0, worstLo = 0, worstTruth = 0;
  for (let i = 0; i < N; i++) {
    const [A, B] = c.pairs[i]!;
    const truth  = mat4MulF64(A, B);
    const df32   = unpackDf32Mat4(cHost, i * matFloats);
    const f32    = mat4MulF32(A, B);
    for (let k = 0; k < 16; k++) {
      const t = truth[k]!;
      const dD = Math.abs(df32[k]! - t);
      const dF = Math.abs(f32[k]!  - t);
      const denom = Math.max(Math.abs(t), 1e-30);
      if (dD > df32MaxAbs) {
        df32MaxAbs = dD;
        worstPair = i; worstK = k; worstTruth = t;
        // Re-extract raw (hi, lo) for this entry.
        const matFloatsCopy = 16 * 2;
        const offset = i * matFloatsCopy;
        const r = Math.floor(k / 4), col = k % 4;
        const idxBuf = offset + (col * 4 + r) * 2;
        worstHi = cHost[idxBuf]!;
        worstLo = cHost[idxBuf + 1]!;
      }
      f32MaxAbs  = Math.max(f32MaxAbs,  dF);
      df32MaxRel = Math.max(df32MaxRel, dD / denom);
      f32MaxRel  = Math.max(f32MaxRel,  dF / denom);
      df32SumRel += dD / denom;
      f32SumRel  += dF / denom;
      count++;
    }
  }
  console.log(`[DF32_DIAG] ${c.name} worst: pair=${worstPair} k=${worstK}` +
    `  truth=${worstTruth}  hi=${worstHi}  lo=${worstLo}` +
    `  hi+lo=${worstHi + worstLo}  err=${df32MaxAbs.toExponential(3)}`);

  // Cleanup.
  aBuf.destroy(); bBuf.destroy(); cBuf.destroy(); readBuf.destroy();

  const df32MeanRel = df32SumRel / count;
  const f32MeanRel  = f32SumRel  / count;

  const fmt = (x: number) => x === 0 ? "0" : x.toExponential(2);
  const cls = (rel: number) =>
    rel < 1e-12 ? "ok"   :
    rel < 1e-6  ? "warn" : "bad";

  return `<div class="case">
<pre><span class="label">${escapeHtml(c.name)}</span> — ${escapeHtml(c.description)}
  pairs: ${N}, scalars compared: ${count}
                       max abs       max rel       mean rel
  df32 (GPU)   <span class="${cls(df32MaxRel)}">${fmt(df32MaxAbs).padStart(10)}    ${fmt(df32MaxRel).padStart(10)}    ${fmt(df32MeanRel).padStart(10)}</span>
  f32 (naive)  <span class="${cls(f32MaxRel)}">${fmt(f32MaxAbs).padStart(10)}    ${fmt(f32MaxRel).padStart(10)}    ${fmt(f32MeanRel).padStart(10)}</span></pre>
</div>`;
}

boot().catch(e => setHtml(`<pre class="bad">${escapeHtml(String(e))}\n\n${escapeHtml(e?.stack ?? "")}</pre>`));
