// heap-demo — wombat.rendering.experimental visual smoke test, no
// scene graph. Hand-builds RenderObjects for box / sphere / cylinder
// against a static camera + camera-headlight light. As the heap-
// everything path lands inside the rendering package this demo
// stays the visual baseline — unchanged code, unchanged pixels.

import {
  AVal,
  AList,
  HashMap,
  cset,
  cval,
  transact,
  type aval,
} from "@aardworx/wombat.adaptive";
import {
  V3d, V3f, V4f, M44f, Trafo3d,
} from "@aardworx/wombat.base";
import {
  IBuffer,
  BufferView,
  ElementType,
  PipelineState,
  RenderTree,
  ITexture,
  ISampler,
  type RenderObject,
  type DrawCall,
  type Command,
  type ClearValues,
} from "@aardworx/wombat.rendering.experimental/core";
import { Runtime } from "@aardworx/wombat.rendering.experimental/runtime";
import { attachCanvas, runFrame } from "@aardworx/wombat.rendering.experimental/window";
import { surface, texturedSurface } from "./effects.js";
import { buildBox, buildSphere, buildCylinder, type GeometryData } from "./geometry.js";

const countParam = new URLSearchParams(location.search).get("count");
const ROCount = countParam !== null ? Math.max(1, parseInt(countParam, 10) | 0) : 4;

// ---------------------------------------------------------------------------
// Status banner
// ---------------------------------------------------------------------------

const status = document.getElementById("status")!;
const setStatus = (msg: string, err = false): void => {
  status.textContent = msg;
  status.style.color = err ? "#ff7777" : "#888";
};
window.addEventListener("error", (e) => setStatus("error: " + (e.error?.message ?? e.message), true));
window.addEventListener("unhandledrejection", (e) => setStatus("rejected: " + (e.reason?.message ?? String(e.reason)), true));

setStatus("requesting GPU adapter…");

// ---------------------------------------------------------------------------
// Geometry → BufferView/DrawCall (one-time, shared across instances)
// ---------------------------------------------------------------------------

interface GeoBundle {
  readonly positions: BufferView;
  readonly normals:   BufferView;
  readonly uvs:       BufferView;
  readonly indices:   BufferView;
  readonly drawCall:  aval<DrawCall>;
}

function bundleOf(g: GeometryData): GeoBundle {
  // Float32Array on its own is ambiguous — could be packed scalars,
  // V2f, V3f, V4f, or whatever. The default inference picks F32; we
  // override per-attribute to match what the shader expects.
  return {
    positions: BufferView.ofArray(g.positions, { elementType: ElementType.V3f }),
    normals:   BufferView.ofArray(g.normals,   { elementType: ElementType.V3f }),
    uvs:       BufferView.ofArray(g.uvs,       { elementType: ElementType.V2f }),
    indices:   BufferView.ofArray(g.indices),                    // Uint32Array → ElementType.U32 ✓
    drawCall: AVal.constant<DrawCall>({
      kind: "indexed",
      indexCount:    g.indices.length,
      instanceCount: 1,
      firstIndex:    0,
      baseVertex:    0,
      firstInstance: 0,
    }),
  };
}

// ---------------------------------------------------------------------------
// Effect uses Positions: V4f (the trafo VS keeps it V4f even though
// the bound attribute is V3f — wombat.shader's vertex stage handles
// the V3f→V4f extension via the WGSL emit). The runtime needs the
// attribute layout to match the IR's V3f input though, so we expose
// positions as v3f (12-byte stride). The WGSL output declares the
// padded V4f and the runtime's stride for the binding remains 12.
// ---------------------------------------------------------------------------

function asV4fBroadcast(color: V4f): BufferView {
  // Single-value broadcast — stride 0, one entry, all vertices read it.
  return BufferView.ofValue(color, ElementType.V4f);
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

const target = new V3d(0,  0, 0);
const up     = new V3d(0,  0, 1);

// viewTrafoRH does NOT re-orthogonalize `up` against `forward` — pass
// already-perpendicular vectors. We compute a proper basis here:
// forward = (target − eye).norm, right = forward × worldUp,
// view-up = right × forward.
function buildView(eye: V3d, target: V3d, worldUp: V3d): Trafo3d {
  const fwd  = target.sub(eye).normalize();
  const right = fwd.cross(worldUp).normalize();
  const upRe = right.cross(fwd).normalize();
  return Trafo3d.viewTrafoRH(eye, upRe, fwd);
}
// Time-driven orbit camera. The cval is ticked in `runFrame`'s
// post-render so the runtime's dirty-skip wakes up every rAF
// (anything depending on `time` marks → the wrap aval re-evaluates).
// Also lets the heap A/B comparison sample steady-state encode times
// instead of one-frame numbers.
const time = cval(0);
function eyeAt(t: number): V3d {
  const radius = 16;
  return new V3d(Math.sin(t * 0.0005) * radius, -Math.cos(t * 0.0005) * radius, 8);
}
const eye: aval<V3d> = time.map(eyeAt);
const view: aval<Trafo3d> = eye.map(e => buildView(e, target, up));
function projFor(aspect: number, far = 100): Trafo3d {
  return Trafo3d.perspectiveProjectionRHFov(Math.PI / 3, aspect, 0.1, far);
}

// ---------------------------------------------------------------------------
// Per-RO uniforms
// ---------------------------------------------------------------------------

function trafoToM44f(t: Trafo3d): M44f {
  // The UBO packer needs an `_data: Float32Array` source. Trafo3d's
  // matrices are M44d (Float64Array) — round-trip through M44f.
  return M44f.fromArray(t.forward.toArray());
}
function trafoInvToM44f(t: Trafo3d): M44f {
  return M44f.fromArray(t.backward.toArray());
}

function makeUniforms(
  modelTrafo: Trafo3d,
  viewProjM44: aval<M44f>,
  lightLoc: aval<V3f>,
): HashMap<string, aval<unknown>> {
  // We provide what the trafo+lambert effect reads:
  //   ModelTrafo, ModelTrafoInv, ViewProjTrafo, LightLocation.
  // ViewProjTrafo and LightLocation are passed pre-shared so all ROs
  // see the same aval identity → one arena allocation + one upload
  // per frame instead of N.
  const m  = AVal.constant(trafoToM44f(modelTrafo));
  const mi = AVal.constant(trafoInvToM44f(modelTrafo));
  return HashMap.empty<string, aval<unknown>>()
    .add("ModelTrafo",     m)
    .add("ModelTrafoInv",  mi)
    .add("ViewProjTrafo",  viewProjM44)
    .add("LightLocation",  lightLoc);
}

// ---------------------------------------------------------------------------
// RenderObject factory
// ---------------------------------------------------------------------------

interface RoSpec {
  readonly geo: GeoBundle;
  readonly modelTrafo: Trafo3d;
  readonly color: V4f;
  readonly texture?: aval<ITexture>;
  readonly sampler?: aval<ISampler>;
}

// Shared PipelineState across all ROs — the heap path's bucket key
// includes PipelineState identity, so a fresh-per-RO call would
// create N buckets (= N pipelines + shader modules) for N
// structurally-identical states. Sharing collapses to one bucket.
const sharedPipelineState = PipelineState.constant({
  rasterizer: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
  depth: { write: true, compare: "less" },
});

function makeRO(spec: RoSpec, viewProjM44: aval<M44f>, lightLoc: aval<V3f>): RenderObject {
  const textured = spec.texture !== undefined && spec.sampler !== undefined;
  const textures = textured
    ? HashMap.empty<string, aval<ITexture>>().add("albedo", spec.texture!)
    : HashMap.empty<string, aval<ITexture>>();
  const samplers = textured
    ? HashMap.empty<string, aval<ISampler>>().add("albedo", spec.sampler!)
    : HashMap.empty<string, aval<ISampler>>();
  const baseAttribs = HashMap.empty<string, BufferView>()
    .add("Positions", spec.geo.positions)
    .add("Normals",   spec.geo.normals)
    .add("Colors",    asV4fBroadcast(spec.color));
  return {
    effect: textured ? texturedSurface : surface,
    pipelineState: sharedPipelineState,
    vertexAttributes: textured ? baseAttribs.add("Uvs", spec.geo.uvs) : baseAttribs,
    uniforms: makeUniforms(spec.modelTrafo, viewProjM44, lightLoc),
    textures,
    samplers,
    indices: spec.geo.indices,
    drawCall: spec.geo.drawCall,
  };
}

// ---------------------------------------------------------------------------
// Boot — async because we need a GPU adapter
// ---------------------------------------------------------------------------

const canvas = document.getElementById("cv") as HTMLCanvasElement;

(async function boot(): Promise<void> {
  if (navigator.gpu === undefined) {
    setStatus("WebGPU not available — try Chrome/Edge on a recent OS", true);
    return;
  }
  // ?dump=1 — compile the surface effect and print the WGSL it emits.
  if (new URLSearchParams(location.search).get("dump") === "1") {
    const compiled = surface.compile({ target: "wgsl" });
    for (const s of compiled.stages) {
      console.log(`=== ${s.stage} ===`);
      console.log(s.source);
    }
    setStatus("WGSL dumped to console");
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) { setStatus("no GPU adapter", true); return; }
  // Raise default device limits to whatever the adapter offers — defaults
  // are 128 MB binding / 256 MB buffer; iOS adapters often expose 1 GB.
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  device.addEventListener("uncapturederror", (ev) => {
    const e = (ev as unknown as { error: GPUError }).error;
    console.error("WebGPU uncaptured error:", e.message);
    setStatus("WebGPU error: " + e.message, true);
  });
  device.lost.then(info => console.error("device lost:", info.reason, info.message));
  setStatus("device acquired, attaching canvas…");

  const attach = attachCanvas(device, canvas, {
    depthFormat: "depth24plus",
    colorAttachmentName: "outColor",     // match the fragment output name in effects.ts
  });

  // Sanity: bypass the runtime, clear the canvas to red ONCE via raw
  // WebGPU. If this doesn't show, the swap-chain wiring is broken
  // before any of our code matters. If it does, the runtime path is
  // the suspect.
  if (new URLSearchParams(location.search).get("probe") === "clear") {
    // Drive the clear inside rAF — the swap-chain texture rotates per
    // rAF and the browser only presents whichever was the *current*
    // one for that tick. Submitting outside rAF targets a texture
    // that's then thrown away, which is what produced the original
    // blank canvas in headless.
    const fbA = attach.framebuffer as aval<import("@aardworx/wombat.rendering.experimental/core").IFramebuffer>;
    const tick = (): void => {
      attach.markFrame();
      const fb = fbA.force(/* allow-force */);
      const colorView = fb.colors.tryFind("outColor")!;
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: colorView,
          clearValue: { r: 1, g: 0, b: 0, a: 1 },
          loadOp: "clear", storeOp: "store",
        }],
      });
      pass.end();
      device.queue.submit([enc.finish()]);
      requestAnimationFrame(tick);                  // keep going so the screenshot lands on a real frame
    };
    requestAnimationFrame(tick);
    setStatus("probe=clear: clearing swap-chain to red every rAF");
    return;
  }

  // Precompute geometry bundles. Both paths consume the same raw
  // GeometryData; the per-RO baseline wraps it in BufferViews, the
  // heap path uploads to its own GPUBuffers.
  const rawBox = buildBox();
  const rawSph = buildSphere(32);
  const rawCyl = buildCylinder(32);

  const xPositions = [-6, -2, 2, 6];
  const colors = [
    new V4f(1.00, 0.55, 0.25, 1),
    new V4f(0.45, 0.75, 1.00, 1),
    new V4f(0.95, 0.85, 0.35, 1),
    new V4f(0.55, 0.95, 0.55, 1),
  ];
  const rawGeos = [rawBox, rawSph, rawCyl, rawBox];
  const trafoOf = (i: number, x: number): Trafo3d =>
    i === 1
      ? Trafo3d.scaling(1.0, 1.0, 1.0)                   // sphere
          .mul(Trafo3d.translation(new V3d(x, 0, 0)))
      : i === 2
        ? Trafo3d.scaling(1.0, 1.0, 1.5)                 // cylinder
            .mul(Trafo3d.translation(new V3d(x, 0, -0.75)))
        : Trafo3d.scaling(1.5, 1.5, 1.5)                  // box / box
            .mul(Trafo3d.translation(new V3d(x - 0.75, -0.75, -0.75)));


  // ─── Per-RO baseline (Runtime → Hybrid task) ────────────────────────
  // `?heap-on=0` flips the global heap toggle off so every RO routes
  // through the legacy per-RO renderer — handy for A/B perf comparisons.
  const heapOnParam = new URLSearchParams(location.search).get("heap-on");
  const heapEnabled = cval(heapOnParam === null || heapOnParam === "1");
  const megacall = new URLSearchParams(location.search).get("megacall") === "1";
  const runtime = new Runtime({ device, heapEnabled, megacall });

  const bBox = bundleOf(rawBox);
  const bSph = bundleOf(rawSph);
  const bCyl = bundleOf(rawCyl);
  const geos = [bBox, bSph, bCyl, bBox];

  // ─── Atlas mode (?atlas=1) ──────────────────────────────────────────
  // Generate N distinct 64×64 textures of `rgba8unorm-srgb` (a Tier-S
  // eligible format → AtlasPool packs them all into one atlas page;
  // ROs sharing a texture by aval identity share an atlas sub-rect).
  // The status banner reflects how many distinct textures the scene
  // is rendering through one atlas-aware bucket.
  const atlasMode = new URLSearchParams(location.search).get("atlas") === "1";
  const NUM_ATLAS_TEXTURES = 8;
  const atlasTextures: aval<ITexture>[] = [];
  let sharedSampler: aval<ISampler> | undefined;
  if (atlasMode) {
    const W = 64, H = 64;
    // Generate 8 distinct stripey/gradient patterns, each tinted by a
    // different hue so a casual visual sweep across the grid clearly
    // shows multiple textures landing in the atlas.
    const hues: [number, number, number][] = [
      [255,  64,  64], [ 64, 255,  64], [ 64,  64, 255], [255, 255,  64],
      [255,  64, 255], [ 64, 255, 255], [255, 160,  32], [160,  64, 255],
    ];
    for (let ti = 0; ti < NUM_ATLAS_TEXTURES; ti++) {
      const data = new Uint8Array(W * H * 4);
      const [hr, hg, hb] = hues[ti % hues.length]!;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const o = (y * W + x) * 4;
          // Diagonal-stripe pattern, tinted by per-texture hue. The
          // stripe period rotates per-texture so two adjacent ROs in
          // the grid sampling different atlas entries look distinct.
          const stripe = ((x + y + ti * 4) >> 3) & 1;
          const k = stripe === 0 ? 1.0 : 0.55;
          data[o    ] = Math.min(255, (hr * k) | 0);
          data[o + 1] = Math.min(255, (hg * k) | 0);
          data[o + 2] = Math.min(255, (hb * k) | 0);
          data[o + 3] = 255;
        }
      }
      atlasTextures.push(AVal.constant(ITexture.fromRaw({
        data, width: W, height: H, format: "rgba8unorm-srgb",
      })));
    }
    sharedSampler = AVal.constant(ISampler.fromDescriptor({
      magFilter: "linear", minFilter: "linear",
      addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
    }));
  }
  // Pick a texture for RO index `k` (rotates through the N textures).
  const pickTex = (k: number): aval<ITexture> | undefined =>
    atlasMode ? atlasTextures[k % NUM_ATLAS_TEXTURES]! : undefined;

  // viewProj depends on both size and view (= time-driven). Use a
  // custom aval so a tick on either input recomputes; the runtime's
  // adaptive system handles the cascade.
  const viewProj: aval<Trafo3d> = AVal.custom(token => {
    const { width, height } = attach.size.getValue(token);
    const aspect = Math.max(1e-3, width / Math.max(1, height));
    const projT  = projFor(aspect);
    return view.getValue(token).mul(projT);   // view first, then proj
  });
  // Single shared M44f / lightLoc avals — every RO references the
  // SAME aval objects so the runtime pool sees one identity → one
  // arena allocation + one upload per frame instead of N.
  const viewProjM44: aval<M44f> = viewProj.map(trafoToM44f);
  const lightLoc: aval<V3f> = eye.map(e => new V3f(e.x as number, e.y as number, e.z as number));

  // Build N RO specs — same grid as the heap path so the visual
  // result is comparable. The Runtime path doesn't share buffers
  // across ROs the way the heap path does, so this is the apples-
  // to-apples scaling test.
  const ros: RenderObject[] = [];
  if (ROCount <= 4) {
    for (let i = 0; i < ROCount; i++) {
      ros.push(makeRO({
        geo: geos[i]!,
        modelTrafo: trafoOf(i, xPositions[i]!),
        color: colors[i]!,
        ...(atlasMode ? { texture: pickTex(i)!, sampler: sharedSampler! } : {}),
      }, viewProjM44, lightLoc));
    }
  } else {
    const side = Math.ceil(Math.sqrt(ROCount));
    const spacing = 2.4;
    const center = (side - 1) * 0.5;
    for (let k = 0; k < ROCount; k++) {
      const ix = k % side, iy = Math.floor(k / side);
      const x = (ix - center) * spacing, y = (iy - center) * spacing;
      const geoIdx = k % 3;
      const bundle = geoIdx === 0 ? bBox : geoIdx === 1 ? bSph : bCyl;
      const t = geoIdx === 0
        ? Trafo3d.scaling(0.7, 0.7, 0.7).mul(Trafo3d.translation(new V3d(x - 0.35, y - 0.35, -0.35)))
        : geoIdx === 1
          ? Trafo3d.scaling(0.5, 0.5, 0.5).mul(Trafo3d.translation(new V3d(x, y, 0)))
          : Trafo3d.scaling(0.5, 0.5, 0.8).mul(Trafo3d.translation(new V3d(x, y, -0.4)));
      ros.push(makeRO({
        geo: bundle, modelTrafo: t, color: colors[k % 4]!,
        ...(atlasMode ? { texture: pickTex(k)!, sampler: sharedSampler! } : {}),
      }, viewProjM44, lightLoc));
    }
  }

  // ?churn=N → wrap leaves in a cset and, each frame, remove N random
  // entries + re-add the N removed on the previous frame. Exercises
  // addDraw/removeDraw + the megacall GPU prefix-sum redispatch.
  const churnParam = new URLSearchParams(location.search).get("churn");
  const churn = churnParam !== null ? Math.max(0, parseInt(churnParam, 10) | 0) : 0;
  const leaves = ros.map(o => RenderTree.leaf(o));
  const sceneSet = churn > 0 ? cset<RenderTree>(leaves) : undefined;
  const tree = sceneSet !== undefined
    ? RenderTree.unorderedFromSet(sceneSet)
    : RenderTree.ordered(...leaves);
  const clear: ClearValues = {
    colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0.07, 0.07, 0.08, 1)),
    depth: 1.0,
  };
  const cmds = AList.ofArray<Command>([
    { kind: "Clear", values: clear },
    { kind: "Render", tree },
  ]);
  const task = runtime.compile(attach.signature, cmds);
  // Frame-time samples (same shape as heap path). Note the per-RO
  // baseline is dirty-skip via runFrame, so for static scenes only
  // the first frame fires and stats won't tick. To stress-time it
  // continuously, use the heap path (which has the orbit-driven
  // continuous loop).
  const samples = new Float32Array(240);
  const frameSamples = new Float32Array(240);
  const gpuSamples = new Float32Array(240);
  let sampleI = 0, sampleN = 0, frames = 0, lastReport = performance.now();
  let frameSampleI = 0, frameSampleN = 0, lastRafNow = 0;
  let gpuSampleI = 0, gpuSampleN = 0;
  let gpuSubmitT = 0;
  const pctOf = (buf: Float32Array, n: number, p: number): number => {
    if (n === 0) return 0;
    const sorted = Array.from(buf.subarray(0, n)).sort((a, b) => a - b);
    return sorted[Math.min(n - 1, Math.max(0, Math.floor(p * (n - 1))))]!;
  };
  const pct = (p: number): number => pctOf(samples, sampleN, p);
  const fpct = (p: number): number => pctOf(frameSamples, frameSampleN, p);
  const gpct = (p: number): number => pctOf(gpuSamples, gpuSampleN, p);
  setStatus(`ready — ${ros.length} render objects${atlasMode ? ` · atlas mode (${NUM_ATLAS_TEXTURES} textures)` : ""}`);
  const fbAval = attach.framebuffer as aval<import("@aardworx/wombat.rendering.experimental/core").IFramebuffer>;
  const startTime = performance.now();
  let churnPool: RenderTree[] = [];
  runFrame(attach, (token) => {
    const rafNow = performance.now();
    if (lastRafNow !== 0) {
      frameSamples[frameSampleI] = rafNow - lastRafNow;
      frameSampleI = (frameSampleI + 1) % frameSamples.length;
      if (frameSampleN < frameSamples.length) frameSampleN++;
    }
    lastRafNow = rafNow;

    const t0 = performance.now();
    task.run(fbAval.getValue(token), token);
    const dt = performance.now() - t0;
    samples[sampleI] = dt;
    sampleI = (sampleI + 1) % samples.length;
    if (sampleN < samples.length) sampleN++;
    frames++;
    gpuSubmitT = performance.now();

    const now = performance.now();
    const shouldReport = sampleN === 1 || now - lastReport > 500;
    if (shouldReport) {
      const elapsed = Math.max(1, now - lastReport);
      const fps = (frames * 1000 / elapsed).toFixed(0);
      const tag = megacall ? "megacall" : (heapEnabled.value ? "heap" : "per-RO");
      const churnTag = churn > 0 ? ` · churn=${churn}/frame` : "";
      const atlasTag = atlasMode ? ` · atlas · ${NUM_ATLAS_TEXTURES} textures` : "";
      setStatus(
        `${tag}${atlasTag} · ${ros.length} draws${churnTag} · ${fps} fps · ` +
        `encode ${pct(0.5).toFixed(2)}/${pct(0.99).toFixed(2)} · ` +
        `gpu ${gpct(0.5).toFixed(1)}/${gpct(0.99).toFixed(1)} · ` +
        `frame ${fpct(0.5).toFixed(1)}/${fpct(0.99).toFixed(1)} ms (p50/p99)`,
      );
      frames = 0;
      lastReport = now;
    }
  }, {
    pacer: async () => {
      await device.queue.onSubmittedWorkDone();
      const gpuT = performance.now() - gpuSubmitT;
      gpuSamples[gpuSampleI] = gpuT;
      gpuSampleI = (gpuSampleI + 1) % gpuSamples.length;
      if (gpuSampleN < gpuSamples.length) gpuSampleN++;
    },
    onAfterFrame: () => {
      time.value = performance.now() - startTime;
      if (sceneSet !== undefined && churn > 0) {
        for (const r of churnPool) sceneSet.add(r);
        const live = Array.from(sceneSet);
        churnPool = [];
        for (let i = 0; i < churn && live.length > 0; i++) {
          const idx = (Math.random() * live.length) | 0;
          const victim = live[idx]!;
          live[idx] = live[live.length - 1]!;
          live.pop();
          sceneSet.remove(victim);
          churnPool.push(victim);
        }
      }
    },
  });
})();
