// heap-demo — wombat.rendering.experimental visual smoke test, no
// scene graph. Hand-builds RenderObjects for box / sphere / cylinder
// against a static camera + camera-headlight light. As the heap-
// everything path lands inside the rendering package this demo
// stays the visual baseline — unchanged code, unchanged pixels.

import {
  AVal,
  AList,
  HashMap,
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
  type RenderObject,
  type DrawCall,
  type Command,
  type ClearValues,
  type ITexture,
  type ISampler,
} from "@aardworx/wombat.rendering.experimental/core";
import { Runtime } from "@aardworx/wombat.rendering.experimental/runtime";
import { attachCanvas, runFrame } from "@aardworx/wombat.rendering.experimental/window";
import { surface } from "./effects.js";
import { buildBox, buildSphere, buildCylinder, type GeometryData } from "./geometry.js";
import { buildHeapRenderer, type HeapDrawSpec } from "./heap.js";

// `?heap=1` selects the experimental heap-backed render path, which
// bypasses Runtime/prepareRenderObject and drives WebGPU directly.
// Anything else → the existing per-RO baseline.
const useHeap = new URLSearchParams(location.search).get("heap") === "1";

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

const eye    = new V3d(0, -16, 8);
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
const view = AVal.constant(buildView(eye, target, up));
const projFor = (aspect: number): Trafo3d =>
  Trafo3d.perspectiveProjectionRHFov(Math.PI / 3, aspect, 0.1, 100);  // 60° hFoV

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
  viewProj: aval<Trafo3d>,
  _viewT: aval<Trafo3d>,
): HashMap<string, aval<unknown>> {
  // We provide what the trafo+lambert effect reads:
  //   ModelTrafo, ModelTrafoInv, ViewProjTrafo, LightLocation.
  // The runtime UBO packer expects `_data`-bearing values; convert
  // Trafo3d → M44f at the source. Same trick as wombat.dom's
  // compile.ts:adaptForGpu.
  const m  = AVal.constant(trafoToM44f(modelTrafo));
  const mi = AVal.constant(trafoInvToM44f(modelTrafo));
  const vp = viewProj.map(trafoToM44f);
  const lightLoc = AVal.constant(new V3f(eye.x as number, eye.y as number, eye.z as number));
  return HashMap.empty<string, aval<unknown>>()
    .add("ModelTrafo",     m)
    .add("ModelTrafoInv",  mi)
    .add("ViewProjTrafo",  vp)
    .add("LightLocation",  lightLoc);
}

// ---------------------------------------------------------------------------
// RenderObject factory
// ---------------------------------------------------------------------------

interface RoSpec {
  readonly geo: GeoBundle;
  readonly modelTrafo: Trafo3d;
  readonly color: V4f;
}

function makeRO(spec: RoSpec, viewProj: aval<Trafo3d>): RenderObject {
  return {
    effect: surface,
    pipelineState: PipelineState.constant({
      rasterizer: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
      depth: { write: true, compare: "less" },
    }),
    vertexAttributes: HashMap.empty<string, BufferView>()
      .add("Positions", spec.geo.positions)
      .add("Normals",   spec.geo.normals)
      .add("Colors",    asV4fBroadcast(spec.color)),
    uniforms: makeUniforms(spec.modelTrafo, viewProj, view),
    textures: HashMap.empty<string, aval<ITexture>>(),
    samplers: HashMap.empty<string, aval<ISampler>>(),
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

  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) { setStatus("no GPU adapter", true); return; }
  const device  = await adapter.requestDevice();
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

  // ─── Heap path ───────────────────────────────────────────────────────
  if (useHeap) {
    // Slot 2 (cylinder): live rotation. Slot 0 (orange box): live
    // colour pulse. Both drive the slot-writer path — per-draw bytes
    // tick up in the stats banner. Other slots stay static (cval
    // wrapping a constant).
    const cylBase = trafoOf(2, xPositions[2]!);
    const cylTrafo = cval(cylBase);
    const boxColor = cval(colors[0]!);
    const draws: HeapDrawSpec[] = xPositions.map((x, i) => ({
      geo: rawGeos[i]!,
      modelTrafo: i === 2 ? cylTrafo : trafoOf(i, x),
      color:      i === 0 ? boxColor : colors[i]!,
    }));
    const renderer = buildHeapRenderer(device, attach, draws);

    // Animate camera in a slow orbit around origin so we drive the
    // globals-upload path every frame while the per-draw heap stays
    // stale (= 0 bytes/frame, demonstrating the slot-writer payoff).
    // Plain rAF instead of runFrame: the demo wants continuous
    // re-render; runFrame is dirty-skip and would idle after one
    // frame for a non-aval-driven loop.
    const baseRadius = 18;
    const baseHeight = 8;
    const start = performance.now();
    let frames = 0, lastReport = start;

    const tick = (): void => {
      const t = (performance.now() - start) * 0.0005;          // ~0.5 rad/s
      const eyeNow = new V3d(
        Math.sin(t) * baseRadius,
        -Math.cos(t) * baseRadius,
        baseHeight,
      );
      const viewNow = buildView(eyeNow, target, up);

      const { width, height } = AVal.force(attach.size);
      const aspect = Math.max(1e-3, width / Math.max(1, height));
      const viewProjT = viewNow.mul(projFor(aspect));

      // Drive slot-2 (cylinder) trafo + slot-0 (box) colour through
      // the cvals — addMarkingCallback inside the renderer fires,
      // marks the slots dirty, and the next frame() picks them up.
      transact(() => {
        const angle = t * 4;                                   // spin faster than the camera orbit
        cylTrafo.value = Trafo3d.rotationZ(angle).mul(cylBase);
        const r = 0.5 + 0.5 * Math.sin(t * 6);
        boxColor.value = new V4f(r, 0.55 * (1 - r * 0.5), 0.25, 1);
      });

      attach.markFrame();
      renderer.frame(viewProjT, eyeNow);

      frames++;
      const now = performance.now();
      if (now - lastReport > 500) {
        const fps = (frames * 1000 / (now - lastReport)).toFixed(0);
        setStatus(
          `heap, ${draws.length} draws · ${fps} fps · ` +
          `globals/frame: ${renderer.stats.globalsBytes} B · ` +
          `per-draw/frame: ${renderer.stats.drawBytes} B · ` +
          `geometry (one-time): ${(renderer.stats.geometryBytes / 1024).toFixed(1)} KiB`,
        );
        frames = 0;
        lastReport = now;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    return;
  }

  // ─── Per-RO baseline (Runtime + prepareRenderObject) ────────────────
  const runtime = new Runtime({ device });

  const bBox = bundleOf(rawBox);
  const bSph = bundleOf(rawSph);
  const bCyl = bundleOf(rawCyl);
  const geos = [bBox, bSph, bCyl, bBox];

  const viewProj: aval<Trafo3d> = attach.size.map(({ width, height }) => {
    const aspect = Math.max(1e-3, width / Math.max(1, height));
    const projT  = projFor(aspect);
    const viewT  = AVal.force(view);
    return viewT.mul(projT);            // view first, then proj
  });

  const ros: RenderObject[] = xPositions.map((x, i) => makeRO({
    geo: geos[i]!,
    modelTrafo: trafoOf(i, x),
    color: colors[i]!,
  }, viewProj));

  const tree = RenderTree.ordered(...ros.map(o => RenderTree.leaf(o)));
  const clear: ClearValues = {
    colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0.07, 0.07, 0.08, 1)),
    depth: 1.0,
  };
  const cmds = AList.ofArray<Command>([
    { kind: "Clear", output: attach.framebuffer, values: clear },
    { kind: "Render", output: attach.framebuffer, tree },
  ]);
  const task = runtime.compile(cmds);
  setStatus(`ready — per-RO path, ${ros.length} render objects (append ?heap=1 for heap path)`);
  runFrame(attach, (token) => task.run(token));
})();
