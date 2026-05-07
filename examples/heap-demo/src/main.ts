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
  type aval,
} from "@aardworx/wombat.adaptive";
import {
  V3d, V4f, Trafo3d,
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
  return {
    positions: BufferView.ofArray(g.positions),                  // → ElementType.F32 (TypedArray)
    normals:   BufferView.ofArray(g.normals),
    indices:   BufferView.ofArray(g.indices),                    // → ElementType.U32
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

function asV3f(view: BufferView): BufferView {
  return { ...view, elementType: ElementType.V3f };
}
function asV4fBroadcast(color: V4f): BufferView {
  // Single-value broadcast — stride 0, one entry, all vertices read it.
  return BufferView.ofValue(color, ElementType.V4f);
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

const eye    = new V3d(0, -10, 6);
const target = new V3d(0,  0, 0);
const up     = new V3d(0,  0, 1);

const view = AVal.constant(Trafo3d.viewTrafoRH(eye, up, target.sub(eye).normalize()));
const projFor = (aspect: number): Trafo3d =>
  Trafo3d.perspectiveProjectionRHFov(Math.PI / 4, aspect, 0.1, 100);

// ---------------------------------------------------------------------------
// Per-RO uniforms
// ---------------------------------------------------------------------------

function makeUniforms(
  modelTrafo: Trafo3d,
  viewProj: aval<Trafo3d>,
  viewT: aval<Trafo3d>,
): HashMap<string, aval<unknown>> {
  // We provide what the trafo+lambert effect reads:
  //   ModelTrafo, ModelTrafoInv, ViewProjTrafo, LightLocation
  // wombat.shader auto-generates the WGSL uniform block layout — we
  // just supply the values; the runtime packs them.
  const m   = AVal.constant(modelTrafo);
  const mi  = AVal.constant(modelTrafo.inverse);
  // Camera-headlight: light = eye position.
  const lightLoc = viewT.map(_v => new V3f(eye.x as number, eye.y as number, eye.z as number));
  return HashMap.empty<string, aval<unknown>>()
    .add("ModelTrafo",     m)
    .add("ModelTrafoInv",  mi)
    .add("ViewProjTrafo",  viewProj)
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
      .add("Positions", asV3f(spec.geo.positions))
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
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) { setStatus("no GPU adapter", true); return; }
  const device  = await adapter.requestDevice();
  setStatus("device acquired, attaching canvas…");

  const attach = attachCanvas(device, canvas, {
    depthFormat: "depth24plus",
  });
  const runtime = new Runtime({ device });

  // Precompute geometry bundles.
  const bBox = bundleOf(buildBox());
  const bSph = bundleOf(buildSphere(32));
  const bCyl = bundleOf(buildCylinder(32));

  // Six instances along X. Solids on z = 0; alternate the species.
  const xPositions = [-6, -2, 2, 6];
  const colors = [
    new V4f(1.00, 0.55, 0.25, 1),
    new V4f(0.45, 0.75, 1.00, 1),
    new V4f(0.95, 0.85, 0.35, 1),
    new V4f(0.55, 0.95, 0.55, 1),
  ];
  const geos = [bBox, bSph, bCyl, bBox];

  // Live aspect — recomputes when the canvas resizes.
  const viewProj: aval<Trafo3d> = attach.size.map(({ width, height }) => {
    const aspect = Math.max(1e-3, width / Math.max(1, height));
    const projT  = projFor(aspect);
    const viewT  = AVal.force(view);
    return projT.mul(viewT);
  });

  const ros: RenderObject[] = xPositions.map((x, i) => makeRO({
    geo: geos[i]!,
    // Box is in [0,1]^3 — recentre via -0.5 offset and scale to ~1.5.
    // Sphere/cylinder are unit-sized at origin already.
    modelTrafo: i === 1
      ? Trafo3d.scaling(1.0, 1.0, 1.0)               // sphere
        .mul(Trafo3d.translation(new V3d(x, 0, 0)))
      : i === 2
        ? Trafo3d.scaling(1.0, 1.0, 1.5)             // cylinder
            .mul(Trafo3d.translation(new V3d(x, 0, -0.75)))
        : Trafo3d.scaling(1.5, 1.5, 1.5)              // box / box
            .mul(Trafo3d.translation(new V3d(x - 0.75, -0.75, -0.75))),
    color: colors[i]!,
  }, viewProj));

  // Build the command list. One Render command, all ROs as ordered children.
  const tree = RenderTree.ordered(...ros.map(o => RenderTree.leaf(o)));
  const clear: ClearValues = {
    colors: HashMap.empty<string, V4f>().add("color", new V4f(0.07, 0.07, 0.08, 1)),
    depth: 1.0,
  };
  const cmds = AList.ofArray<Command>([
    { kind: "Clear", target: attach.framebuffer, values: clear },
    { kind: "Render", output: attach.framebuffer, tree },
  ]);

  const task = runtime.compile(cmds);
  setStatus(`ready — ${ros.length} render objects`);

  runFrame(attach, (token) => {
    task.run(token);
  });
})();
