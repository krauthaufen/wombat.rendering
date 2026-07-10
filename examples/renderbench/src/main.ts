// renderbench — wombat port of aardvark.rendering-heap's HeapSpike
// renderbench: heap megacall vs baked single-draw, identical shaders,
// identical measurement protocol, on the Vienna (d1) asset sliced into
// per-building mesh render-objects.
//
// Protocol (mirrors RenderBench.measure):
//   1024×1024 offscreen fbo · 1 build frame · 30 warmup frames ·
//   3 rounds × `frames` (default 60) · report the MEDIAN round's
//   avg gpuMs (min round alongside). GPU time = render-pass
//   timestamp-query when the adapter has it; wall = submit→done.
//
// URL params: ?mode=heap|baked  &parts=N  &frames=60  &size=1024
// Result: window.__benchResult = { mode, parts, verts, gpuMs, gpuMinMs,
//   wallMs, buildMs, pages, timestamps: bool }

import { AVal, AdaptiveToken, HashMap, cval, transact } from "@aardworx/wombat.adaptive";
import type { aval } from "@aardworx/wombat.adaptive";
import { V3d, M44f, Trafo3d } from "@aardworx/wombat.base";
import {
  BufferView, ElementType, PipelineState,
  type RenderObject, type DrawCall, type IFramebuffer,
} from "@aardworx/wombat.rendering/core";
import {
  createFramebufferSignature, allocateFramebuffer, prepareRenderObject,
} from "@aardworx/wombat.rendering/resources";
import { beginPassDescriptor } from "@aardworx/wombat.rendering/commands";
import { buildHeapScene, oct32, c4b, type HeapDrawSpec, type HeapScene } from "@aardworx/wombat.rendering/runtime";
import { heapSurface, bakedSurface } from "./effects.js";

const status = document.getElementById("status")!;
const log = (msg: string): void => {
  status.textContent += "\n" + msg;
  console.log("[bench]", msg);
};

const P = new URLSearchParams(location.search);
const MODE   = (P.get("mode") ?? "heap") as "heap" | "baked" | "compare" | "view";
const PARTS  = P.get("parts") !== null ? Math.max(1, parseInt(P.get("parts")!, 10) | 0) : Infinity;
const FRAMES = P.get("frames") !== null ? Math.max(1, parseInt(P.get("frames")!, 10) | 0) : 60;
const SIZE   = P.get("size") !== null ? Math.max(64, parseInt(P.get("size")!, 10) | 0) : 1024;
/** Model: "d19" = the aardvark CadSceneDemo vienna d01..d09 asset (53 M verts,
 *  ~200 k parts — the scene behind aardvark's demo-floor numbers); "v2" = the
 *  small central-Vienna test patch (4.6 M verts, 43.5 k buildings). */
const MODEL  = (P.get("model") ?? "d19") as "d19" | "v2";
/** Arena page cap in MB (default: the library default, 256 MB). `?pageMB=2048`
 *  ≈ one page for d1-9 — isolates the sub-draw-split cost from the decode cost. */
const PAGE_MB = P.get("pageMB") !== null ? Math.max(64, parseInt(P.get("pageMB")!, 10) | 0) : 0;
/** Adapter pick on dual-GPU boxes: "low" → integrated, "high" → discrete. */
const GPU = (P.get("gpu") ?? "high") as "low" | "high";
/** Which d1-9 districts to load. A count ("3" → d01..d03) or an explicit
 *  comma list ("1,4,8" → d01+d04+d08 — lets a phone combine SMALL districts;
 *  d02 alone is 16.3 M verts, 3× d01). View mode defaults to 1, bench to all 9.
 *  Rough per-district vertex millions: d01 5.4 · d02 16.3 · d03 11.0 · d04 3.1
 *  · d05 3.5 · d06 2.8 · d07 3.2 · d08 2.2 · d09 5.5. */
/** Packed attribute storage (oct32 normals + C4b colors, 4 B/elt in both the
 *  avals and the arena — ~36→20 B/vertex total). Default ON for the
 *  interactive view (memory-bound phones); OFF for bench/compare modes so
 *  their numbers stay comparable with the f32 baseline & aardvark. d19 only. */
const PACKED = P.get("packed") !== null
  ? P.get("packed") === "1"
  : MODE === "view";
const DISTRICT_LIST: number[] = (() => {
  const s = P.get("districts");
  if (s === null) return MODE === "view" ? [1] : [1, 2, 3, 4, 5, 6, 7, 8, 9];
  if (s.includes(",")) {
    return [...new Set(s.split(",").map(x => Math.min(9, Math.max(1, parseInt(x, 10) | 0))))];
  }
  const n = Math.min(9, Math.max(1, parseInt(s, 10) | 0));
  return Array.from({ length: n }, (_, i) => i + 1);
})();

interface Part { readonly v0: number; readonly vn: number; readonly c0?: number; readonly cn?: number }
interface Manifest {
  readonly radius: number;
  readonly buildings?: readonly Part[];
  readonly trees?: readonly Part[];
  readonly ground?: readonly Part[];
  readonly water?: readonly Part[];
}
/** All kinds — buildings, trees, ground, water — ride the bench as plain
 *  parts (the aardvark demo draws them all; flow data is ignored). */
const KINDS = ["buildings", "trees", "ground", "water"] as const;
const partsOf = (m: Manifest): Part[] => KINDS.flatMap((k) => [...(m[k] ?? [])]);

interface Scene {
  /** `c0`/`cn` (PACKED only): absolute range in `colorsC4b`; cn=1 = singleton.
   *  `di` (STREAMING only): district index, with v0/c0 district-LOCAL. */
  readonly parts: readonly { v0: number; vn: number; c0?: number; cn?: number; di?: number }[];
  /** STREAMING (view+packed): per-district dirs; data fetched during ingest. */
  readonly stream?: { readonly dirs: readonly string[] };
  readonly positions: Float32Array;  // V3f, world-space
  readonly normals: Float32Array;    // V3f expanded (empty when PACKED)
  readonly colors: Float32Array;     // tight RGB f32 (empty when PACKED)
  /** PACKED: raw oct32 words, one per vertex. */
  readonly normalsOct?: Uint32Array;
  /** PACKED: raw C4b words (decoupled — parts slice via c0/cn). */
  readonly colorsC4b?: Uint32Array;
  readonly radius: number;
}

/** Fetch `url`.gz when the server has it, falling back to the raw file.
 *  City geometry gzips ~4-13×, cutting the phone-facing transfer from ~90 MB
 *  to ~21 MB per district. Some servers (vite) send `.gz` files with
 *  `Content-Encoding: gzip` — the browser then decodes transparently — while
 *  dumb static servers hand over raw gzip bytes. Sniff the gzip magic
 *  (0x1f 0x8b) and only run DecompressionStream when the body is still raw. */
async function gunzipMaybe(r: Response): Promise<ArrayBuffer> {
  const buf = await r.arrayBuffer();
  const u8 = new Uint8Array(buf, 0, 2);
  if (u8[0] === 0x1f && u8[1] === 0x8b) {
    const ds = new Response(buf).body!.pipeThrough(new DecompressionStream("gzip"));
    return new Response(ds).arrayBuffer();
  }
  return buf;  // server already content-decoded it
}
async function fetchBin(url: string): Promise<ArrayBuffer> {
  const gz = await fetch(`${url}.gz`);
  if (gz.ok) return gunzipMaybe(gz);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.arrayBuffer();
}

async function fetchJson<T>(url: string): Promise<T> {
  const gz = await fetch(`${url}.gz`);
  if (gz.ok) {
    const buf = await gunzipMaybe(gz);
    return JSON.parse(new TextDecoder().decode(buf)) as T;
  }
  return fetch(url).then(r => r.json()) as Promise<T>;
}

/** RGBA f32 (16 B/vertex) → tight RGB f32 (12 B/vertex). Alpha dropped;
 *  both paths re-assemble w=1 (heap decode / classic vertex fetch). */
function rgbaToRgb(src: Float32Array): Float32Array {
  const n = src.length / 4;
  const dst = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    dst[i * 3] = src[i * 4]!; dst[i * 3 + 1] = src[i * 4 + 1]!; dst[i * 3 + 2] = src[i * 4 + 2]!;
  }
  return dst;
}

/** vienna_v2: coupled per-vertex RGBA-f32 colors, V3f normals. */
async function loadV2(): Promise<Scene> {
  const [manifest, posBuf, nrmBuf, colBuf] = await Promise.all([
    fetchJson<Manifest>("/vienna_v2/manifest.json"),
    fetchBin("/vienna_v2/positions.bin"),
    fetchBin("/vienna_v2/normals.bin"),
    fetchBin("/vienna_v2/colors.bin"),
  ]);
  return {
    parts: partsOf(manifest),
    positions: new Float32Array(posBuf),
    normals: new Float32Array(nrmBuf),
    colors: rgbaToRgb(new Float32Array(colBuf)),
    radius: manifest.radius,
  };
}

/** Decode OCT32-packed normals (build_vienna.py oct_pack: x = low u16, y =
 *  high u16, unorm16 → [-1,1], octahedral fold for z<0) to V3f. */
function decodeOct32(packed: Int32Array): Float32Array {
  const n = packed.length;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const w = packed[i]!;
    let x = ((w & 0xffff) / 65535) * 2 - 1;
    let y = (((w >>> 16) & 0xffff) / 65535) * 2 - 1;
    const z = 1 - Math.abs(x) - Math.abs(y);
    if (z < 0) {
      const ox = x;
      x = (1 - Math.abs(y)) * (ox >= 0 ? 1 : -1);
      y = (1 - Math.abs(ox)) * (y >= 0 ? 1 : -1);
    }
    const il = 1 / Math.hypot(x, y, z);
    out[i * 3] = x * il; out[i * 3 + 1] = y * il; out[i * 3 + 2] = z * il;
  }
  return out;
}

/** d1-9 (aardvark CadSceneDemo asset): 9 districts sharing one origin, OCT32
 *  normals, DECOUPLED C4b colors (`c0`/`cn`; cn=1 = single-colour part).
 *  Decoded/expanded to the same V3f/V3f/RGB-f32 layout as v2 so both models
 *  drive identical shaders. */
async function loadD19(): Promise<Scene> {
  const dirs = DISTRICT_LIST.map((n) => `/vienna/d0${n}`);
  // Two passes, SEQUENTIAL per district, so peak memory is one district's
  // temporaries + the merged arrays — not every district's buffers at once
  // (that peak is what OOM-kills a Safari tab).
  // Pass 1: sizes only (cheap: gz manifest + positions content-length would
  // lie, so fetch manifests and read the last part's extent).
  const manifests: Manifest[] = [];
  for (const d of dirs) manifests.push(await fetchJson<Manifest>(`${d}/manifest.json`));
  const districtVerts = manifests.map((m) =>
    partsOf(m).reduce((a, p) => Math.max(a, p.v0 + p.vn), 0));
  const totalVerts = districtVerts.reduce((a, b) => a + b, 0);
  log(`districts ${DISTRICT_LIST.join(",")} → ${(totalVerts / 1e6).toFixed(1)} M verts${PACKED ? " (packed)" : ""}`);
  // STREAMING (view + packed): return the merged parts list with
  // district-LOCAL offsets and no data — the view ingests district by
  // district (fetch → stage → payload released by the heap → GC), so peak
  // host memory is ONE district, and the GPU arena is the only resident
  // copy of the geometry.
  if (MODE === "view" && PACKED) {
    const parts: { v0: number; vn: number; c0: number; cn: number; di: number }[] = [];
    manifests.forEach((m, di) => {
      for (const p of partsOf(m)) {
        parts.push({ v0: p.v0, vn: p.vn, c0: p.c0 ?? p.v0, cn: p.cn ?? p.vn, di });
      }
    });
    return {
      parts,
      positions: new Float32Array(0), normals: new Float32Array(0), colors: new Float32Array(0),
      radius: manifests.reduce((a, m) => Math.max(a, m.radius), 0),
      stream: { dirs },
    };
  }
  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(PACKED ? 0 : totalVerts * 3);
  const colors = new Float32Array(PACKED ? 0 : totalVerts * 3);
  const normalsOct = PACKED ? new Uint32Array(totalVerts) : undefined;
  const colorChunks: Uint32Array[] = [];           // PACKED: per-district raw C4b
  const parts: { v0: number; vn: number; c0?: number; cn?: number }[] = [];
  let base = 0;                                   // running vertex offset
  let colorBase = 0;                              // running C4b-entry offset (PACKED)
  let radius = 0;
  for (let di = 0; di < dirs.length; di++) {
    const d = dirs[di]!;
    const manifest = manifests[di]!;
    {
      // Scope the big temporaries so each district's buffers are GC-able
      // before the next one downloads.
      const posBuf = await fetchBin(`${d}/positions.bin`);
      positions.set(new Float32Array(posBuf), base * 3);
    }
    {
      const nrmBuf = await fetchBin(`${d}/normals.bin`);
      if (PACKED) normalsOct!.set(new Uint32Array(nrmBuf), base);
      else normals.set(decodeOct32(new Int32Array(nrmBuf)), base * 3);
    }
    {
      const colBuf = await fetchBin(`${d}/colors.bin`);
      if (PACKED) {
        // Keep the district's C4b table raw; parts address it via c0/cn
        // (cn=1 singletons stay 4 bytes — the heap broadcasts them).
        colorChunks.push(new Uint32Array(colBuf));
        for (const p of partsOf(manifest)) {
          parts.push({
            v0: base + p.v0, vn: p.vn,
            c0: colorBase + (p.c0 ?? p.v0), cn: p.cn ?? p.vn,
          });
        }
        colorBase += colBuf.byteLength / 4;
      } else {
        const col = new Uint8Array(colBuf);
        for (const p of partsOf(manifest)) {
          // Expand this part's C4b colours (per-vertex or singleton) to
          // per-vertex RGB f32 aligned with its vertices.
          const c0 = p.c0 ?? p.v0;
          const cn = p.cn ?? p.vn;
          for (let i = 0; i < p.vn; i++) {
            const ci = (c0 + (cn === 1 ? 0 : i)) * 4;
            const o = (base + p.v0 + i) * 3;
            colors[o] = col[ci]! / 255; colors[o + 1] = col[ci + 1]! / 255; colors[o + 2] = col[ci + 2]! / 255;
          }
          parts.push({ v0: base + p.v0, vn: p.vn });
        }
      }
    }
    radius = Math.max(radius, manifest.radius);
    base += districtVerts[di]!;
    log(`  ${d.slice(-3)} loaded (${(districtVerts[di]! / 1e6).toFixed(1)} M verts)`);
  }
  let colorsC4b: Uint32Array | undefined;
  if (PACKED) {
    colorsC4b = new Uint32Array(colorBase);
    let o = 0;
    for (const c of colorChunks) { colorsC4b.set(c, o); o += c.length; }
    colorChunks.length = 0;
  }
  return {
    parts, positions, normals, colors, radius,
    ...(normalsOct !== undefined ? { normalsOct } : {}),
    ...(colorsC4b !== undefined ? { colorsC4b } : {}),
  };
}

// buildView from heap-demo (re-orthogonalized basis for viewTrafoRH).
function buildView(eye: V3d, target: V3d, worldUp: V3d): Trafo3d {
  const fwd = target.sub(eye).normalize();
  const right = fwd.cross(worldUp).normalize();
  const upRe = right.cross(fwd).normalize();
  return Trafo3d.viewTrafoRH(eye, upRe, fwd);
}

const trafoToM44f = (t: Trafo3d): M44f => M44f.fromArray(t.forward.toArray());

// Shared pipeline state — cull NONE (winding-agnostic city soup), depth on.
// Identical for both paths.
const pipelineState = PipelineState.constant({
  rasterizer: { topology: "triangle-list", cullMode: "none", frontFace: "ccw" },
  depth: { write: true, compare: "less" },
});

interface RoundResult { gpuMs: number; wallMs: number }

(async () => {
  // ─── Device (maxed storage limits so one page holds the scene) ────
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: GPU === "low" ? "low-power" : "high-performance",
  });
  if (adapter === null) throw new Error("no WebGPU adapter");
  const hasTs = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: hasTs ? ["timestamp-query"] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  const info = adapter.info;
  log(`adapter: ${info.vendor} ${info.architecture} ${info.description || ""} · timestamps: ${hasTs}`);

  // ─── Asset ────────────────────────────────────────────────────────
  log(`loading model '${MODEL}' …`);
  const t0 = performance.now();
  const scene = MODEL === "d19" ? await loadD19() : await loadV2();
  const { positions, normals, colors } = scene;
  const allParts = scene.parts;
  const assetStream = scene.stream;   // streaming view: per-district dirs (no data yet)
  const buildings = Number.isFinite(PARTS) ? allParts.slice(0, PARTS as number) : allParts;
  const totalVerts = buildings.reduce((a, b) => a + b.vn, 0);
  log(`loaded in ${(performance.now() - t0).toFixed(0)} ms · ${allParts.length} parts ` +
      `(using ${buildings.length}) · ${(totalVerts / 1e6).toFixed(2)} M verts · r=${scene.radius.toFixed(0)}`);

  // ─── Camera (ORBITING; aardvark orbit protocol: fov 70°, near 0.1, far 10r) ─
  // One full revolution per measured round (FRAMES frames); compare mode pins
  // the start pose so both paths see the identical matrix. The ViewProjTrafo
  // is a cval — every frame re-marks it, so the heap re-packs its shared
  // region(s) and the baked path re-uploads its UBO: the realistic per-frame
  // uniform churn, not a frozen constant.
  const r = scene.radius;
  const ORBIT_R = Math.hypot(0.6 * r, 0.6 * r);
  const ANGLE0 = Math.atan2(-0.6, 0.6);
  const proj = Trafo3d.perspectiveProjectionRHFov(70 * Math.PI / 180, 1.0, 0.1, 10 * r);
  const vpAt = (angle: number): Trafo3d => {
    const eye = new V3d(Math.cos(angle) * ORBIT_R, Math.sin(angle) * ORBIT_R, 0.35 * r);
    // model→clip composition order per heap-demo (model.mul(view))
    return buildView(eye, new V3d(0, 0, 0), new V3d(0, 0, 1)).mul(proj);
  };
  const viewProjC = cval(vpAt(ANGLE0));
  let orbitFrame = 0;
  const orbitStep = (): void => {
    orbitFrame++;
    transact(() => { viewProjC.value = vpAt(ANGLE0 + (orbitFrame / FRAMES) * 2 * Math.PI); });
  };

  // ─── Offscreen framebuffer ────────────────────────────────────────
  const sig = createFramebufferSignature({
    colors: { outColor: "rgba8unorm" },
    depthStencil: { format: "depth24plus" },
  });
  const fbRes = allocateFramebuffer(device, sig, AVal.constant({ width: SIZE, height: SIZE }), {
    extraUsage: GPUTextureUsage.COPY_SRC,   // compare-mode readback
  });
  fbRes.acquire();
  const fb: IFramebuffer = fbRes.getValue(AdaptiveToken.top);

  // ─── Timestamp plumbing ───────────────────────────────────────────
  const qs = hasTs ? device.createQuerySet({ type: "timestamp", count: 2 }) : undefined;
  const qResolve = hasTs ? device.createBuffer({
    size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  }) : undefined;
  const qRead = hasTs ? device.createBuffer({
    size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  }) : undefined;
  const tsWrites = hasTs
    ? { querySet: qs!, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
    : undefined;
  async function readGpuMs(): Promise<number> {
    if (!hasTs) return 0;
    await qRead!.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(qRead!.getMappedRange().slice(0));
    qRead!.unmap();
    return Number(t[1]! - t[0]!) / 1e6;
  }

  // ─── Scene builders ───────────────────────────────────────────────
  // One frame = encode (+ heap compute prep) + ONE render pass with
  // timestampWrites + submit + await onSubmittedWorkDone.
  let buildMs = 0;
  let pages = 1;

  // Every part gets its OWN changeable ModelTrafo cval — each object is
  // individually movable, so §7 materializes a per-slot constituent (no
  // shared-identity-aval shortcut). This is the honest per-object-uniform
  // cost structure.
  const usePacked = PACKED && scene.normalsOct !== undefined && scene.colorsC4b !== undefined;
  const buildSpecs = (): HeapDrawSpec[] => buildings.map((b) => ({
    effect: heapSurface,
    pipelineState,
    inputs: {
      Positions: positions.subarray(b.v0 * 3, (b.v0 + b.vn) * 3),
      // Packed: oct32 normals + raw C4b colors ride the arena at 4 B/elt
      // (colors often as 4-byte singletons via cn=1 broadcast); the VS
      // decodes. Unpacked: expanded tight f32 (bench-comparable baseline).
      Normals: usePacked
        ? oct32(scene.normalsOct!.subarray(b.v0, b.v0 + b.vn)) as unknown
        : normals.subarray(b.v0 * 3, (b.v0 + b.vn) * 3),
      Colors: usePacked
        ? c4b(scene.colorsC4b!.subarray(b.c0!, b.c0! + b.cn!)) as unknown
        : colors.subarray(b.v0 * 3, (b.v0 + b.vn) * 3),
      ModelTrafo: cval(Trafo3d.identity) as unknown,
      ViewProjTrafo: viewProjC as unknown,
    },
    vertexCount: b.vn,             // non-indexed
  }));

  function makeHeap(): { frame: () => void; dispose: () => void } {
    const specs = buildSpecs();
    log(`building heap scene (${specs.length} draws) …`);
    const tb = performance.now();
    const scene: HeapScene = buildHeapScene(device, sig, specs, {
      fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
      ...(PAGE_MB > 0 ? {
        maxChunkBytes: Math.min(PAGE_MB * 1024 * 1024, device.limits.maxStorageBufferBindingSize),
      } : {}),
    });
    scene.update(AdaptiveToken.top);
    buildMs = performance.now() - tb;
    pages = (scene as unknown as { _debug: { pageCount(): number } })._debug.pageCount();
    log(`heap scene built in ${(buildMs / 1000).toFixed(1)} s · ${pages} page(s) · ` +
        `${scene.stats.groups} bucket(s)`);
    const frame = (): void => {
      scene.update(AdaptiveToken.top);
      const enc = device.createCommandEncoder();
      scene.encodeComputePrep(enc, AdaptiveToken.top);
      const desc = beginPassDescriptor(fb, {
        colors: HashMap.empty<string, unknown>().add("outColor", { _data: [0.05, 0.05, 0.06, 1] }) as never,
        depth: 1.0,
      });
      const pass = enc.beginRenderPass({ ...desc, ...(tsWrites !== undefined ? { timestampWrites: tsWrites } : {}) });
      scene.encodeIntoPass(pass);
      pass.end();
      if (hasTs) {
        enc.resolveQuerySet(qs!, 0, 2, qResolve!, 0);
        enc.copyBufferToBuffer(qResolve!, 0, qRead!, 0, 16);
      }
      device.queue.submit([enc.finish()]);
    };
    return { frame, dispose: () => scene.dispose() };
  }

  function makeBaked(): { frame: () => void; dispose: () => void } {
    // Baked: ONE classic RenderObject over the whole soup (positions are
    // already world-space — this IS aardvark's baked floor). Draw [0, maxV):
    // for the full model the parts tile the buffer exactly (maxV == Σvn);
    // under a --parts cap this covers the used prefix (incl. any gap verts).
    const maxV = buildings.reduce((a, b) => Math.max(a, b.v0 + b.vn), 0);
    if (maxV !== totalVerts) log(`note: parts don't tile [0,maxV): maxV=${maxV} Σvn=${totalVerts}`);
    const tb = performance.now();
    const attribs = HashMap.empty<string, BufferView>()
      .add("Positions", BufferView.ofArray(positions.subarray(0, maxV * 3), { elementType: ElementType.V3f }))
      .add("Normals",   BufferView.ofArray(normals.subarray(0, maxV * 3),   { elementType: ElementType.V3f }))
      .add("Colors",    BufferView.ofArray(colors.subarray(0, maxV * 3),    { elementType: ElementType.V3f }));
    const uniforms = HashMap.empty<string, aval<unknown>>()
      .add("ViewProjTrafo", viewProjC.map(trafoToM44f) as unknown as aval<unknown>);
    const ro: RenderObject = {
      effect: bakedSurface,
      pipelineState,
      vertexAttributes: attribs as never,
      uniforms: uniforms as never,
      textures: HashMap.empty() as never,
      samplers: HashMap.empty() as never,
      drawCall: AVal.constant<DrawCall>({
        kind: "non-indexed", vertexCount: maxV, instanceCount: 1, firstVertex: 0, firstInstance: 0,
      }),
    };
    const compiled = bakedSurface.compile({
      target: "wgsl",
      fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
    } as never);
    const prepared = prepareRenderObject(device, ro, compiled as never, sig, { effectId: bakedSurface.id });
    prepared.acquire();
    prepared.update(AdaptiveToken.top);
    buildMs = performance.now() - tb;
    log(`baked RO prepared in ${(buildMs / 1000).toFixed(1)} s · ${(totalVerts / 1e6).toFixed(2)} M verts, 1 draw`);
    const frame = (): void => {
      prepared.update(AdaptiveToken.top);   // re-upload the (cval-driven) VP UBO
      const enc = device.createCommandEncoder();
      const desc = beginPassDescriptor(fb, {
        colors: HashMap.empty<string, unknown>().add("outColor", { _data: [0.05, 0.05, 0.06, 1] }) as never,
        depth: 1.0,
      });
      const pass = enc.beginRenderPass({ ...desc, ...(tsWrites !== undefined ? { timestampWrites: tsWrites } : {}) });
      prepared.record(pass, AdaptiveToken.top);
      pass.end();
      if (hasTs) {
        enc.resolveQuerySet(qs!, 0, 2, qResolve!, 0);
        enc.copyBufferToBuffer(qResolve!, 0, qRead!, 0, 16);
      }
      device.queue.submit([enc.finish()]);
    };
    return { frame, dispose: () => prepared.release() };
  }

  // ─── Framebuffer readback (compare mode) ──────────────────────────
  async function readbackColor(): Promise<Uint8Array> {
    const tex = fb.colorTextures!.tryFind("outColor")!;
    const bytesPerRow = SIZE * 4;                     // 4096 for 1024 — 256-aligned
    const buf = device.createBuffer({ size: bytesPerRow * SIZE, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow }, { width: SIZE, height: SIZE });
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const px = new Uint8Array(buf.getMappedRange().slice(0));
    buf.unmap(); buf.destroy();
    return px;
  }
  function blit(px: Uint8Array, label: string): void {
    const cv = document.createElement("canvas");
    cv.width = SIZE; cv.height = SIZE;
    cv.style.width = "480px"; cv.style.imageRendering = "auto";
    cv.title = label;
    document.body.appendChild(cv);
    const ctx = cv.getContext("2d")!;
    const img = ctx.createImageData(SIZE, SIZE);
    img.data.set(px);
    for (let i = 3; i < img.data.length; i += 4) img.data[i] = 255; // opaque for display
    ctx.putImageData(img, 0, 0);
  }

  if (MODE === "view") {
    // ─── Interactive viewer: fullscreen canvas + touch orbit/pinch ──
    // One-finger drag orbits, two-finger pinch zooms, two-finger drag pans,
    // mouse wheel zooms (desktop). Renders the HEAP path on every rAF.
    const cvEl = document.createElement("canvas");
    Object.assign(cvEl.style, {
      position: "fixed", inset: "0", width: "100vw", height: "100vh",
      touchAction: "none", background: "#000",
    } as CSSStyleDeclaration);
    document.body.appendChild(cvEl);
    status.style.position = "fixed";
    status.style.zIndex = "10";
    const ctx = cvEl.getContext("webgpu")!;
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format: canvasFormat, alphaMode: "opaque" });
    const viewSig = createFramebufferSignature({
      colors: { outColor: canvasFormat as never },
      depthStencil: { format: "depth24plus" },
    });

    // Camera: spherical around the scene centre.
    const target = { x: 0, y: 0, z: 0 };
    let theta = ANGLE0, phi = Math.atan2(0.35, 0.849), dist = 0.918 * r;
    const vpNow = (aspect: number): Trafo3d => {
      const cp = Math.cos(phi), sp = Math.sin(phi);
      const eye = new V3d(
        target.x + Math.cos(theta) * cp * dist,
        target.y + Math.sin(theta) * cp * dist,
        target.z + sp * dist,
      );
      const view = buildView(eye, new V3d(target.x, target.y, target.z), new V3d(0, 0, 1));
      const projA = Trafo3d.perspectiveProjectionRHFov(
        70 * Math.PI / 180, aspect, Math.max(0.05, dist * 0.001), 10 * r);
      return view.mul(projA);
    };

    // Touch / pointer controls.
    const pointers = new Map<number, { x: number; y: number }>();
    let lastSpan = 0;
    cvEl.addEventListener("pointerdown", (e) => {
      cvEl.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        lastSpan = Math.hypot(a!.x - b!.x, a!.y - b!.y);
      }
    });
    cvEl.addEventListener("pointermove", (e) => {
      const prev = pointers.get(e.pointerId);
      if (prev === undefined) return;
      const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 1) {
        theta -= dx * 0.005;
        phi = Math.min(1.45, Math.max(0.02, phi + dy * 0.005));
      } else if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const span = Math.hypot(a!.x - b!.x, a!.y - b!.y);
        if (lastSpan > 0) dist = Math.min(3 * r, Math.max(0.002 * r, dist * lastSpan / span));
        lastSpan = span;
        // two-finger drag pans in the camera's ground plane
        const k = dist * 0.001;
        target.x += (-dx * Math.sin(-theta) - dy * Math.cos(-theta)) * k * 0.5;
        target.y += (-dx * Math.cos(theta) + dy * Math.sin(theta)) * k * 0.5;
      }
    });
    const drop = (e: PointerEvent): void => { pointers.delete(e.pointerId); lastSpan = 0; };
    cvEl.addEventListener("pointerup", drop);
    cvEl.addEventListener("pointercancel", drop);
    cvEl.addEventListener("wheel", (e) => {
      e.preventDefault();
      dist = Math.min(3 * r, Math.max(0.002 * r, dist * (e.deltaY > 0 ? 1.1 : 0.9)));
    }, { passive: false });

    // Heap scene against the canvas signature. Pre-size the arena from the
    // manifest math — pow2-grow during ingest keeps OLD+NEW GPU buffers
    // alive across each doubling copy, and that transient (not steady state)
    // is what OOM-kills Safari tabs mid-load.
    const packedLayout = usePacked || assetStream !== undefined;
    const colorEntries = packedLayout
      ? buildings.reduce((a, b) => a + (b.cn ?? b.vn), 0)
      : totalVerts * 3;
    const arenaEstimate = packedLayout
      ? totalVerts * 16 + colorEntries * 4 + buildings.length * 192
      : totalVerts * 36 + buildings.length * 192;
    // Surface GPU failures VISIBLY — a lost device otherwise just renders a
    // black canvas while the rAF loop reports a healthy fps.
    device.lost.then((info) => log(`DEVICE LOST: ${info.reason} — ${info.message}`));
    device.onuncapturederror = (e) => log(`GPU ERROR: ${e.error.message.slice(0, 300)}`);
    log(`limits: storBind ${(device.limits.maxStorageBufferBindingSize / 1e6).toFixed(0)} MB · ` +
        `buf ${(device.limits.maxBufferSize / 1e6).toFixed(0)} MB`);

    log(`building heap scene (${buildings.length} draws, arena ≈ ${(arenaEstimate / 1e6).toFixed(0)} MB) …`);
    const tb = performance.now();
    // Streaming ingest: draws land in slices with a queue submit between,
    // so WebGPU's writeBuffer staging is bounded to one slice (iOS Safari's
    // GPU process otherwise holds the WHOLE arena's staging until the first
    // submit — a transient that kills the tab). With `stream`, districts are
    // additionally fetched one at a time and their host buffers dropped as
    // soon as the heap has staged them (releaseConstantAttributes) — the GPU
    // arena is the only resident copy of the geometry.
    const scene: HeapScene = buildHeapScene(device, viewSig, [], {
      fragmentOutputLayout: { locations: new Map([["outColor", 0]]) },
      initialArenaBytes: Math.ceil(arenaEstimate * 1.05),
      releaseConstantAttributes: true,
    });
    const SLICE = 4096;
    let partsDone = 0;
    let ingestDone = false;
    const nextTick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));
    const ingest = async (): Promise<void> => {
      if (assetStream !== undefined) {
        // Per-district: fetch → build specs over district-local views →
        // stage → let the district buffers go (heap released the payloads).
        for (let di = 0; di < assetStream.dirs.length; di++) {
          const d = assetStream.dirs[di]!;
          const [posBuf, nrmBuf, colBuf] = await Promise.all([
            fetchBin(`${d}/positions.bin`), fetchBin(`${d}/normals.bin`), fetchBin(`${d}/colors.bin`),
          ]);
          const pos = new Float32Array(posBuf);
          const oct = new Uint32Array(nrmBuf);
          const col = new Uint32Array(colBuf);
          const dParts = buildings.filter((p) => p.di === di);
          for (let i = 0; i < dParts.length; i += SLICE) {
            const end = Math.min(i + SLICE, dParts.length);
            for (let j = i; j < end; j++) {
              const p = dParts[j]!;
              scene.addDraw({
                effect: heapSurface,
                pipelineState,
                inputs: {
                  Positions: pos.subarray(p.v0 * 3, (p.v0 + p.vn) * 3),
                  Normals:   oct32(oct.subarray(p.v0, p.v0 + p.vn)) as unknown,
                  Colors:    c4b(col.subarray(p.c0!, p.c0! + p.cn!)) as unknown,
                  ModelTrafo: cval(Trafo3d.identity) as unknown,
                  ViewProjTrafo: viewProjC as unknown,
                },
                vertexCount: p.vn,
              });
            }
            partsDone += end - i;
            scene.update(AdaptiveToken.top);
            device.queue.submit([]);           // flush writeBuffer staging
            await nextTick();                   // progressive render + GC breath
          }
          log(`  ${d.slice(-3)} streamed`);
        }
      } else {
        const allSpecs = buildSpecs();
        for (let i = 0; i < allSpecs.length; i += SLICE) {
          for (let j = i; j < Math.min(i + SLICE, allSpecs.length); j++) scene.addDraw(allSpecs[j]!);
          partsDone = Math.min(allSpecs.length, i + SLICE);
          scene.update(AdaptiveToken.top);
          device.queue.submit([]);
          await nextTick();
        }
      }
      ingestDone = true;
      log(`built in ${((performance.now() - tb) / 1000).toFixed(1)} s · ` +
          `${(scene as unknown as { _debug: { pageCount(): number } })._debug.pageCount()} page(s)`);
    };
    void ingest().catch((e) => log(`INGEST ERROR: ${(e as Error).message}`));

    let depth: GPUTexture | undefined;
    let dw = 0, dh = 0;
    let frames = 0, lastFps = performance.now();
    const loop = (): void => {
      const w = Math.max(1, Math.floor(cvEl.clientWidth * devicePixelRatio));
      const h = Math.max(1, Math.floor(cvEl.clientHeight * devicePixelRatio));
      if (cvEl.width !== w || cvEl.height !== h) { cvEl.width = w; cvEl.height = h; }
      if (depth === undefined || dw !== w || dh !== h) {
        depth?.destroy();
        depth = device.createTexture({
          size: { width: w, height: h }, format: "depth24plus",
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        dw = w; dh = h;
      }
      transact(() => { viewProjC.value = vpNow(w / h); });
      scene.update(AdaptiveToken.top);
      const enc = device.createCommandEncoder();
      scene.encodeComputePrep(enc, AdaptiveToken.top);
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: ctx.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.05, b: 0.06, a: 1 },
          loadOp: "clear", storeOp: "store",
        }],
        depthStencilAttachment: {
          view: depth.createView(),
          depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
        },
      });
      scene.encodeIntoPass(pass);
      pass.end();
      device.queue.submit([enc.finish()]);
      frames++;
      const now = performance.now();
      if (now - lastFps > 500) {
        status.textContent =
          `${(frames * 1000 / (now - lastFps)).toFixed(0)} fps · ` +
          (ingestDone
            ? `${buildings.length} parts`
            : `loading ${partsDone}/${buildings.length} parts`) +
          ` · ${(totalVerts / 1e6).toFixed(1)} M verts · d[${DISTRICT_LIST.join(",")}]`;
        frames = 0; lastFps = now;
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
    return;
  }

  if (MODE === "compare") {
    // Render one frame per path against the SAME framebuffer, read back,
    // pixel-diff. ModelTrafo is identity, so the two paths compute
    // bit-identical clip positions — any structural error (wrong page
    // binding, bad _indexStart, mis-seated refs) shows as differing pixels.
    const heap = makeHeap();
    heap.frame();
    await device.queue.onSubmittedWorkDone();
    const a = await readbackColor();
    heap.dispose();
    const baked = makeBaked();
    baked.frame();
    await device.queue.onSubmittedWorkDone();
    const b = await readbackColor();
    baked.dispose();
    let diff = 0, maxDelta = 0, nonClearA = 0, nonClearB = 0;
    for (let i = 0; i < a.length; i += 4) {
      const da = Math.max(Math.abs(a[i]! - b[i]!), Math.abs(a[i + 1]! - b[i + 1]!), Math.abs(a[i + 2]! - b[i + 2]!));
      if (da > 0) { diff++; if (da > maxDelta) maxDelta = da; }
      // clear = (0.05, 0.05, 0.06) ≈ (13, 13, 15) in unorm8
      if (a[i]! !== 13 || a[i + 1]! !== 13 || a[i + 2]! !== 15) nonClearA++;
      if (b[i]! !== 13 || b[i + 1]! !== 13 || b[i + 2]! !== 15) nonClearB++;
    }
    blit(a, "heap"); blit(b, "baked");
    const total = (SIZE * SIZE);
    const result = {
      mode: "compare", parts: buildings.length, verts: totalVerts, pages,
      diffPixels: diff, diffPct: (100 * diff / total), maxChannelDelta: maxDelta,
      coveredHeapPct: (100 * nonClearA / total), coveredBakedPct: (100 * nonClearB / total),
    };
    (window as unknown as { __benchResult: unknown }).__benchResult = result;
    log(`COMPARE: covered heap ${result.coveredHeapPct.toFixed(1)}% / baked ${result.coveredBakedPct.toFixed(1)}% · ` +
        `diff ${diff} px (${result.diffPct.toFixed(3)}%) · maxΔ ${maxDelta}`);
    return;
  }

  const path = MODE === "heap" ? makeHeap() : makeBaked();
  const frame = path.frame;

  // ─── measure() — aardvark protocol (orbiting camera) ─────────────
  const runFrame = async (): Promise<{ gpu: number; wall: number }> => {
    orbitStep();                            // advance the camera: VP cval marks
    const w0 = performance.now();
    frame();
    await device.queue.onSubmittedWorkDone();
    const wall = performance.now() - w0;
    const gpu = await readGpuMs();
    return { gpu, wall };
  };

  await runFrame();                                   // build + first submit
  log("warmup (30 frames) …");
  for (let i = 0; i < 30; i++) await runFrame();      // clock ramp

  const round = async (): Promise<RoundResult> => {
    let gpu = 0, wall = 0;
    for (let i = 0; i < FRAMES; i++) {
      const s = await runFrame();
      gpu += s.gpu; wall += s.wall;
    }
    return { gpuMs: gpu / FRAMES, wallMs: wall / FRAMES };
  };
  log(`3 rounds × ${FRAMES} frames …`);
  const rounds: RoundResult[] = [];
  for (let i = 0; i < 3; i++) rounds.push(await round());
  const key = hasTs ? (x: RoundResult) => x.gpuMs : (x: RoundResult) => x.wallMs;
  const sorted = [...rounds].sort((a, b) => key(a) - key(b));
  const med = sorted[1]!;
  const min = sorted[0]!;

  const result = {
    mode: MODE, parts: buildings.length, verts: totalVerts,
    gpuMs: med.gpuMs, gpuMinMs: min.gpuMs, wallMs: med.wallMs,
    buildMs, pages, timestamps: hasTs,
  };
  (window as unknown as { __benchResult: unknown }).__benchResult = result;
  log(`RESULT ${MODE}: gpu ${med.gpuMs.toFixed(2)} ms (min ${min.gpuMs.toFixed(2)}) · ` +
      `wall ${med.wallMs.toFixed(2)} ms · build ${(buildMs / 1000).toFixed(1)} s`);
})().catch((e) => {
  log(`ERROR: ${(e as Error).message}`);
  (window as unknown as { __benchResult: unknown }).__benchResult = { error: (e as Error).message };
  console.error(e);
});
