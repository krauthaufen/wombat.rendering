// compileRenderPipeline — produce a `GPURenderPipeline` from a
// description and memoize per-device. Pipeline state is treated
// as static (not adaptive); when the user wants pipeline switching
// at runtime they should construct multiple PreparedRenderObjects
// and pick between them at the RenderTree level.
//
// The descriptor is intentionally low-level: all the hard work of
// turning a `CompiledEffect` + `ProgramInterface` + `PipelineState`
// + `FramebufferSignature` into a GPUVertexBufferLayout[] /
// GPUBindGroupLayout[] / GPUColorTargetState[] etc. lives in
// `prepareRenderObject`. This function is just the caching layer
// over `device.createRenderPipeline`.

export interface CompileRenderPipelineDescription {
  readonly label?: string;
  /**
   * Stable identity of the source effect (wombat.shader's
   * build-time `Effect.id`). Used as the strong key for the
   * pipeline cache — two `compileRenderPipeline` calls with the
   * same `effectId` + the rest of the descriptor share a
   * `GPURenderPipeline`. Falls back to FNV-hashing the shader
   * source when omitted (only useful for hand-built sources).
   */
  readonly effectId?: string;
  readonly vertexShaderSource: string;
  readonly fragmentShaderSource: string;
  /** Optional source map for the vertex stage. Used to enrich compile-error logs. */
  readonly vertexSourceMap?: import("@aardworx/wombat.shader/ir").SourceMap | null;
  readonly fragmentSourceMap?: import("@aardworx/wombat.shader/ir").SourceMap | null;
  readonly vertexEntryPoint: string;
  readonly fragmentEntryPoint: string;
  readonly vertexBufferLayouts: readonly GPUVertexBufferLayout[];
  readonly bindGroupLayouts: readonly GPUBindGroupLayout[];
  readonly colorTargets: readonly GPUColorTargetState[];
  readonly depthStencil?: GPUDepthStencilState;
  readonly primitive: GPUPrimitiveState;
  readonly multisample?: GPUMultisampleState;
}

import { installShaderDiagnostics } from "./shaderDiagnostics.js";

interface DeviceCache {
  modules: Map<string, GPUShaderModule>;
  pipelines: Map<string, GPURenderPipeline>;
}

const caches = new WeakMap<GPUDevice, DeviceCache>();
function cacheFor(device: GPUDevice): DeviceCache {
  let c = caches.get(device);
  if (c === undefined) {
    c = { modules: new Map(), pipelines: new Map() };
    caches.set(device, c);
  }
  return c;
}

function moduleFor(
  device: GPUDevice,
  source: string,
  label?: string,
  sourceMap?: import("@aardworx/wombat.shader/ir").SourceMap | null,
): GPUShaderModule {
  const cache = cacheFor(device);
  let m = cache.modules.get(source);
  if (m === undefined) {
    m = device.createShaderModule({ code: source, ...(label !== undefined ? { label } : {}) });
    cache.modules.set(source, m);
    installShaderDiagnostics(m, source, {
      ...(label !== undefined ? { label } : {}),
      ...(sourceMap !== undefined && sourceMap !== null ? { sourceMap } : {}),
    });
  }
  return m;
}

export function compileRenderPipeline(
  device: GPUDevice,
  desc: CompileRenderPipelineDescription,
): GPURenderPipeline {
  const cache = cacheFor(device);
  const k = pipelineKey(desc);
  let p = cache.pipelines.get(k);
  if (p !== undefined) return p;

  const vsModule = moduleFor(device, desc.vertexShaderSource, desc.label ? `${desc.label}.vs` : undefined, desc.vertexSourceMap);
  const fsModule = desc.vertexShaderSource === desc.fragmentShaderSource
    ? vsModule
    : moduleFor(device, desc.fragmentShaderSource, desc.label ? `${desc.label}.fs` : undefined, desc.fragmentSourceMap);

  const layout = device.createPipelineLayout({
    bindGroupLayouts: desc.bindGroupLayouts as GPUBindGroupLayout[],
    ...(desc.label !== undefined ? { label: `${desc.label}.layout` } : {}),
  });

  const pdesc: GPURenderPipelineDescriptor = {
    layout,
    vertex: {
      module: vsModule,
      entryPoint: desc.vertexEntryPoint,
      buffers: desc.vertexBufferLayouts as GPUVertexBufferLayout[],
    },
    fragment: {
      module: fsModule,
      entryPoint: desc.fragmentEntryPoint,
      targets: desc.colorTargets as GPUColorTargetState[],
    },
    primitive: desc.primitive,
    ...(desc.depthStencil !== undefined ? { depthStencil: desc.depthStencil } : {}),
    ...(desc.multisample !== undefined ? { multisample: desc.multisample } : {}),
    ...(desc.label !== undefined ? { label: desc.label } : {}),
  };
  p = device.createRenderPipeline(pdesc);
  cache.pipelines.set(k, p);
  return p;
}

function pipelineKey(d: CompileRenderPipelineDescription): string {
  // Identity = effect id AND a fingerprint of the actual source.
  //
  // `effectId` is wombat.shader's hash of the effect TEMPLATE — its shape and
  // types. It is NOT a hash of the emitted code: closure-hole values are
  // specialised into the source as literals, so one effect id can legitimately
  // emit different shaders (`const NULLU = 4294967295u` vs another cap, a
  // baked tint, a compile-time branch). Keying on the id alone handed such an
  // effect a pipeline built from a PREVIOUS hole value — a stale shader that
  // silently keeps running. The source hash is what actually distinguishes two
  // programs, so it always participates; the id stays in the key as a cheap
  // discriminator (and keeps hash collisions from crossing effects).
  const ident = d.effectId !== undefined
    ? `${d.effectId}/${sourceHash(d.vertexShaderSource)}/${sourceHash(d.fragmentShaderSource)}`
    : `${sourceHash(d.vertexShaderSource)}/${sourceHash(d.fragmentShaderSource)}`;
  const slim = {
    id: ident,
    vEntry: d.vertexEntryPoint,
    fEntry: d.fragmentEntryPoint,
    vb: d.vertexBufferLayouts,
    bgl: d.bindGroupLayouts.length,
    ct: d.colorTargets,
    ds: d.depthStencil,
    pr: d.primitive,
    ms: d.multisample,
  };
  return JSON.stringify(slim);
}

// Source hashes are memoised: `compileRenderPipeline` runs per prepared render
// object, and a scene with thousands of them re-presents the SAME source string
// (out of the effect's compile cache) every time. Hashing a multi-KB shader on
// each of those would be pure waste; the map lookup is one hash of an already-
// hashed string instance.
const sourceHashes = new Map<string, number>();
function sourceHash(s: string): number {
  let h = sourceHashes.get(s);
  if (h === undefined) {
    h = hashString(s);
    // Bounded: a pathological generator could otherwise grow this without end.
    if (sourceHashes.size > 4096) sourceHashes.clear();
    sourceHashes.set(s, h);
  }
  return h;
}

function hashString(s: string): number {
  // FNV-1a 32-bit; collision risk is negligible for our purposes
  // and the cost is one call per uncached pipeline.
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
