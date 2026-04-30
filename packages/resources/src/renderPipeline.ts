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
  readonly vertexEntryPoint: string;
  readonly fragmentEntryPoint: string;
  readonly vertexBufferLayouts: readonly GPUVertexBufferLayout[];
  readonly bindGroupLayouts: readonly GPUBindGroupLayout[];
  readonly colorTargets: readonly GPUColorTargetState[];
  readonly depthStencil?: GPUDepthStencilState;
  readonly primitive: GPUPrimitiveState;
  readonly multisample?: GPUMultisampleState;
}

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

function moduleFor(device: GPUDevice, source: string, label?: string): GPUShaderModule {
  const cache = cacheFor(device);
  let m = cache.modules.get(source);
  if (m === undefined) {
    m = device.createShaderModule({ code: source, ...(label !== undefined ? { label } : {}) });
    cache.modules.set(source, m);
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

  const vsModule = moduleFor(device, desc.vertexShaderSource, desc.label ? `${desc.label}.vs` : undefined);
  const fsModule = desc.vertexShaderSource === desc.fragmentShaderSource
    ? vsModule
    : moduleFor(device, desc.fragmentShaderSource, desc.label ? `${desc.label}.fs` : undefined);

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
  // Strong identity component: prefer `effectId` (wombat.shader's
  // build-time stable hash); fall back to FNV-hashing the shader
  // source for hand-built effects. The rest of the descriptor is
  // small enough to JSON-stringify.
  const ident = d.effectId !== undefined
    ? d.effectId
    : `${hashString(d.vertexShaderSource)}/${hashString(d.fragmentShaderSource)}`;
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
