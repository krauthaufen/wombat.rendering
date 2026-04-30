// Public API of @aardworx/wombat.rendering-resources.
//
// Layer 2 — resource-creator functions on `GPUDevice`. Each
// `prepare*` function lifts a user-provided `aval<I…>` source
// type into a ref-counted `AdaptiveResource<GPU…>` that:
//   - Allocates the GPU handle on first compute / on host
//     mismatch.
//   - Uploads on host-side change, reuses the existing handle
//     when shape-compatible.
//   - Frees the handle on last `release()`.

export {
  prepareAdaptiveBuffer,
  type PrepareAdaptiveBufferOptions,
} from "./adaptiveBuffer.js";

export {
  prepareAdaptiveTexture,
  type PrepareAdaptiveTextureOptions,
} from "./adaptiveTexture.js";

export {
  prepareAdaptiveSampler,
} from "./adaptiveSampler.js";

export {
  prepareUniformBuffer,
  type PrepareUniformBufferOptions,
} from "./uniformBuffer.js";

export {
  compileRenderPipeline,
  type CompileRenderPipelineDescription,
} from "./renderPipeline.js";

export {
  installShaderDiagnostics,
  type ShaderDiagnosticsOptions,
} from "./shaderDiagnostics.js";

export {
  generateMips,
  type GenerateMipsOptions,
} from "./mipGen.js";

export {
  prepareRenderObject,
  PreparedRenderObject,
  type PrepareRenderObjectOptions,
} from "./preparedRenderObject.js";

export {
  createFramebufferSignature,
  type FramebufferSignatureSpec,
} from "./framebufferSignature.js";

export {
  allocateFramebuffer,
  type AllocateFramebufferOptions,
  type FramebufferSize,
} from "./framebuffer.js";

export {
  BufferUsage,
  ColorWrite,
  ShaderStage,
  TextureUsage,
} from "./webgpuFlags.js";
