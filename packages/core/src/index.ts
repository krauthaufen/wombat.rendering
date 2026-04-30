// Public API of @aardworx/wombat.rendering-core.
//
// Phase 0: types only. Implementations of `compileEffect`,
// `prepareRenderObject`, `runRenderTask`, etc. live in higher
// packages (`resources`, `commands`, `runtime`).

export {
  IBuffer,
} from "./buffer.js";
export type {
  HostBufferSource,
} from "./buffer.js";

export type {
  BufferView,
} from "./bufferView.js";

export type {
  ClearColor,
  ClearValues,
} from "./clear.js";

export type {
  Command,
  CopySpec,
  BufferCopy,
  TextureCopy,
  BufferCopyRange,
} from "./command.js";

export type {
  DrawCall,
  IndexedDrawCall,
  NonIndexedDrawCall,
} from "./drawCall.js";

export type {
  IFramebuffer,
} from "./framebuffer.js";

export type {
  DepthStencilAttachmentSignature,
  FramebufferSignature,
} from "./framebufferSignature.js";

export type {
  BlendComponentState,
  BlendState,
  CullMode,
  DepthState,
  FrontFace,
  PipelineState,
  RasterizerState,
  StencilFaceState,
  StencilState,
  Topology,
} from "./pipelineState.js";

export type {
  RenderObject,
} from "./renderObject.js";

export {
  RenderTree,
} from "./renderTree.js";

export type {
  IRenderTask,
} from "./renderTask.js";

export {
  ISampler,
} from "./sampler.js";

export {
  ITexture,
} from "./texture.js";
export type {
  ExternalTextureSource,
  HostTextureSource,
  RawTextureSource,
} from "./texture.js";

export type {
  CompiledEffect,
  Effect,
  ProgramInterface,
  UniformBufferLayout,
} from "./shader.js";

export {
  AdaptiveResource,
} from "./adaptiveResource.js";

export {
  RenderContext,
} from "./renderContext.js";
