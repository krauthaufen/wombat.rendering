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

export {
  BufferView,
} from "./bufferView.js";

export {
  ElementType,
} from "./elementType.js";

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
  DepthBiasState,
  DepthState,
  FrontFace,
  PlainBlendComponentState,
  PlainBlendState,
  PlainDepthState,
  PlainPipelineState,
  PlainRasterizerState,
  PlainStencilFaceState,
  PlainStencilState,
  RasterizerState,
  StencilFaceState,
  StencilState,
  Topology,
} from "./pipelineState.js";

export { PipelineState } from "./pipelineState.js";

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
  AttributeInfo,
  CompiledEffect,
  CompiledStage,
  ComputeShader,
  Effect,
  HoleGetter,
  HoleGetters,
  LooseUniformInfo,
  OutputInfo,
  ProgramInterface,
  SamplerInfo,
  Stage,
  StageInfo,
  StorageBufferInfo,
  Target,
  TextureInfo,
  UniformBlockInfo,
  UniformFieldInfo,
} from "./shader.js";

export {
  AdaptiveResource,
} from "./adaptiveResource.js";

export {
  tryAcquire,
  tryRelease,
} from "./acquire.js";

export {
  RenderContext,
} from "./renderContext.js";
