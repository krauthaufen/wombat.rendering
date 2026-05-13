// Pipeline cache module — bitfield-encoded GPURenderPipeline registry.
// Consumed by the multi-pipeline-bucket machinery to (a) pre-warm
// pipelines at scene-build, (b) sync-create on the runtime-mutation
// path, (c) look up pipelines by per-RO modeKey at encode.
//
// Manifest persistence (IndexedDB) is part of the design but not yet
// wired — added in a follow-up phase. The in-memory cache is the
// load-bearing piece.

export type {
  PipelineStateDescriptor,
  AttachmentBlend,
  BlendComponent,
  DepthSlice,
  StencilSlice,
  StencilFace,
  CullMode,
  FrontFace,
  Topology,
} from "./descriptor.js";

export {
  DEFAULT_DESCRIPTOR,
  DEFAULT_ATTACHMENT_BLEND,
  descriptorEquals,
} from "./descriptor.js";

export {
  encodeModeKey,
  decodeModeKey,
  MAX_ATTACHMENTS,
} from "./bitfield.js";

export {
  PipelineCache,
  type PipelineBuilder,
} from "./cache.js";
