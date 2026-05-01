// Re-exports of the wombat.shader runtime types consumed by the
// rendering layer. We do NOT duplicate or re-shape these — the
// rendering layer is just a consumer of wombat.shader's
// `Effect` / `CompiledEffect` / `ProgramInterface`.

export type {
  Effect,
  ComputeShader,
  Stage,
  HoleGetter,
  HoleGetters,
  CompiledEffect,
  CompiledStage,
  Target,
} from "@aardworx/wombat.shader-runtime";

export type {
  ProgramInterface,
  StageInfo,
  AttributeInfo,
  OutputInfo,
  LooseUniformInfo,
  UniformBlockInfo,
  UniformFieldInfo,
  SamplerInfo,
  TextureInfo,
  StorageBufferInfo,
} from "@aardworx/wombat.shader-runtime";
