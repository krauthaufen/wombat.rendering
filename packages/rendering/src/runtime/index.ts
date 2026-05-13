// Public API of @aardworx/wombat.rendering-runtime.

export {
  Runtime,
  type RuntimeOptions,
} from "./runtime.js";

export {
  compileRenderTask,
  type RuntimeContext,
} from "./renderTask.js";

export { copy } from "./copy.js";

export {
  renderTo,
  type RenderToOptions,
  type RenderToResult,
} from "./renderTo.js";

export {
  ScenePass,
  type WalkerStats,
} from "./scenePass.js";

export {
  buildHeapScene,
  type HeapScene,
  type HeapSceneStats,
  type HeapDrawSpec,
  type HeapGeometry,
  type HeapTextureSet,
  type BuildHeapSceneOptions,
} from "./heapScene.js";

export {
  compileHeapEffect,
  type CompiledHeapEffect,
  type FragmentOutputLayout,
} from "./heapEffect.js";

export { flattenRenderTree } from "./flattenTree.js";
export { isHeapEligible } from "./heapEligibility.js";
export { renderObjectToHeapSpec } from "./heapAdapter.js";

// ─── Derived uniforms (§7): a uniform binding is a value (aval/constant) OR a rule ──
export {
  derivedUniform, DerivedExpr, type DerivedScope,
  ruleFromIR, uniformRef, isDerivedRule, hashIR,
  type DerivedRule, type IRFragment,
  STANDARD_DERIVED_RULES, isStandardDerivedName,
} from "./derivedUniforms/index.js";

export {
  compileHybridScene,
  type CompileHybridSceneOptions,
  type HybridScene,
} from "./hybridScene.js";

// ─── Texture atlas (heap path Tier-S texture sharing) ───────────────
export {
  BvhTree2d,
  SPLIT_LIMIT_DEFAULT,
} from "./textureAtlas/bvhTree2d.js";

export {
  TexturePacking,
  empty as emptyTexturePacking,
  isEmpty as isEmptyTexturePacking,
  count as countTexturePacking,
  occupancy as texturePackingOccupancy,
  square as squareTexturePacking,
  tryAddMany as texturePackingTryAddMany,
  tryOfArray as texturePackingTryOfArray,
} from "./textureAtlas/packer.js";

export {
  AtlasPool,
  ATLAS_PAGE_SIZE,
  ATLAS_MAX_DIM,
  ATLAS_MAX_PAGES_PER_FORMAT,
  ATLAS_PAGE_FORMATS,
  atlasFormatIndex,
  type AtlasPage,
  type AtlasPageFormat,
  type AtlasAcquisition,
  type AtlasSource,
} from "./textureAtlas/atlasPool.js";

// Pipeline-state cache + bitfield encoding (Task 1 of derived-modes).
export {
  PipelineCache,
  encodeModeKey,
  decodeModeKey,
  MAX_ATTACHMENTS,
  DEFAULT_DESCRIPTOR,
  DEFAULT_ATTACHMENT_BLEND,
  descriptorEquals,
  type PipelineBuilder,
  type PipelineStateDescriptor,
  type AttachmentBlend,
  type BlendComponent,
  type DepthSlice,
  type StencilSlice,
  type StencilFace,
  type CullMode as PipelineCullMode,
  type FrontFace as PipelineFrontFace,
  type Topology as PipelineTopology,
} from "./pipelineCache/index.js";

export {
  ModeKeyTracker,
  snapshotDescriptor,
} from "./derivedModes/modeKeyCpu.js";

export {
  SlotTable,
  type SlotTableEntry,
} from "./derivedModes/slotTable.js";
