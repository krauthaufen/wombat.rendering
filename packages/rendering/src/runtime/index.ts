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
  type HeapGroupShader,
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

export {
  compileHybridScene,
  type CompileHybridSceneOptions,
  type HybridScene,
} from "./hybridScene.js";
