// §7 derived-uniforms module — public surface.
//
// Caller responsibilities (typically heapScene.ts):
//
//   1. Construct one DerivedUniformsScene per heap scene, one
//      DerivedUniformsBucket per render bucket. The scene owns the
//      shared constituents storage buffer + dirty bitmask + pipeline.
//   2. In the parent scene-object's `inputChanged(t, o)` router, call
//      `scene.routeInputChanged(o)`. Returns true iff o matched.
//   3. In each RO's add path: `registerRoDerivations(scene, bucket, req)`
//      allocates constituent slots and pushes records into the bucket's
//      records buffer. In remove path: `deregisterRoDerivations(...)`.
//   4. In the per-frame compute prep, inside the parent sceneObj's
//      `evaluateAlways(token, ...)` scope:
//        const dirty = scene.constituents.pullDirty(token);
//        scene.uploadDirty(dirty);
//        for each bucket: bucket.dispatcher.encode(enc);
//      All steps are O(changed); on clean frames they no-op.
//
// Identifying which uniform names are §7-owned: DERIVED_UNIFORM_NAMES.

export {
  RecipeId,
  ALL_RECIPES, DERIVED_UNIFORM_NAMES,
  recipeIdByName, recipeName, recipeInputs, recipeInputCount, recipeOutput,
  type ConstituentRef, type RecipeOutput,
} from "./recipes.js";

export {
  ConstituentSlots, DF32_MAT4_BYTES,
  type SlotIndex, type PairedSlots, type DerivationRecord, type SubscribeFn,
} from "./slots.js";

export {
  DerivedUniformsDispatcher, DerivedUniformsPipeline,
  RecordsBuffer, uploadConstituentsRange,
  type DerivedUniformsResources,
} from "./dispatch.js";

export {
  DerivedUniformsScene,
  registerRoDerivations,
  deregisterRoDerivations,
  isDerivedUniformName,
  type RoTrafoInputs,
  type RoDerivedRequest,
  type RoRegistration,
} from "./sceneIntegration.js";

export {
  DERIVED_UBER_KERNEL_WGSL,
} from "./uberKernel.wgsl.js";
