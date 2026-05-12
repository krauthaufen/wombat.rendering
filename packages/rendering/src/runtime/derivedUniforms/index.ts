// §7 derived-uniforms module — public surface (v2: content-keyed rule registry,
// chain flattening, codegen-emitted uber-kernel, one dispatch).
//
// Caller responsibilities (heapScene.ts):
//
//   1. Construct one DerivedUniformsScene per heap scene (owns the rule registry,
//      the records buffer, the constituents heap + GPU buffer, the dispatcher).
//   2. In the parent scene-object's `inputChanged(t, o)` router, call
//      `scene.routeInputChanged(o)` — returns true iff `o` is a tracked trafo aval.
//   3. Per RO add: `registerRoDerivations(scene, owner, req)` — flattens + registers
//      each rule, resolves its input leaves to constituent / host slots, pushes one
//      record per derived uniform. Per RO remove: `deregisterRoDerivations(scene, reg)`.
//   4. Per frame, inside the scene's `evaluateAlways(token, …)` scope:
//        const dirty = scene.pullDirty(token);   // re-subscribes on changed trafo avals
//        scene.uploadDirty(dirty);
//        scene.encode(enc);                       // one compute dispatch
//      All O(changed); clean frames no-op.
//
// The standard trafo recipes (ModelViewProjTrafo etc.) live in ./recipes as
// `DerivedRule` constants — `STANDARD_DERIVED_RULES` / `isStandardDerivedName`.

export {
  type DerivedRule, type IRFragment,
  ruleFromIR, uniformRef, sameType, hashIR, isDerivedRule,
} from "./rule.js";

export {
  derivedUniform, DerivedExpr, type DerivedScope,
} from "./marker.js";

export {
  type RuleInput,
  flatten, inputsOf,
} from "./flatten.js";

export {
  DerivedUniformRegistry, type RuleEntry,
} from "./registry.js";

export {
  RecordsBuffer, SlotTag,
  makeHandle, handleTag, handlePayload,
  type RecordOwner,
} from "./records.js";

export {
  buildUberKernel, type UberKernel,
} from "./codegen.js";

export {
  DerivedUniformsDispatcher, uploadConstituentsRange,
  type DerivedUniformsResources,
} from "./dispatch.js";

export {
  ConstituentSlots, DF32_MAT4_BYTES,
  type SlotIndex, type PairedSlots, type SubscribeFn,
} from "./slots.js";

export {
  DerivedUniformsScene, type DerivedUniformsSceneOptions,
  registerRoDerivations, deregisterRoDerivations,
  type RoDerivedRequest, type RoRegistration,
} from "./sceneIntegration.js";

export {
  STANDARD_DERIVED_RULES, STANDARD_TRAFO_LEAVES, isStandardDerivedName,
} from "./recipes.js";
