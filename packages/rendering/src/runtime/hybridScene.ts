// hybridScene — composes the heap-bucket fast path with the legacy
// per-RO path into a single render-pass renderer.
//
// Pipeline:
//   1. `flattenRenderTree(tree)` → `aset<RenderObject>` (drops intra-
//      pass ordering by contract).
//   2. `partitionA(aset, isHeapEligible)` → (heapAset, legacyAset).
//      Reactive: an RO migrates when its eligibility predicate marks
//      (e.g. an `aval<IBuffer>` flips host↔gpu).
//   3. `heapAset.map(renderObjectToHeapSpec)` → `aset<HeapDrawSpec>`,
//      fed to `buildHeapScene`. The adapter is memoized per-RO so
//      the same RO always maps to the same spec object — needed for
//      delta-driven removal to identify the right entry.
//   4. `legacyAset` wrapped as `RenderTree.unorderedFromSet(map(leaf))`
//      and fed to `ScenePass`. Same NodeWalker machinery that the
//      legacy `RenderTask` uses.
//
// Per-frame:
//   - `update(token)` runs both backends' update phases (CPU work
//     only — pool repacks, RO preparation).
//   - `encodeIntoPass(passEnc, token)` runs both backends' draw
//     emission into the caller-managed render pass.
//
// Order between the two batches inside the pass is irrelevant by
// the aset contract — the framebuffer's depth/blend ops resolve any
// apparent ordering. State changes between batches are limited to
// pipeline + bind-group switches at the boundary; both batches retain
// their internal amortization.

import {
  ASet, AVal, AdaptiveToken, type aset, type aval, type AdaptiveToken as _T,
} from "@aardworx/wombat.adaptive";
import type { CompiledEffect, Effect } from "../core/shader.js";
import type { FramebufferSignature } from "../core/framebufferSignature.js";
import { RenderTree } from "../core/renderTree.js";
import type { RenderObject } from "../core/renderObject.js";
import { ScenePass } from "./scenePass.js";
import { flattenRenderTree } from "./flattenTree.js";
import { isHeapEligible } from "./heapEligibility.js";
import { renderObjectToHeapSpec } from "./heapAdapter.js";
import { buildHeapScene, type HeapDrawSpec, type HeapScene } from "./heapScene.js";
import type { FragmentOutputLayout } from "./heapEffect.js";
import { AtlasPool } from "./textureAtlas/atlasPool.js";

void (null as AdaptiveToken | _T | null);

export interface CompileHybridSceneOptions {
  /**
   * Maps each fragment-output name an effect emits to its framebuffer
   * attachment location. Used by both backends — the heap path uses
   * it during effect introspection, the legacy path passes it to
   * `Effect.compile`. Omitted ⇒ derive from `signature.colorNames`
   * by index.
   */
  readonly fragmentOutputLayout?: FragmentOutputLayout;
  /**
   * Override `Effect → CompiledEffect`. Defaults to
   * `effect.compile({ target: "wgsl", fragmentOutputLayout })` with
   * the `fragmentOutputLayout` resolved as above. Used only by the
   * legacy path; the heap path calls `compileHeapEffect` itself.
   */
  readonly compileEffect?: (e: Effect, sig: FramebufferSignature) => CompiledEffect;
  /**
   * Global on/off switch for the heap fast path. When `false`, every
   * `RenderObject` routes through the legacy per-RO path regardless
   * of its own eligibility — equivalent to forcing `isHeapEligible`
   * to `false` for the whole scene. Reactive: flipping it migrates
   * ROs between subsets via the existing `filterA` partition.
   *
   * Use cases:
   *  - A/B perf comparisons (heap vs legacy on identical scenes).
   *  - Quick fallback if a heap-path bug shows up on a specific
   *    workload — flip the cval, ship.
   *
   * Default: `AVal.constant(true)` (heap path on for eligible ROs).
   */
  readonly heapEnabled?: aval<boolean>;
  /**
   * §6 family-merge bypass — pass-through to `BuildHeapSceneOptions`.
   * When true, each effect lands in its own bucket / shader / pipeline.
   */
  readonly disableFamilyMerge?: boolean;
}

export interface HybridScene {
  /**
   * Run both backends' CPU-side update phases. Idempotent within a
   * frame. Call before opening the render pass.
   */
  update(token: AdaptiveToken): void;
  /**
   * Encode pre-pass compute work the heap path needs (drawTable GPU
   * prefix-sum). Must be called BEFORE `beginRenderPass`. No-op when
   * the heap path has no buckets or none are dirty.
   */
  encodeComputePrep(enc: GPUCommandEncoder, token: AdaptiveToken): void;
  /**
   * Encode draws from both backends into the caller-managed render
   * pass. Heap batch first, legacy batch second — order is contract-
   * irrelevant inside one pass.
   */
  encodeIntoPass(passEnc: GPURenderPassEncoder, token: AdaptiveToken): void;
  /**
   * Cheap check for "is there anything to draw?". Lets the caller
   * skip opening the render pass entirely when both backends are
   * empty AND no clear is requested. Must be called after `update`.
   */
  hasDraws(): boolean;
  /**
   * Number of GPU buckets the heap path is currently emitting. With
   * §6 family-merge in v1 this typically collapses to 1 per
   * pipelineState. Useful for status / dev-overlay text.
   */
  heapBucketCount(): number;
  dispose(): void;
}

function defaultFragmentOutputLayout(sig: FramebufferSignature): FragmentOutputLayout {
  const locations = new Map<string, number>();
  sig.colorNames.forEach((name, i) => locations.set(name, i));
  return { locations };
}

export function compileHybridScene(
  device: GPUDevice,
  signature: FramebufferSignature,
  tree: RenderTree,
  opts: CompileHybridSceneOptions = {},
): HybridScene {
  const fragmentOutputLayout =
    opts.fragmentOutputLayout ?? defaultFragmentOutputLayout(signature);
  const compileEffect =
    opts.compileEffect ?? ((e: Effect, _sig: FramebufferSignature) =>
      e.compile({ target: "wgsl", fragmentOutputLayout }));

  // ─── Partition ───────────────────────────────────────────────────
  // Memoize the eligibility predicate per-RO so the two filterA calls
  // share one underlying observation. Without this, both calls would
  // build independent custom-avals with overlapping subscriptions.
  // The per-RO eligibility is ANDed with the global `heapEnabled`
  // toggle — flipping that off forces every RO to legacy.
  const heapEnabled = opts.heapEnabled ?? AVal.constant(true);
  const eligCache = new WeakMap<RenderObject, aval<boolean>>();
  const elig = (ro: RenderObject): aval<boolean> => {
    let av = eligCache.get(ro);
    if (av === undefined) {
      const perRO = isHeapEligible(ro);
      av = AVal.custom(t => heapEnabled.getValue(t) && perRO.getValue(t));
      eligCache.set(ro, av);
    }
    return av;
  };

  const flat = flattenRenderTree(tree);
  const heapAset   = flat.filterA(ro => elig(ro));
  const legacyAset = flat.filterA(ro => elig(ro).map(b => !b));

  // ─── Atlas pool ──────────────────────────────────────────────────
  // Per-scene Tier-S atlas pool. Owned by this hybrid scene; disposed
  // alongside the heap path. The adapter consults it to classify
  // textures into Tier S (atlas-packed) vs Tier L (standalone). MVP:
  // the adapter currently always returns Tier L (see heapAdapter
  // comments) — the pool is wired here so the next PR can add Tier-S
  // classification without touching this file.
  const atlasPool = new AtlasPool(device);

  // ─── Heap subset → HeapDrawSpec aset ─────────────────────────────
  // Memoize the adapter: aset removal must identify the SAME spec
  // object that addition produced; the underlying HashSet uses
  // identity/equality and HeapDrawSpec is plain (no structural hash).
  // WeakMap keyed by RO ensures the same RO always maps to the same
  // spec instance, even if the RO migrates out and back in.
  const specCache = new WeakMap<RenderObject, HeapDrawSpec>();
  const heapSpecAset = heapAset.map((ro: RenderObject) => {
    let spec = specCache.get(ro);
    if (spec === undefined) {
      spec = renderObjectToHeapSpec(ro, AdaptiveToken.top, atlasPool);
      specCache.set(ro, spec);
    }
    return spec;
  });

  const heapScene: HeapScene = buildHeapScene(device, signature, heapSpecAset, {
    fragmentOutputLayout,
    atlasPool,
    ...(opts.disableFamilyMerge === true ? { disableFamilyMerge: true } : {}),
  });

  // ─── Legacy subset → RenderTree → ScenePass ──────────────────────
  // The unordered-from-set wrapper drops index/order info; the legacy
  // path inherits the same "no order inside a pass" contract. Re-using
  // the existing NodeWalker machinery means RO preparation, caching,
  // and resource ref-counting all behave exactly as the master path.
  const legacyTree: RenderTree =
    RenderTree.unorderedFromSet(legacyAset.map(ro => RenderTree.leaf(ro)));
  const scenePass = new ScenePass(device, signature, legacyTree, compileEffect);

  return {
    update(token: AdaptiveToken): void {
      heapScene.update(token);
      scenePass.update(token);
    },
    encodeComputePrep(enc: GPUCommandEncoder, token: AdaptiveToken): void {
      heapScene.encodeComputePrep(enc, token);
    },
    encodeIntoPass(passEnc: GPURenderPassEncoder, token: AdaptiveToken): void {
      heapScene.encodeIntoPass(passEnc);
      scenePass.encodeIntoPass(passEnc, token);
    },
    hasDraws(): boolean {
      // Heap path tracks count in stats; legacy path's leaf count is
      // a cheap walker-tree scan when empty (collect into a fresh
      // array). The sentinel here is "both empty" — caller decides
      // whether a clear-only pass is still worth opening.
      if (heapScene.stats.totalDraws > 0) return true;
      return scenePass.collect().length > 0;
    },
    heapBucketCount(): number {
      return heapScene.stats.groups;
    },
    dispose(): void {
      heapScene.dispose();
      scenePass.dispose();
      atlasPool.dispose();
    },
  };
}

// `aset` reference keeps the import live for documentation comments.
const _asetTypeGuard: aset<unknown> | undefined = undefined;
void _asetTypeGuard;
// Same for ASet: re-exported here as a hint (the helper uses ASet.* internally).
void ASet;
