// Walk a `RenderTree`, lazily preparing leaves through a cache,
// and encode the resulting list of `PreparedRenderObject`s into a
// single render pass on the given output. Used by both
// `RenderTask` (for `Render` commands) and `renderTo` (for the
// implicit single-pass scene render).

import type {
  ClearValues,
  CompiledEffect,
  Effect,
  FramebufferSignature,
  IFramebuffer,
  RenderObject,
  RenderTree,
} from "@aardworx/wombat.rendering-core";
// `Effect.id` is the wombat.shader build-time stable hash; we use
// it as the strong identity for the pipeline cache key.
import {
  prepareRenderObject,
  type PreparedRenderObject,
} from "@aardworx/wombat.rendering-resources";
import { render, renderMany } from "@aardworx/wombat.rendering-commands";
import type { AdaptiveToken, aval } from "@aardworx/wombat.adaptive";

export interface PreparedCache {
  /**
   * Get-or-compute the prepared object for `obj` rendered into a
   * framebuffer of shape `sig`. Keyed on
   * `(RenderObject, FramebufferSignature)` because the pipeline's
   * color-target formats depend on the signature.
   *
   * Aardvark.Rendering's ResourceManager keys equivalent caches on
   * `(IRenderObject, IFramebufferSignature)`; same structure here.
   */
  get(
    obj: RenderObject,
    sig: FramebufferSignature,
    eff: CompiledEffect,
    effectId: string,
    device: GPUDevice,
  ): PreparedRenderObject;
  releaseAll(): void;
}

export function makeCache(): PreparedCache {
  const outer = new Map<RenderObject, Map<FramebufferSignature, PreparedRenderObject>>();
  return {
    get(obj, sig, eff, effectId, device) {
      let bySig = outer.get(obj);
      if (bySig === undefined) {
        bySig = new Map();
        outer.set(obj, bySig);
      }
      let p = bySig.get(sig);
      if (p === undefined) {
        p = prepareRenderObject(device, obj, eff, sig, { effectId });
        p.acquire();
        bySig.set(sig, p);
      }
      return p;
    },
    releaseAll() {
      for (const bySig of outer.values()) {
        for (const p of bySig.values()) p.release();
      }
      outer.clear();
    },
  };
}

export function collectLeaves(
  tree: RenderTree,
  sig: FramebufferSignature,
  token: AdaptiveToken,
  cache: PreparedCache,
  compileEffect: (e: Effect) => CompiledEffect,
  device: GPUDevice,
  out: PreparedRenderObject[] = [],
): PreparedRenderObject[] {
  switch (tree.kind) {
    case "Empty": return out;
    case "Leaf":
      out.push(cache.get(
        tree.object, sig,
        compileEffect(tree.object.effect),
        tree.object.effect.id,
        device,
      ));
      return out;
    case "Ordered":
      for (const c of tree.children) collectLeaves(c, sig, token, cache, compileEffect, device, out);
      return out;
    case "Unordered": {
      // Collect into a fresh list so we can sort just THIS subtree's
      // leaves without disturbing surrounding Ordered context, then
      // append the sorted result.
      const local: PreparedRenderObject[] = [];
      for (const c of tree.children) collectLeaves(c, sig, token, cache, compileEffect, device, local);
      sortByState(local);
      out.push(...local);
      return out;
    }
    case "Adaptive":
      collectLeaves((tree.tree as aval<RenderTree>).getValue(token), sig, token, cache, compileEffect, device, out);
      return out;
    case "OrderedFromList": {
      const list = tree.children.content.getValue(token);
      for (const c of list) collectLeaves(c, sig, token, cache, compileEffect, device, out);
      return out;
    }
    case "UnorderedFromSet": {
      const set = tree.children.content.getValue(token);
      const local: PreparedRenderObject[] = [];
      for (const c of set) collectLeaves(c, sig, token, cache, compileEffect, device, local);
      sortByState(local);
      out.push(...local);
      return out;
    }
  }
}

// Per-handle identity counter used to give pipelines / bind-group
// layouts stable numeric ranks for the Unordered sort. We don't
// have direct access to the underlying GPU handle's identity, but
// the PreparedRenderObject's `pipeline` reference IS stable across
// frames (the pipeline cache returns the same handle for the same
// effect+sig+state), so sorting by pipeline reference is a robust
// proxy.
const sortRanks = new WeakMap<object, number>();
let nextSortRank = 1;
function rankOf(o: object): number {
  let r = sortRanks.get(o);
  if (r === undefined) { r = nextSortRank++; sortRanks.set(o, r); }
  return r;
}

/**
 * Sort `Unordered` / `UnorderedFromSet` children to minimise GPU
 * state changes: pipeline first, then group-0 layout (uniforms /
 * textures / samplers usually live there), then vertex buffers.
 * The exact heuristic is allowed to change as long as adjacent
 * leaves share more state than a random ordering would.
 */
function sortByState(leaves: PreparedRenderObject[]): void {
  leaves.sort((a, b) => {
    const pa = rankOf(a.pipeline);
    const pb = rankOf(b.pipeline);
    if (pa !== pb) return pa - pb;
    const la = a.groups[0]?.layout;
    const lb = b.groups[0]?.layout;
    if (la !== undefined && lb !== undefined && la !== lb) {
      return rankOf(la) - rankOf(lb);
    }
    return 0;
  });
}

export function encodeTree(
  enc: GPUCommandEncoder,
  output: IFramebuffer,
  tree: RenderTree,
  token: AdaptiveToken,
  cache: PreparedCache,
  compileEffect: (e: Effect) => CompiledEffect,
  device: GPUDevice,
  clearValues?: ClearValues,
): void {
  const leaves = collectLeaves(tree, output.signature, token, cache, compileEffect, device);
  if (leaves.length === 0) {
    // Empty tree but `clearValues` set → still need a pass that
    // clears. Fall through to `render(empty)`-style: a single
    // begin/end pass with the requested loadOps. Easiest to do
    // via clear() — the runtime walker hits this path only when
    // a Clear had no following Render to fuse with.
    return;
  }
  if (leaves.length === 1) render(enc, leaves[0]!, output, token, clearValues);
  else renderMany(enc, leaves, output, token, clearValues);
}
