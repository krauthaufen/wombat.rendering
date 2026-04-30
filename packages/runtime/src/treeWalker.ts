// Walk a `RenderTree`, lazily preparing leaves through a cache,
// and encode the resulting list of `PreparedRenderObject`s into a
// single render pass on the given output. Used by both
// `RenderTask` (for `Render` commands) and `renderTo` (for the
// implicit single-pass scene render).

import type {
  CompiledEffect,
  Effect,
  FramebufferSignature,
  IFramebuffer,
  RenderObject,
  RenderTree,
} from "@aardworx/wombat.rendering-core";
import {
  prepareRenderObject,
  type PreparedRenderObject,
} from "@aardworx/wombat.rendering-resources";
import { render, renderMany } from "@aardworx/wombat.rendering-commands";
import type { AdaptiveToken, aval } from "@aardworx/wombat.adaptive";

export interface PreparedCache {
  /** Get-or-compute the prepared object for `obj` against `signature`. */
  get(obj: RenderObject, sig: FramebufferSignature, eff: CompiledEffect, device: GPUDevice): PreparedRenderObject;
  /** Release every cached entry; called on disposal. */
  releaseAll(): void;
}

export function makeCache(): PreparedCache {
  // TODO: cache key should include `(effect-id, signature)` so the same
  // RenderObject in two different FBOs produces two pipelines (different
  // color targets). Today we only key on the RenderObject identity; same
  // limitation as the RenderTask version.
  const map = new Map<RenderObject, PreparedRenderObject>();
  return {
    get(obj, sig, eff, device) {
      let p = map.get(obj);
      if (p === undefined) {
        p = prepareRenderObject(device, obj, eff, sig);
        p.acquire();
        map.set(obj, p);
      }
      return p;
    },
    releaseAll() {
      for (const p of map.values()) p.release();
      map.clear();
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
      out.push(cache.get(tree.object, sig, compileEffect(tree.object.effect), device));
      return out;
    case "Ordered":
    case "Unordered":
      // TODO: Unordered should sort by pipeline / bind-group / vertex-buffer.
      for (const c of tree.children) collectLeaves(c, sig, token, cache, compileEffect, device, out);
      return out;
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
      for (const c of set) collectLeaves(c, sig, token, cache, compileEffect, device, out);
      return out;
    }
  }
}

export function encodeTree(
  enc: GPUCommandEncoder,
  output: IFramebuffer,
  tree: RenderTree,
  token: AdaptiveToken,
  cache: PreparedCache,
  compileEffect: (e: Effect) => CompiledEffect,
  device: GPUDevice,
): void {
  const leaves = collectLeaves(tree, output.signature, token, cache, compileEffect, device);
  if (leaves.length === 0) return;
  if (leaves.length === 1) render(enc, leaves[0]!, output, token);
  else renderMany(enc, leaves, output, token);
}
