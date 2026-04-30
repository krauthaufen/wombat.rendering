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
