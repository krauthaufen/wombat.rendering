// RenderTask — compile an `alist<Command>` into a runnable
// IRenderTask. The implementation walks the command list every
// frame; it is correct but unoptimised — see TODO.md for follow-ups.
//
// Per-frame flow:
//   1. Open a GPUCommandEncoder.
//   2. Set RenderContext.encoder so AdaptiveResources can encode
//      uploads / compute work during their `compute(token)`.
//   3. For each Command:
//        Render: walk the tree, lazily prepare each leaf
//                (cached per RenderObject identity), batch leaves
//                that target the same FBO into one render pass.
//        Clear:  emit clear pass.
//        Copy:   emit buffer/texture copy.
//        Custom: invoke user callback.
//   4. Submit.

import {
  RenderContext,
  type Command,
  type CompiledEffect,
  type Effect,
  type IFramebuffer,
  type IRenderTask,
  type RenderObject,
  type RenderTree,
  type FramebufferSignature,
} from "@aardworx/wombat.rendering-core";
import {
  prepareRenderObject,
  type PreparedRenderObject,
} from "@aardworx/wombat.rendering-resources";
import { clear, render, renderMany } from "@aardworx/wombat.rendering-commands";
import {
  AdaptiveToken,
  type alist,
  type aval,
} from "@aardworx/wombat.adaptive";
import { copy } from "./copy.js";

export interface RuntimeContext {
  readonly device: GPUDevice;
  /**
   * Resolves a user-facing `Effect` (the placeholder type from
   * core) to a concrete `CompiledEffect`. Until wombat.shader is
   * wired in, callers can supply an identity function and stash
   * a hand-built CompiledEffect in `RenderObject.effect`.
   */
  readonly compileEffect: (effect: Effect) => CompiledEffect;
}

class RenderTask implements IRenderTask {
  private readonly _prepared = new Map<RenderObject, PreparedRenderObject>();
  private _disposed = false;

  constructor(
    private readonly ctx: RuntimeContext,
    private readonly commands: alist<Command>,
  ) {}

  run(token: AdaptiveToken): void {
    if (this._disposed) throw new Error("RenderTask: run after dispose");
    const enc = this.ctx.device.createCommandEncoder();
    RenderContext.withEncoder(enc, () => {
      const list = this.commands.content.getValue(token);
      for (const cmdItem of list) {
        this.encodeCommand(enc, cmdItem, token);
      }
    });
    this.ctx.device.queue.submit([enc.finish()]);
  }

  dispose(): void {
    if (this._disposed) return;
    for (const p of this._prepared.values()) p.release();
    this._prepared.clear();
    this._disposed = true;
  }

  // -------------------------------------------------------------------------

  private encodeCommand(enc: GPUCommandEncoder, c: Command, token: AdaptiveToken): void {
    switch (c.kind) {
      case "Clear":  clear(enc, c.output, c.values); return;
      case "Copy":   copy(enc, c.copy); return;
      case "Custom": c.encode(enc); return;
      case "Render": this.encodeRender(enc, c.output, c.tree, token); return;
    }
  }

  private encodeRender(enc: GPUCommandEncoder, output: IFramebuffer, tree: RenderTree, token: AdaptiveToken): void {
    const leaves: PreparedRenderObject[] = [];
    this.collect(tree, leaves, output.signature, token);
    if (leaves.length === 0) return;
    if (leaves.length === 1) {
      render(enc, leaves[0]!, output, token);
    } else {
      renderMany(enc, leaves, output, token);
    }
  }

  private collect(
    tree: RenderTree,
    out: PreparedRenderObject[],
    sig: FramebufferSignature,
    token: AdaptiveToken,
  ): void {
    switch (tree.kind) {
      case "Empty": return;
      case "Leaf": out.push(this.prepFor(tree.object, sig)); return;
      case "Ordered":
      case "Unordered":
        // TODO: Unordered should sort by pipeline / bind-group / vertex-buffer
        // to minimise state changes. v0.1 emits children in argument order.
        for (const c of tree.children) this.collect(c, out, sig, token);
        return;
      case "Adaptive":
        this.collect((tree.tree as aval<RenderTree>).getValue(token), out, sig, token);
        return;
      case "OrderedFromList": {
        const list = tree.children.content.getValue(token);
        for (const c of list) this.collect(c, out, sig, token);
        return;
      }
      case "UnorderedFromSet": {
        const set = tree.children.content.getValue(token);
        // TODO: same Unordered ordering note as above.
        for (const c of set) this.collect(c, out, sig, token);
        return;
      }
    }
  }

  private prepFor(obj: RenderObject, sig: FramebufferSignature): PreparedRenderObject {
    let p = this._prepared.get(obj);
    if (p === undefined) {
      const compiled = this.ctx.compileEffect(obj.effect);
      p = prepareRenderObject(this.ctx.device, obj, compiled, sig);
      p.acquire();
      this._prepared.set(obj, p);
    }
    return p;
    // TODO: also key on `sig` — same RenderObject in two FBOs with different
    // signatures requires two PreparedRenderObjects (different pipelines).
    // Today we only store one; mismatch will assert when the pipeline's
    // color-target formats fail to match.
  }
}

export function compileRenderTask(ctx: RuntimeContext, commands: alist<Command>): IRenderTask {
  return new RenderTask(ctx, commands);
}
