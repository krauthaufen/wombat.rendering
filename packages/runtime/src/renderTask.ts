// RenderTask — compile an `alist<Command>` into a runnable
// IRenderTask. The implementation walks the command list every
// frame; it is correct but unoptimised — see TODO.md for follow-ups.

import {
  RenderContext,
  type Command,
  type CompiledEffect,
  type Effect,
  type IFramebuffer,
  type IRenderTask,
  type RenderTree,
} from "@aardworx/wombat.rendering-core";
import { clear } from "@aardworx/wombat.rendering-commands";
import {
  AdaptiveToken,
  type alist,
} from "@aardworx/wombat.adaptive";
import { copy } from "./copy.js";
import { encodeTree, makeCache, type PreparedCache } from "./treeWalker.js";

export interface RuntimeContext {
  readonly device: GPUDevice;
  readonly compileEffect: (effect: Effect) => CompiledEffect;
}

class RenderTask implements IRenderTask {
  private readonly cache: PreparedCache = makeCache();
  private _disposed = false;

  constructor(
    private readonly ctx: RuntimeContext,
    private readonly commands: alist<Command>,
  ) {}

  run(token: AdaptiveToken): void {
    if (this._disposed) throw new Error("RenderTask: run after dispose");
    const enc = this.ctx.device.createCommandEncoder();
    this.encode(enc, token);
    this.ctx.device.queue.submit([enc.finish()]);
  }

  /**
   * Encode commands into the given encoder without finishing or
   * submitting. Used by `renderTo` to compose an inner task into
   * an outer frame's encoder.
   */
  encode(enc: GPUCommandEncoder, token: AdaptiveToken): void {
    if (this._disposed) throw new Error("RenderTask: encode after dispose");
    RenderContext.withEncoder(enc, () => {
      const list = this.commands.content.getValue(token);
      for (const cmdItem of list) this.encodeCommand(enc, cmdItem, token);
    });
  }

  dispose(): void {
    if (this._disposed) return;
    this.cache.releaseAll();
    this._disposed = true;
  }

  private encodeCommand(enc: GPUCommandEncoder, c: Command, token: AdaptiveToken): void {
    switch (c.kind) {
      case "Clear":  clear(enc, c.output, c.values); return;
      case "Copy":   copy(enc, c.copy); return;
      case "Custom": c.encode(enc); return;
      case "Render": this.encodeRender(enc, c.output, c.tree, token); return;
    }
  }

  private encodeRender(enc: GPUCommandEncoder, output: IFramebuffer, tree: RenderTree, token: AdaptiveToken): void {
    encodeTree(enc, output, tree, token, this.cache, this.ctx.compileEffect, this.ctx.device);
  }
}

export function compileRenderTask(ctx: RuntimeContext, commands: alist<Command>): IRenderTask & {
  encode(enc: GPUCommandEncoder, token: AdaptiveToken): void;
} {
  return new RenderTask(ctx, commands);
}
