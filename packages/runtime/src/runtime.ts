// Runtime — the user-facing entry point. Holds the device +
// effect-compilation hook, exposes `compile(commands)` and
// (eventually) `renderTo(...)`.

import type { Command, CompiledEffect, Effect, IRenderTask, RenderTree } from "@aardworx/wombat.rendering-core";
import type { alist } from "@aardworx/wombat.adaptive";
import { compileRenderTask, type RuntimeContext } from "./renderTask.js";
import { renderTo, type RenderToOptions, type RenderToResult } from "./renderTo.js";

export interface RuntimeOptions {
  readonly device: GPUDevice;
  readonly compileEffect: (effect: Effect) => CompiledEffect;
}

export class Runtime {
  private readonly ctx: RuntimeContext;

  constructor(opts: RuntimeOptions) {
    this.ctx = { device: opts.device, compileEffect: opts.compileEffect };
  }

  get device(): GPUDevice { return this.ctx.device; }

  /** Compile an `alist<Command>` into a runnable `IRenderTask`. */
  compile(commands: alist<Command>): IRenderTask {
    return compileRenderTask(this.ctx, commands);
  }

  /**
   * Render `scene` into a freshly-allocated framebuffer, returning
   * lazy `aval<ITexture>` handles for each color/depth attachment.
   * The framebuffer + inner render task come live when any
   * returned aval is acquired (transitively, e.g. by binding it
   * to a downstream RenderObject), and shut down on last release.
   */
  renderTo(scene: RenderTree, opts: RenderToOptions): RenderToResult {
    return renderTo(this.ctx, scene, opts);
  }
}
