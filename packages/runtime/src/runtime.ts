// Runtime — the user-facing entry point. Holds the device +
// effect-compilation hook, exposes `compile(commands)` and
// (eventually) `renderTo(...)`.

import type { Command, CompiledEffect, Effect, IRenderTask } from "@aardworx/wombat.rendering-core";
import type { alist } from "@aardworx/wombat.adaptive";
import { compileRenderTask, type RuntimeContext } from "./renderTask.js";

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

  // TODO: renderTo(scene, size: aval<{w,h}>): aval<ITexture>
  //   - allocate framebuffer
  //   - compile a hidden RenderTask wrapping the scene
  //   - return one of the FBO's color textures as aval<ITexture>
  //     (acquired/released by lifecycle of the returned aval)
  // The pieces are in place (allocateFramebuffer, AdaptiveResource.acquire/release,
  // compileRenderTask); just needs the gluing.
}
