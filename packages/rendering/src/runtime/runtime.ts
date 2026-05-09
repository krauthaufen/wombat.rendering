// Runtime — the user-facing entry point. Holds the device +
// effect-compilation hook, exposes `compile(commands)` and
// (eventually) `renderTo(...)`.

import type { Command, CompiledEffect, Effect, FramebufferSignature, IRenderTask, RenderTree } from "../core/index.js";
import type { alist, aval } from "@aardworx/wombat.adaptive";
import { compileRenderTask, type RuntimeContext } from "./renderTask.js";
import { renderTo, type RenderToOptions, type RenderToResult } from "./renderTo.js";

export interface RuntimeOptions {
  readonly device: GPUDevice;
  /**
   * Optional override for `Effect → CompiledEffect`. Defaults to
   * `effect.compile({ target: "wgsl" })`. Override when tests want
   * a hand-built CompiledEffect or when the caller needs custom
   * `CompileOptions` (`skipMatrixReversal`, source-file labelling,
   * …).
   */
  readonly compileEffect?: (effect: Effect, signature: FramebufferSignature) => CompiledEffect;
  /**
   * Global on/off toggle for the heap fast path. When `false`,
   * every RO routes through the legacy per-RO renderer. Reactive:
   * ROs migrate between heap and legacy subsets when the cval
   * flips. Useful for A/B perf comparisons. Default: heap on.
   */
  readonly heapEnabled?: aval<boolean>;
  /**
   * §6 family-merge bypass. When `true`, each effect lands in its
   * own bucket / shader module / pipeline (no layoutId switch
   * dispatch). Useful for A/B perf comparison against the merged
   * path. Default: merge on.
   */
  readonly disableFamilyMerge?: boolean;
}

/**
 * Derive the fragment-output layout from a framebuffer signature:
 * `colorNames[i]` ↔ `@location(i)`. The shader's `linkFragmentOutputs`
 * pass uses this to re-pin and DCE fragment outputs at compile time —
 * so two compiles of the same Effect against differently-ordered
 * signatures produce different shaders.
 */
function layoutFromSignature(sig: FramebufferSignature): { locations: ReadonlyMap<string, number> } {
  const locations = new Map<string, number>();
  sig.colorNames.forEach((name, i) => locations.set(name, i));
  return { locations };
}

export class Runtime {
  private readonly ctx: RuntimeContext;
  private readonly _tasks = new Set<IRenderTask>();
  private _disposed = false;
  private _deviceLost = false;
  /**
   * Resolves to the lost-info when the device is reported lost.
   * Stays pending while the device is alive.
   */
  readonly deviceLost: Promise<GPUDeviceLostInfo>;

  constructor(opts: RuntimeOptions) {
    this.ctx = {
      device: opts.device,
      compileEffect: opts.compileEffect ?? ((e: Effect, sig: FramebufferSignature) => e.compile({
        target: "wgsl",
        fragmentOutputLayout: layoutFromSignature(sig),
      })),
      ...(opts.heapEnabled !== undefined ? { heapEnabled: opts.heapEnabled } : {}),
      ...(opts.disableFamilyMerge === true ? { disableFamilyMerge: true } : {}),
    };
    // `device.lost` is a real-WebGPU promise; mock devices may not
    // expose it. Treat as "never lost" in that case.
    const lost = (opts.device as { lost?: Promise<GPUDeviceLostInfo> }).lost;
    this.deviceLost = lost !== undefined
      ? lost.then((info) => {
          this._deviceLost = true;
          // Best-effort: dispose all outstanding tasks so user code
          // that still holds them stops trying to encode against a
          // dead device.
          this.disposeAll();
          return info;
        })
      : new Promise(() => { /* never resolves on mock */ });
  }

  get device(): GPUDevice { return this.ctx.device; }
  get isDeviceLost(): boolean { return this._deviceLost; }

  // Recovery story (after `device.lost`):
  //
  // 1. The `lost`-handler above fires `disposeAll()`, releasing all
  //    `IRenderTask`s and the `PreparedRenderObject`s they hold.
  //    User-level avals (`cval`, `clist`, `cset`) survive — they're
  //    device-agnostic.
  // 2. Caller re-requests an adapter + device, constructs a new
  //    `Runtime` from it, and re-`compile()`s their original
  //    `alist<Command>`. The new Runtime builds fresh
  //    PreparedRenderObjects + a fresh ScenePass tree from the same
  //    user data; the per-device caches inside compile-pipeline /
  //    sampler / mip-gen are keyed on `WeakMap<GPUDevice, ...>` so
  //    they re-populate naturally for the new device.
  //
  // We don't expose a single-call `replaceDevice()` because every
  // prepared object's GPU handles are baked in at construction; the
  // cleanest path is to discard the old Runtime and construct a new
  // one from the same source `alist<Command>`.

  /**
   * Compile an `alist<Command>` into a runnable `IRenderTask`. The
   * `signature` is constant across the task's lifetime — pipelines
   * specialise against it at compile time. Framebuffer instances are
   * supplied at `task.run(framebuffer, token)`; their signatures
   * must match.
   */
  compile(signature: FramebufferSignature, commands: alist<Command>): IRenderTask {
    if (this._disposed) throw new Error("Runtime: compile after disposeAll");
    const task = compileRenderTask(this.ctx, signature, commands);
    this._tasks.add(task);
    const origDispose = task.dispose.bind(task);
    task.dispose = () => { origDispose(); this._tasks.delete(task); };
    return task;
  }

  /**
   * Render `scene` into a freshly-allocated framebuffer, returning
   * lazy `aval<ITexture>` handles for each color/depth attachment.
   * The framebuffer + inner render task come live when any
   * returned aval is acquired (transitively, e.g. by binding it
   * to a downstream RenderObject), and shut down on last release.
   */
  renderTo(scene: RenderTree, opts: RenderToOptions): RenderToResult {
    if (this._disposed) throw new Error("Runtime: renderTo after disposeAll");
    return renderTo(this.ctx, scene, opts);
  }

  /**
   * Tear down every IRenderTask compiled through this Runtime.
   * After calling this, `compile()` and `renderTo()` throw.
   * Idempotent. Intended for page-unload + fatal error paths.
   *
   * Note: doesn't destroy the GPUDevice itself — the user still
   * owns its lifecycle.
   */
  disposeAll(): void {
    if (this._disposed) return;
    this._disposed = true;
    for (const t of [...this._tasks]) {
      try { t.dispose(); } catch (e) { console.error("Runtime.disposeAll: task.dispose threw", e); }
    }
    this._tasks.clear();
  }
}
