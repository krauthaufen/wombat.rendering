// AdaptiveResource<T> — a ref-counted, adaptively-updated GPU
// handle that **is itself an `aval<T>`**. Lifted directly from
// Aardvark.Base's `AdaptiveResource<'a>`.
//
// The shape lets `runtime.renderTo(scene, size) → aval<ITexture>`
// work the way Aardvark.Rendering's `RenderTask.renderTo` works:
// a downstream consumer treats it like any other `aval<ITexture>`,
// the FBO is allocated lazily on first `acquire`, and torn down
// when the last consumer releases.
//
// Lifecycle:
//   acquire()  — refCount++. On 0→1 transition, calls `create()`,
//                which is where one-time setup happens (allocate
//                an FBO, register input subscriptions, etc.).
//   getValue(token) — the standard aval contract. Calls
//                `compute(token)`, which is where adaptive work
//                lives (re-render scene if size or scene changed,
//                return the texture handle). May encode commands
//                on the currently-active `RenderContext.encoder`.
//   release()  — refCount--. On 1→0 transition, calls `destroy()`,
//                which frees the GPU resources.
//
// Subscribing to value changes (`addCallback`) does **not** acquire
// — acquisition is an explicit lifetime concern and a subscription
// alone does not justify keeping GPU memory live. The runtime's
// RenderTask compilation walks its referenced AdaptiveResources
// and acquires them at task-construction time, releases them on
// task disposal.

import {
  AbstractVal,
  type AdaptiveToken,
} from "@aardworx/wombat.adaptive";

export abstract class AdaptiveResource<T> extends AbstractVal<T> {
  private _refCount = 0;

  /** Allocate one-time GPU resources. Called on the 0→1 acquire. */
  protected abstract create(): void;
  /** Free GPU resources. Called on the 1→0 release. */
  protected abstract destroy(): void;
  /**
   * The standard aval contract: produce the current handle for
   * this resource. May read other adaptive inputs via `token` and
   * may encode work onto `RenderContext.encoder` if one is active
   * (uploads, compute dispatches, render-to-texture, ...).
   */
  abstract override compute(token: AdaptiveToken): T;

  acquire(): void {
    this._refCount++;
    if (this._refCount === 1) this.create();
  }

  release(): void {
    if (this._refCount <= 0) {
      throw new Error("AdaptiveResource: release without matching acquire");
    }
    this._refCount--;
    if (this._refCount === 0) this.destroy();
  }

  /** True iff at least one consumer holds an acquire. */
  get isActive(): boolean { return this._refCount > 0; }
  get refCount(): number { return this._refCount; }

  /**
   * Derive a new `AdaptiveResource<R>` whose lifetime forwards to
   * `this`. Consumers of the derived resource transitively
   * acquire/release `this`. Use this to expose structured outputs
   * of a complex resource (e.g. the colour textures of a `renderTo`
   * framebuffer) as individual `aval<ITexture>`s.
   */
  derive<R>(project: (t: T) => R): AdaptiveResource<R> {
    return new DerivedAdaptiveResource<T, R>(this, project);
  }
}

class DerivedAdaptiveResource<T, R> extends AdaptiveResource<R> {
  constructor(private readonly parent: AdaptiveResource<T>, private readonly project: (t: T) => R) {
    super();
  }
  protected override create(): void { this.parent.acquire(); }
  protected override destroy(): void { this.parent.release(); }
  override compute(token: AdaptiveToken): R {
    return this.project(this.parent.getValue(token));
  }
}
