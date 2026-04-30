// tryAcquire / tryRelease — propagate acquire/release through
// the dependency graph. If a value is an `AdaptiveResource`, drive
// its lifetime; otherwise no-op. Used by wrapper resources
// (`prepareAdaptiveBuffer`, `prepareAdaptiveTexture`, …) so that
// when a downstream consumer acquires them, any upstream
// AdaptiveResources they read also get acquired.
//
// This is what makes `renderTo` work: the returned `aval<ITexture>`
// is an AdaptiveResource holding an FBO + inner RenderTask. When
// a downstream RenderObject's prepareAdaptiveTexture acquires its
// source, that source is the renderTo result, and the FBO + task
// come live automatically.

import { AdaptiveResource } from "./adaptiveResource.js";

export function tryAcquire(av: unknown): void {
  if (av instanceof AdaptiveResource) av.acquire();
}

export function tryRelease(av: unknown): void {
  if (av instanceof AdaptiveResource) av.release();
}
