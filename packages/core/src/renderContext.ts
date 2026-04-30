// RenderContext — ambient state available to AdaptiveResources
// during evaluation. The runtime sets `encoder` before walking a
// task's resources so that `compute(token)` calls inside an
// AdaptiveResource can encode commands (uploads, compute passes,
// render-to-texture) onto the current frame's encoder.
//
// Outside a frame (`encoder === null`) AdaptiveResources should
// either return their cached handle or throw — they cannot do
// GPU work without an encoder.

interface RenderContextSlot {
  encoder: GPUCommandEncoder | null;
  withEncoder<R>(encoder: GPUCommandEncoder, fn: () => R): R;
  requireEncoder(): GPUCommandEncoder;
}

export const RenderContext: RenderContextSlot = {
  encoder: null,

  /**
   * Run `fn` with `encoder` exposed as the current frame encoder.
   * Restores the previous value on exit. Nestable.
   */
  withEncoder<R>(encoder: GPUCommandEncoder, fn: () => R): R {
    const prev = RenderContext.encoder;
    RenderContext.encoder = encoder;
    try {
      return fn();
    } finally {
      RenderContext.encoder = prev;
    }
  },

  /**
   * Convenience: throws if no encoder is active. Use inside
   * `compute(token)` of an AdaptiveResource that genuinely needs
   * to encode work.
   */
  requireEncoder(): GPUCommandEncoder {
    if (RenderContext.encoder === null) {
      throw new Error(
        "RenderContext: no active encoder. AdaptiveResource.compute() " +
        "was called outside RenderContext.withEncoder(...).",
      );
    }
    return RenderContext.encoder;
  },
};
