// CPU-side GPURenderPipeline cache keyed on the bitfield-encoded
// modeKey from bitfield.ts.
//
// Two access modes:
//
//   - `precompile(builder, descriptors[])` — async batch warm. Used at
//     scene-build with the enumerated pipeline domain so frame 0 sees a
//     hot cache. Concurrent calls for the same modeKey deduplicate
//     against the in-flight pipeline.
//
//   - `getOrCreateSync(builder, descriptor)` — sync. Used on the
//     runtime-mutation path when a never-before-seen descriptor
//     appears. The returned `GPURenderPipeline` is usable immediately
//     (driver compile happens in the background; first submit using it
//     may stall the GPU queue if compile isn't done).
//
// The cache is layered on a user-supplied `PipelineBuilder` — the
// adapter that turns a `PipelineStateDescriptor` + the bucket's
// effect / vertex layout / bind group layout / framebuffer signature
// into the full `GPURenderPipelineDescriptor` and calls
// `device.createRenderPipeline{,Async}`.  This module knows nothing
// about effect compilation; it just dedupes by modeKey and bookkeeps.

import { encodeModeKey } from "./bitfield.js";
import type { PipelineStateDescriptor } from "./descriptor.js";

/** Builder hook: turn a descriptor into a real pipeline. */
export interface PipelineBuilder {
  /** Sync — pipeline returned immediately; compile may continue in BG. */
  createSync(descriptor: PipelineStateDescriptor): GPURenderPipeline;
  /** Async — Promise resolves when compile + link finish. */
  createAsync(descriptor: PipelineStateDescriptor): Promise<GPURenderPipeline>;
}

interface CacheEntry {
  readonly modeKey: bigint;
  readonly descriptor: PipelineStateDescriptor;
  pipeline?: GPURenderPipeline;
  pending?: Promise<GPURenderPipeline>;
}

export class PipelineCache {
  private readonly entries = new Map<bigint, CacheEntry>();

  /** Number of populated (or in-flight) entries. */
  get size(): number { return this.entries.size; }

  /** Sync lookup. Returns null if not in cache. */
  lookup(modeKey: bigint): GPURenderPipeline | null {
    const e = this.entries.get(modeKey);
    return e?.pipeline ?? null;
  }

  /** True if any pipeline has been registered for this key (even pending). */
  has(modeKey: bigint): boolean {
    return this.entries.has(modeKey);
  }

  /**
   * Sync create-or-fetch. The returned pipeline is usable immediately;
   * driver compile may still be in flight. First GPU submission using
   * it may queue-stall while compile completes.
   */
  getOrCreateSync(
    builder: PipelineBuilder,
    descriptor: PipelineStateDescriptor,
  ): GPURenderPipeline {
    const key = encodeModeKey(descriptor);
    const existing = this.entries.get(key);
    if (existing?.pipeline) return existing.pipeline;
    if (existing && !existing.pipeline && existing.pending) {
      // An async create is in flight. Bridge to a sync object by
      // calling createSync — we still produce a usable pipeline now;
      // the async one will resolve later (and be redundant, but
      // harmless — the second pipeline object simply isn't used by
      // anything past this point if we wire callers to read from
      // `entry.pipeline`).
      const pipeline = builder.createSync(descriptor);
      existing.pipeline = pipeline;
      return pipeline;
    }
    const pipeline = builder.createSync(descriptor);
    this.entries.set(key, { modeKey: key, descriptor, pipeline });
    return pipeline;
  }

  /**
   * Async precompile a batch of descriptors. Resolves when every
   * pipeline has linked. Idempotent: duplicate descriptors (by
   * modeKey) share one compile.
   */
  async precompile(
    builder: PipelineBuilder,
    descriptors: readonly PipelineStateDescriptor[],
  ): Promise<void> {
    const promises: Promise<unknown>[] = [];
    for (const d of descriptors) {
      const key = encodeModeKey(d);
      const existing = this.entries.get(key);
      if (existing?.pipeline) continue;            // already linked
      if (existing?.pending)  { promises.push(existing.pending); continue; }
      const pending = builder.createAsync(d).then((pipeline) => {
        const e = this.entries.get(key);
        if (e) { e.pipeline = pipeline; delete e.pending; }
        return pipeline;
      });
      this.entries.set(key, { modeKey: key, descriptor: d, pending });
      promises.push(pending);
    }
    await Promise.all(promises);
  }

  /** Iterate every entry with a linked pipeline. */
  *ready(): IterableIterator<readonly [bigint, GPURenderPipeline, PipelineStateDescriptor]> {
    for (const [k, e] of this.entries) {
      if (e.pipeline) yield [k, e.pipeline, e.descriptor];
    }
  }

  /** All registered descriptors (linked or pending). Used by manifest. */
  *all(): IterableIterator<readonly [bigint, PipelineStateDescriptor]> {
    for (const [k, e] of this.entries) yield [k, e.descriptor];
  }

  /** Forget everything. Caller is responsible for dropping pipeline refs. */
  clear(): void {
    this.entries.clear();
  }
}
