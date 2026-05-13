import { describe, test, expect } from "vitest";
import {
  PipelineCache,
  DEFAULT_DESCRIPTOR,
  encodeModeKey,
  type PipelineBuilder,
  type PipelineStateDescriptor,
} from "@aardworx/wombat.rendering/runtime";

function mockBuilder(): PipelineBuilder & { syncCalls: number; asyncCalls: number } {
  let syncCalls = 0;
  let asyncCalls = 0;
  return {
    get syncCalls()  { return syncCalls;  },
    get asyncCalls() { return asyncCalls; },
    createSync(): GPURenderPipeline {
      syncCalls++;
      return { label: `sync#${syncCalls}` } as unknown as GPURenderPipeline;
    },
    async createAsync(): Promise<GPURenderPipeline> {
      asyncCalls++;
      // Resolve on the microtask queue so back-to-back calls overlap.
      await Promise.resolve();
      return { label: `async#${asyncCalls}` } as unknown as GPURenderPipeline;
    },
  };
}

describe("pipelineCache/cache", () => {
  test("precompile populates entries; lookup hits", async () => {
    const cache = new PipelineCache();
    const b = mockBuilder();
    const a = DEFAULT_DESCRIPTOR;
    const c: PipelineStateDescriptor = { ...DEFAULT_DESCRIPTOR, cullMode: "front" };
    await cache.precompile(b, [a, c]);
    expect(cache.size).toBe(2);
    expect(cache.lookup(encodeModeKey(a))).not.toBeNull();
    expect(cache.lookup(encodeModeKey(c))).not.toBeNull();
    expect(b.asyncCalls).toBe(2);
  });

  test("duplicate descriptors deduplicate at precompile time", async () => {
    const cache = new PipelineCache();
    const b = mockBuilder();
    await cache.precompile(b, [DEFAULT_DESCRIPTOR, DEFAULT_DESCRIPTOR, DEFAULT_DESCRIPTOR]);
    expect(cache.size).toBe(1);
    expect(b.asyncCalls).toBe(1);
  });

  test("concurrent precompile calls share one in-flight compile", async () => {
    const cache = new PipelineCache();
    const b = mockBuilder();
    const p1 = cache.precompile(b, [DEFAULT_DESCRIPTOR]);
    const p2 = cache.precompile(b, [DEFAULT_DESCRIPTOR]);
    await Promise.all([p1, p2]);
    expect(b.asyncCalls).toBe(1);
    expect(cache.size).toBe(1);
  });

  test("getOrCreateSync returns existing entry after precompile", async () => {
    const cache = new PipelineCache();
    const b = mockBuilder();
    await cache.precompile(b, [DEFAULT_DESCRIPTOR]);
    const p = cache.getOrCreateSync(b, DEFAULT_DESCRIPTOR);
    expect(p).not.toBeNull();
    expect(b.syncCalls).toBe(0);
  });

  test("getOrCreateSync hits sync path on miss", () => {
    const cache = new PipelineCache();
    const b = mockBuilder();
    const p = cache.getOrCreateSync(b, DEFAULT_DESCRIPTOR);
    expect(p).not.toBeNull();
    expect(b.syncCalls).toBe(1);
    expect(cache.lookup(encodeModeKey(DEFAULT_DESCRIPTOR))).toBe(p);
  });

  test("clear empties the cache", async () => {
    const cache = new PipelineCache();
    const b = mockBuilder();
    await cache.precompile(b, [DEFAULT_DESCRIPTOR]);
    expect(cache.size).toBe(1);
    cache.clear();
    expect(cache.size).toBe(0);
  });
});
