import { describe, test, expect } from "vitest";
import {
  SlotTable,
  PipelineCache,
  DEFAULT_DESCRIPTOR,
  encodeModeKey,
  type PipelineBuilder,
  type PipelineStateDescriptor,
} from "@aardworx/wombat.rendering/runtime";

function builder(): PipelineBuilder & { syncCalls: number; asyncCalls: number } {
  let syncCalls = 0;
  let asyncCalls = 0;
  return {
    get syncCalls()  { return syncCalls;  },
    get asyncCalls() { return asyncCalls; },
    createSync(d): GPURenderPipeline {
      syncCalls++;
      return { label: `sync#${syncCalls}#${d.cullMode}` } as unknown as GPURenderPipeline;
    },
    async createAsync(d): Promise<GPURenderPipeline> {
      asyncCalls++;
      await Promise.resolve();
      return { label: `async#${asyncCalls}#${d.cullMode}` } as unknown as GPURenderPipeline;
    },
  };
}

const D_BACK:  PipelineStateDescriptor = DEFAULT_DESCRIPTOR;
const D_FRONT: PipelineStateDescriptor = { ...DEFAULT_DESCRIPTOR, cullMode: "front" };
const D_NONE:  PipelineStateDescriptor = { ...DEFAULT_DESCRIPTOR, cullMode: "none" };

describe("derivedModes/SlotTable", () => {
  test("addRO allocates one slot per distinct modeKey, refcounts duplicates", () => {
    const t = new SlotTable(new PipelineCache(), builder());
    const a = t.addRO(D_BACK);
    const b = t.addRO(D_BACK);
    const c = t.addRO(D_FRONT);
    expect(a.slotIndex).toBe(0);
    expect(b.slotIndex).toBe(0);
    expect(b).toBe(a);
    expect(c.slotIndex).toBe(1);
    expect(t.slotCount).toBe(2);
    expect(a.refCount).toBe(2);
    expect(c.refCount).toBe(1);
  });

  test("ready() resolves after async precompile completes", async () => {
    const b = builder();
    const t = new SlotTable(new PipelineCache(), b);
    t.addRO(D_BACK);
    t.addRO(D_FRONT);
    expect(t.slotCount).toBe(2);
    // Before ready: pipelines are null (precompile in flight).
    for (const e of t.all()) expect(e.pipeline).toBeNull();
    await t.ready();
    for (const e of t.all()) expect(e.pipeline).not.toBeNull();
    expect(b.asyncCalls).toBe(2);
  });

  test("sync mode populates pipeline immediately", () => {
    const b = builder();
    const t = new SlotTable(new PipelineCache(), b);
    const e = t.addRO(D_BACK, { sync: true });
    expect(e.pipeline).not.toBeNull();
    expect(b.syncCalls).toBe(1);
  });

  test("layoutVersion bumps only on new slots", () => {
    const t = new SlotTable(new PipelineCache(), builder());
    const v0 = t.layoutVersion;
    t.addRO(D_BACK); const v1 = t.layoutVersion;
    t.addRO(D_BACK); const v2 = t.layoutVersion;
    t.addRO(D_FRONT); const v3 = t.layoutVersion;
    expect(v1).toBeGreaterThan(v0);
    expect(v2).toBe(v1);
    expect(v3).toBeGreaterThan(v2);
  });

  test("rebindRO moves refcount across slots, creating a new slot on miss", () => {
    const t = new SlotTable(new PipelineCache(), builder());
    const initial = t.addRO(D_BACK);
    expect(initial.refCount).toBe(1);
    const moved = t.rebindRO(encodeModeKey(D_BACK), D_NONE, { sync: true });
    expect(moved.slotIndex).toBe(1);
    expect(moved.refCount).toBe(1);
    // The 'back' slot lost its ref.
    expect(initial.refCount).toBe(0);
  });

  test("lookup returns null for unknown key, entry for known", () => {
    const t = new SlotTable(new PipelineCache(), builder());
    const e = t.addRO(D_BACK);
    expect(t.lookup(encodeModeKey(D_BACK))).toBe(e);
    expect(t.lookup(encodeModeKey(D_FRONT))).toBeNull();
  });

  test("ready() with no pending resolves immediately", async () => {
    const t = new SlotTable(new PipelineCache(), builder());
    await t.ready(); // no slots
    t.addRO(D_BACK, { sync: true });
    await t.ready(); // all sync, no pending
    expect(true).toBe(true);
  });
});
