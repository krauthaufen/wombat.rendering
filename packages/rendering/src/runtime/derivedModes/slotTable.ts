// SlotTable — per-bucket (or per-scene-scope) registry that maps a
// per-RO modeKey to a fixed slot index. Each slot owns one
// `GPURenderPipeline` and one offset into a shared indirect-args
// buffer. The partition kernel reads `modeKeyToSlot` to atomic-add a
// record's counts into the right slot; encode iterates the slots and
// emits one `drawIndirect` per slot (zero-record slots draw zero).
//
// Lifecycle:
//
//   - addRO(modeKey, descriptor) → ensures the slot exists.
//     - On miss, allocates a new slot, fires `precompile` for the
//       missing descriptor (or `getOrCreateSync` if `sync=true`).
//   - removeRO(modeKey) → drops a refcount on the slot. v1 keeps the
//     slot around even at refcount=0 (pipelines are cheap to retain
//     and the future case is "RO reappears next frame").
//   - ready(): Promise<void> → resolves when every registered slot
//     has a linked pipeline.
//   - syncLookup(modeKey) → slot index or -1 (used by partition kernel
//     setup; missing slot is a logical error post-ready()).
//
// This module does NOT touch heapScene. It's a generic data structure
// that the heap integration drives. Tests use a mock builder.

import {
  PipelineCache,
  encodeModeKey as encodeMK,
  type PipelineBuilder,
  type PipelineStateDescriptor,
} from "../pipelineCache/index.js";

export interface SlotTableEntry {
  readonly slotIndex: number;
  readonly modeKey:   bigint;
  readonly descriptor: PipelineStateDescriptor;
  /** Refcount of ROs currently bound to this slot. v1 doesn't GC at 0. */
  refCount: number;
  /** Populated lazily by precompile; null until linked. */
  pipeline: GPURenderPipeline | null;
}

/**
 * One SlotTable per `(effect, textureSet)` bucket. The cache may be
 * shared across buckets — pipelines that happen to share a descriptor
 * (rare across distinct effects, since pipeline objects bake the
 * shader / layout) still dedupe at the cache level if a builder
 * legitimately produces the same GPURenderPipeline for the same
 * descriptor key.
 */
export class SlotTable {
  private readonly entries: SlotTableEntry[] = [];
  private readonly byKey   = new Map<bigint, SlotTableEntry>();
  /** Tracks pending precompile promises so ready() waits for the union. */
  private pending: Set<Promise<unknown>> = new Set();
  private version = 0;

  constructor(
    readonly cache: PipelineCache,
    readonly builder: PipelineBuilder,
  ) {}

  /** Total slot count (linked or pending). */
  get slotCount(): number { return this.entries.length; }
  /**
   * Bumps whenever a new slot is appended. Consumers (partition setup,
   * indirect-args resize, GPU lookup-table upload) compare against
   * their last-seen value to know when to re-upload.
   */
  get layoutVersion(): number { return this.version; }
  /** Iterate all slots in slot-index order. */
  *all(): IterableIterator<SlotTableEntry> { yield* this.entries; }

  /**
   * Add (or reference-bump) the slot for `descriptor`. Returns the
   * existing slot if it already exists, otherwise creates one and
   * kicks off compilation via `builder` (async by default).
   *
   * Pass `sync: true` on the runtime-mutation path to use
   * `builder.createSync` instead — the returned slot's `pipeline` will
   * be populated immediately.
   */
  addRO(
    descriptor: PipelineStateDescriptor,
    options: { sync?: boolean } = {},
  ): SlotTableEntry {
    const modeKey = encodeMK(descriptor);
    const existing = this.byKey.get(modeKey);
    if (existing !== undefined) {
      existing.refCount += 1;
      return existing;
    }
    const slotIndex = this.entries.length;
    const entry: SlotTableEntry = {
      slotIndex, modeKey, descriptor, refCount: 1, pipeline: null,
    };
    this.entries.push(entry);
    this.byKey.set(modeKey, entry);
    this.version += 1;
    if (options.sync === true) {
      entry.pipeline = this.cache.getOrCreateSync(this.builder, descriptor);
    } else {
      // Use the cache's precompile path so concurrent requests
      // deduplicate. Tracks promise so `ready()` blocks until done.
      const p = this.cache.precompile(this.builder, [descriptor]).then(() => {
        const got = this.cache.lookup(modeKey);
        if (got !== null) entry.pipeline = got;
      });
      this.pending.add(p);
      // Auto-clean once settled.
      p.finally(() => this.pending.delete(p));
    }
    return entry;
  }

  /** Drop one reference from the slot. Pipelines stay resident. */
  removeRO(modeKey: bigint): void {
    const e = this.byKey.get(modeKey);
    if (e === undefined) return;
    if (e.refCount > 0) e.refCount -= 1;
  }

  /**
   * Move an RO from one slot to another (reactive aval mutation
   * caused its modeKey to change). Returns the new (possibly newly
   * allocated) slot.
   */
  rebindRO(
    fromModeKey: bigint,
    toDescriptor: PipelineStateDescriptor,
    options: { sync?: boolean } = {},
  ): SlotTableEntry {
    this.removeRO(fromModeKey);
    return this.addRO(toDescriptor, options);
  }

  /** Sync slot lookup by modeKey. Returns the entry or null. */
  lookup(modeKey: bigint): SlotTableEntry | null {
    return this.byKey.get(modeKey) ?? null;
  }

  /**
   * Resolves when every currently-registered slot has a linked
   * pipeline. Subsequent `addRO` calls after this resolves with
   * `sync: false` will produce fresh pending work; call `ready()`
   * again to wait.
   */
  async ready(): Promise<void> {
    if (this.pending.size === 0) return;
    // Snapshot current pending set; new ones added during await
    // resolve on their own callers' awaits if needed.
    const snap = [...this.pending];
    await Promise.all(snap);
  }
}
