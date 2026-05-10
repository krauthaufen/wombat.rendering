// §7 constituent slot allocator + derivation record store.
//
// One refcounted heap of df32 mat4 slots, keyed by `aval<Trafo3d>`.
// Each acquired aval gets an adjacent (fwd, inv) slot pair on the
// constituents heap. The pull phase reads each tracked aval through
// `getValue(token)` and re-packs the slot pair only if the returned
// Trafo3d differs from the previously cached value (reference identity
// — Trafo3d is immutable).
//
// Lifetime is RO-driven: addDraw → acquire(...), removeDraw → release.

import type { aval, AdaptiveToken, IAdaptiveObject } from "@aardworx/wombat.adaptive";
import type { M44d, Trafo3d } from "@aardworx/wombat.base";

/** Slot index — pure linear index into the constituents heap. */
export type SlotIndex = number & { readonly __slotIndex: unique symbol };

/** Bytes per df32 mat4 slot: 16 entries × 2 floats × 4 bytes. */
export const DF32_MAT4_BYTES = 128;

/** Forward + inverse slot indices for one Trafo3d aval. */
export interface PairedSlots {
  readonly fwd: SlotIndex;
  readonly inv: SlotIndex;
}

/** Caller-provided hook: register the parent scene-object as a
 *  downstream output of `av`. Without this, the framework never
 *  delivers `inputChanged` and our dirty queue stays empty. */
export type SubscribeFn = (av: aval<unknown>) => void;

/** A single derivation request: "run this recipe over these constituent
 *  slots, write its output to `out_byte` in main heap." Identity-tracked
 *  so add/remove can locate the row in the records buffer. */
export interface DerivationRecord {
  readonly recipe:  number;            // RecipeId
  readonly in0:     SlotIndex;
  readonly in1:     SlotIndex;         // 0 if unused
  readonly in2:     SlotIndex;         // 0 if unused
  readonly outByte: number;
}

class IndexPool {
  private next = 0;
  private free: number[] = [];
  alloc(): number { return this.free.pop() ?? this.next++; }
  release(idx: number): void { this.free.push(idx); }
  get highWaterMark(): number { return this.next; }
}

interface ConstituentEntry {
  readonly slots: PairedSlots;
  refs: number;
}

function packEntry(mirror: Float32Array, off: number, v: number): void {
  const hi = Math.fround(v);
  mirror[off]     = hi;
  mirror[off + 1] = Math.fround(v - hi);
}

function packMat4ToSlot(mirror: Float32Array, slotIdx: number, m: M44d): void {
  const base = slotIdx * 32;
  packEntry(mirror, base + 0,  m.M00); packEntry(mirror, base + 2,  m.M01);
  packEntry(mirror, base + 4,  m.M02); packEntry(mirror, base + 6,  m.M03);
  packEntry(mirror, base + 8,  m.M10); packEntry(mirror, base + 10, m.M11);
  packEntry(mirror, base + 12, m.M12); packEntry(mirror, base + 14, m.M13);
  packEntry(mirror, base + 16, m.M20); packEntry(mirror, base + 18, m.M21);
  packEntry(mirror, base + 20, m.M22); packEntry(mirror, base + 22, m.M23);
  packEntry(mirror, base + 24, m.M30); packEntry(mirror, base + 26, m.M31);
  packEntry(mirror, base + 28, m.M32); packEntry(mirror, base + 30, m.M33);
}

export class ConstituentSlots {
  private readonly pool = new IndexPool();
  private readonly byAval = new Map<aval<Trafo3d>, ConstituentEntry>();
  private readonly dirtyAvals = new Set<aval<Trafo3d>>();
  private mirror: Float32Array;
  private readonly subscribe: SubscribeFn;

  constructor(subscribe: SubscribeFn, initialCapacity = 64) {
    this.subscribe = subscribe;
    this.mirror = new Float32Array(initialCapacity * 32);
  }

  acquire(av: aval<Trafo3d>): PairedSlots {
    let entry = this.byAval.get(av);
    if (entry === undefined) {
      const fwd = this.pool.alloc() as SlotIndex;
      const inv = this.pool.alloc() as SlotIndex;
      entry = { slots: { fwd, inv }, refs: 0 };
      this.byAval.set(av, entry);
      this.ensureCapacity(this.pool.highWaterMark);
      this.subscribe(av);
      this.dirtyAvals.add(av);
    }
    entry.refs++;
    return entry.slots;
  }

  release(av: aval<Trafo3d>): void {
    const entry = this.byAval.get(av);
    if (entry === undefined) {
      throw new Error("ConstituentSlots.release: aval was never acquired");
    }
    entry.refs--;
    if (entry.refs === 0) {
      this.pool.release(entry.slots.fwd);
      this.pool.release(entry.slots.inv);
      this.byAval.delete(av);
      this.dirtyAvals.delete(av);
    }
  }

  has(o: IAdaptiveObject): boolean {
    return this.byAval.has(o as unknown as aval<Trafo3d>);
  }
  markDirty(o: IAdaptiveObject): void {
    const av = o as unknown as aval<Trafo3d>;
    if (this.byAval.has(av)) this.dirtyAvals.add(av);
  }

  /** Drain the dirty queue: pack each marked aval's value into the
   *  mirror, return the affected slot indices. O(changed). The
   *  returned Set is OWNED by ConstituentSlots and reused frame to
   *  frame — callers must not retain it past the current frame. */
  pullDirty(token: AdaptiveToken): Set<SlotIndex> {
    this.dirtyOut.clear();
    for (const av of this.dirtyAvals) {
      const entry = this.byAval.get(av);
      if (entry === undefined) continue;
      const value = av.getValue(token);
      packMat4ToSlot(this.mirror, entry.slots.fwd, value.forward);
      packMat4ToSlot(this.mirror, entry.slots.inv, value.backward);
      this.dirtyOut.add(entry.slots.fwd);
      this.dirtyOut.add(entry.slots.inv);
    }
    this.dirtyAvals.clear();
    return this.dirtyOut;
  }
  private readonly dirtyOut = new Set<SlotIndex>();

  get cpuMirror(): Float32Array { return this.mirror; }
  get slotCount(): number { return this.pool.highWaterMark; }

  private ensureCapacity(slots: number): void {
    const need = slots * 32;
    if (need <= this.mirror.length) return;
    let cap = this.mirror.length;
    while (cap < need) cap *= 2;
    const grown = new Float32Array(cap);
    grown.set(this.mirror);
    this.mirror = grown;
  }
}
