// §7 v2 — derived-uniform records buffer (CPU-side authoring; uploaded to a GPU storage buffer).
//
// One record per derived-uniform instance (one per (RenderObject, derived name)):
//   [ rule_id, out_slot, in_slot[0], in_slot[1], … ]
// `out_slot` / each `in_slot` is a tagged 32-bit handle (see resolveSource):
//   top 3 bits = source binding, low 29 = payload (constituent slot index / drawHeader
//   byte offset).
//
// Records are flattened (chains substituted away) ⇒ every record is independent ⇒ the
// uber-kernel runs one thread per record, no levels. Layout is fixed-stride: STRIDE_U32 =
// 2 + maxArity. If a higher-arity rule shows up the stride grows and `data` is re-packed
// (rare; the kernel recompiles on that registry-version bump anyway). Fixed stride ⇒ no
// separate offsets array, trivial swap-remove. See docs/derived-uniforms-extensible.md.

/** Tagged-handle source bindings (top 3 bits of an in_slot / out_slot word). */
export const enum SlotTag {
  /** Constituent slot — a Trafo3d half in df32 storage. Payload = slot index. */
  Constituent = 0,
  /** A host uniform on the same RO — payload = its data byte offset in the main heap. Also used for the output.
   *  There is no separate "global" tag: whether a uniform is global is a per-RO fact (the sg can override it
   *  anywhere), so anything that isn't a constituent trafo resolves to this; a uniform that's the same on every
   *  RO is the same aval ⇒ the UniformPool interns it to one heap slot anyway. */
  HostHeap = 1,
  // 2..7 reserved (per-instance attribute arena, indirect df32 pair, …)
}

const TAG_SHIFT = 29;
const PAYLOAD_MASK = (1 << TAG_SHIFT) - 1; // 0x1FFFFFFF

export function makeHandle(tag: SlotTag, payload: number): number {
  if (payload < 0 || payload > PAYLOAD_MASK) {
    throw new Error(`derived records: slot payload ${payload} out of range (max ${PAYLOAD_MASK})`);
  }
  return ((tag << TAG_SHIFT) | payload) >>> 0;
}
export function handleTag(h: number): SlotTag {
  return (h >>> TAG_SHIFT) as SlotTag;
}
export function handlePayload(h: number): number {
  return h & PAYLOAD_MASK;
}

/** Owner key — typically the RenderObject (or any stable object). */
export type RecordOwner = object;

const MIN_STRIDE_U32 = 5; // rule_id, out_slot, in0, in1, in2 — covers the 13 standard recipes

export class RecordsBuffer {
  private buf: Uint32Array;
  private owners: (RecordOwner | undefined)[] = [];
  private readonly byOwner = new Map<RecordOwner, Set<number>>();
  private strideU32 = MIN_STRIDE_U32;
  private count = 0;
  /** Bumps whenever the byte layout changes shape (stride growth) so the dispatcher recompiles. */
  layoutVersion = 0;
  /** Bumps on every mutation (add / remove / stride growth) so the dispatcher knows to re-upload. */
  generation = 0;

  constructor(initialCapacityRecords = 256) {
    this.buf = new Uint32Array(Math.max(1, initialCapacityRecords) * this.strideU32);
  }

  get recordCount(): number {
    return this.count;
  }
  get strideWords(): number {
    return this.strideU32;
  }
  get strideBytes(): number {
    return this.strideU32 * 4;
  }
  /** The packed record data, exactly `recordCount * strideWords` long. Upload this. */
  get data(): Uint32Array {
    return this.buf.subarray(0, this.count * this.strideU32);
  }

  /** Append a record. Returns its index. `inSlots` length may exceed the current stride (triggers growth). */
  add(owner: RecordOwner, ruleId: number, outSlot: number, inSlots: readonly number[]): number {
    const need = 2 + inSlots.length;
    if (need > this.strideU32) this.growStride(need);
    this.ensureCapacity(this.count + 1);
    const idx = this.count++;
    const base = idx * this.strideU32;
    this.buf[base] = ruleId >>> 0;
    this.buf[base + 1] = outSlot >>> 0;
    for (let i = 0; i < inSlots.length; i++) this.buf[base + 2 + i] = inSlots[i]! >>> 0;
    for (let i = base + 2 + inSlots.length; i < base + this.strideU32; i++) this.buf[i] = 0;
    this.owners[idx] = owner;
    let set = this.byOwner.get(owner);
    if (set === undefined) {
      set = new Set();
      this.byOwner.set(owner, set);
    }
    set.add(idx);
    this.generation++;
    return idx;
  }

  /** Remove every record owned by `owner` (swap-remove against the tail). */
  removeAllForOwner(owner: RecordOwner): void {
    const set = this.byOwner.get(owner);
    if (set === undefined || set.size === 0) {
      this.byOwner.delete(owner);
      return;
    }
    // Process highest indices first so a swap can never move a not-yet-removed
    // record of this owner that we still hold a stale index for.
    const indices = [...set].sort((a, b) => b - a);
    for (const idx of indices) this.removeAt(idx);
    this.byOwner.delete(owner);
    this.generation++;
  }

  private removeAt(idx: number): void {
    const last = this.count - 1;
    // The record currently at `idx` is the one being removed.
    this.byOwner.get(this.owners[idx]!)?.delete(idx);
    if (idx !== last) {
      // Move the tail record's words into the hole, and re-home it.
      const dst = idx * this.strideU32;
      const src = last * this.strideU32;
      this.buf.copyWithin(dst, src, src + this.strideU32);
      const lastOwner = this.owners[last]!;
      const lset = this.byOwner.get(lastOwner)!;
      lset.delete(last);
      lset.add(idx);
      this.owners[idx] = lastOwner;
    }
    this.owners[last] = undefined;
    this.owners.length = last;
    this.count = last;
  }

  private ensureCapacity(records: number): void {
    const needWords = records * this.strideU32;
    if (needWords <= this.buf.length) return;
    let cap = Math.max(1, this.buf.length);
    while (cap < needWords) cap *= 2;
    const next = new Uint32Array(cap);
    next.set(this.buf);
    this.buf = next;
  }

  private growStride(newStride: number): void {
    const old = this.strideU32;
    const next = new Uint32Array(Math.max(this.buf.length, this.count * newStride));
    for (let i = 0; i < this.count; i++) {
      const o = i * old;
      const n = i * newStride;
      next.set(this.buf.subarray(o, o + old), n);
      // remaining words already zero
    }
    this.buf = next;
    this.strideU32 = newStride;
    this.layoutVersion++;
    this.generation++;
  }
}
