// GrowBuffer — pow2-grown GPUBuffer with a high-water mark and
// resize-listener machinery for dependents (bind groups, mostly).
//
// On grow, a fresh buffer is created at the next pow2 capacity, the
// live tail is copied over via copyBufferToBuffer, and the listeners
// (registered via `onResize`) are invoked. The old buffer is
// destroyed.

import type { IDisposable } from "@aardworx/wombat.adaptive";

export const MIN_BUFFER_BYTES = 64 * 1024;

export const POW2 = (n: number): number => {
  let p = 1; while (p < n) p <<= 1; return p;
};

export const ALIGN16 = (n: number) => (n + 15) & ~15;

/**
 * A GPUBuffer that can grow to next power-of-two on demand. On grow,
 * a fresh buffer is created at the new size, the live tail copied
 * over via copyBufferToBuffer, and dependents (bind groups, mostly)
 * are notified to rebuild via the `onResize` callback.
 *
 * `usedBytes` is the high-water mark — the runtime advances this as
 * it allocates, and `ensureCapacity` grows when required. This
 * separates allocation policy from grow policy.
 */
/**
 * Default chunk capacity cap when none is provided. We pick 256 MB as
 * a safety ceiling that's well within typical adapter
 * `maxStorageBufferBindingSize` (WebGPU spec minimum is 128 MB; most
 * desktop GPUs expose 2 GiB+, mobile/integrated ≥ 256 MB). Callers
 * that know the adapter limit should pass it explicitly so chunks
 * can grow to whatever the hardware supports — see §3 in
 * `docs/heap-future-work.md` and `runtime/heapScene.ts`'s scene
 * factory where the device limit is threaded in.
 */
export const DEFAULT_MAX_BUFFER_BYTES = 256 * 1024 * 1024;

export class GrowBuffer {
  private buf: GPUBuffer;
  private cap: number;
  private used = 0;
  private readonly listeners = new Set<() => void>();
  private readonly maxBytes: number;
  constructor(
    private readonly device: GPUDevice,
    private readonly label: string,
    private readonly usage: GPUBufferUsageFlags,
    initialBytes: number,
    maxBytes: number = DEFAULT_MAX_BUFFER_BYTES,
  ) {
    this.cap = Math.max(MIN_BUFFER_BYTES, POW2(initialBytes));
    this.maxBytes = Math.max(this.cap, maxBytes);
    if (this.cap > this.maxBytes) {
      throw new Error(
        `GrowBuffer '${label}': initial capacity ${this.cap} exceeds maxBytes ${this.maxBytes}`,
      );
    }
    this.buf = device.createBuffer({
      size: this.cap,
      usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label,
    });
  }
  get buffer(): GPUBuffer { return this.buf; }
  get capacity(): number { return this.cap; }
  get maxCapacity(): number { return this.maxBytes; }
  get usedBytes(): number { return this.used; }
  setUsed(n: number): void { this.used = n; }
  onResize(cb: () => void): IDisposable {
    this.listeners.add(cb);
    return { dispose: () => { this.listeners.delete(cb); } };
  }
  /**
   * Ensure the buffer is at least `bytes` capacity. Grows by pow2 +
   * copies live tail. Throws if `bytes` exceeds the configured
   * `maxBytes` cap — caller is expected to either open a new chunk
   * (full §3 multi-draw-call path, deferred) or reject the
   * allocation upstream (see `isHeapEligible`'s §2 large-object
   * eject, which keeps multi-MB single ROs out of the heap arena
   * entirely).
   */
  ensureCapacity(bytes: number): void {
    if (bytes <= this.cap) return;
    if (bytes > this.maxBytes) {
      throw new Error(
        `GrowBuffer '${this.label}': requested ${bytes} bytes exceeds maxBytes ${this.maxBytes}. ` +
        `This is the adapter's per-binding storage limit — landing larger workloads needs the ` +
        `multi-chunk path (§3 in docs/heap-future-work.md) or upstream eviction via §2.`,
      );
    }
    let newCap = POW2(bytes);
    if (newCap > this.maxBytes) newCap = this.maxBytes;
    const newBuf = this.device.createBuffer({
      size: newCap,
      usage: this.usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: this.label,
    });
    if (this.used > 0) {
      const enc = this.device.createCommandEncoder({ label: `${this.label}: grow-copy` });
      enc.copyBufferToBuffer(this.buf, 0, newBuf, 0, ALIGN16(this.used));
      this.device.queue.submit([enc.finish()]);
    }
    this.buf.destroy();
    this.buf = newBuf;
    this.cap = newCap;
    for (const cb of this.listeners) cb();
  }
  destroy(): void { this.buf.destroy(); }
}
