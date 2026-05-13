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
export class GrowBuffer {
  private buf: GPUBuffer;
  private cap: number;
  private used = 0;
  private readonly listeners = new Set<() => void>();
  constructor(
    private readonly device: GPUDevice,
    private readonly label: string,
    private readonly usage: GPUBufferUsageFlags,
    initialBytes: number,
  ) {
    this.cap = Math.max(MIN_BUFFER_BYTES, POW2(initialBytes));
    this.buf = device.createBuffer({
      size: this.cap,
      usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label,
    });
  }
  get buffer(): GPUBuffer { return this.buf; }
  get capacity(): number { return this.cap; }
  get usedBytes(): number { return this.used; }
  setUsed(n: number): void { this.used = n; }
  onResize(cb: () => void): IDisposable {
    this.listeners.add(cb);
    return { dispose: () => { this.listeners.delete(cb); } };
  }
  /** Ensure the buffer is at least `bytes` capacity. Grows by pow2 + copies live tail. */
  ensureCapacity(bytes: number): void {
    if (bytes <= this.cap) return;
    const newCap = POW2(bytes);
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
