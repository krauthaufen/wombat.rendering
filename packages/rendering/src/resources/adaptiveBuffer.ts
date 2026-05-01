// prepareAdaptiveBuffer — lift `aval<IBuffer>` to
// `AdaptiveResource<GPUBuffer>`.
//
// Behaviour:
//   - On `acquire` (refCount 0→1): subscribes, allocates nothing yet.
//   - On `compute(token)`:
//     - reads the source `aval<IBuffer>`;
//     - if it's `gpu`: returns the user's handle directly. We
//       don't own it; `destroy()` will not free it.
//     - if it's `host`: ensures we own a `GPUBuffer` of sufficient
//       size + correct usage flags, encodes a `writeBuffer` (via
//       device.queue) to upload the bytes, returns the owned
//       buffer. Reuses the existing buffer when capacity allows;
//       reallocates on growth.
//   - On `destroy`: frees the owned buffer (if any).
//
// `writeBuffer` does not require a GPUCommandEncoder — it goes via
// `GPUQueue.writeBuffer`, scheduled at the next submit. That keeps
// the upload path simple and avoids needing `RenderContext.encoder`
// for plain CPU→GPU buffer flow. (Compute-produced buffers will
// need the encoder; that's a different code path.)

import {
  AdaptiveResource,
  tryAcquire,
  tryRelease,
  type IBuffer,
} from "../core/index.js";
import {
  type AdaptiveToken,
  type aval,
} from "@aardworx/wombat.adaptive";
import { BufferUsage } from "./webgpuFlags.js";

export interface PrepareAdaptiveBufferOptions {
  /** Bitwise OR of GPUBufferUsage flags. `COPY_DST` is added automatically when needed. */
  readonly usage: GPUBufferUsageFlags;
  /** Optional debug label forwarded to created GPUBuffers. */
  readonly label?: string;
  /**
   * Reuse the existing GPUBuffer when the new host data fits in
   * the current allocation. Default `true`. Set `false` to force
   * a fresh buffer on every host-side change (rarely useful).
   */
  readonly reuseOnFit?: boolean;
}

class AdaptiveBuffer extends AdaptiveResource<GPUBuffer> {
  private _owned: GPUBuffer | undefined = undefined;
  private _ownedCapacity = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly source: aval<IBuffer>,
    private readonly opts: PrepareAdaptiveBufferOptions,
  ) {
    super();
  }

  protected create(): void {
    tryAcquire(this.source);
  }

  protected destroy(): void {
    if (this._owned !== undefined) {
      this._owned.destroy();
      this._owned = undefined;
      this._ownedCapacity = 0;
    }
    tryRelease(this.source);
  }

  override compute(token: AdaptiveToken): GPUBuffer {
    const src = this.source.getValue(token);
    if (src.kind === "gpu") {
      // Caller-owned buffer; drop any owned allocation we still hold.
      if (this._owned !== undefined) {
        this._owned.destroy();
        this._owned = undefined;
        this._ownedCapacity = 0;
      }
      return src.buffer;
    }
    // host — upload via queue.writeBuffer.
    const usage = this.opts.usage | BufferUsage.COPY_DST;
    const reuseOnFit = this.opts.reuseOnFit ?? true;
    const requiredSize = roundUp4(src.sizeBytes);
    if (
      this._owned === undefined
      || !reuseOnFit
      || requiredSize > this._ownedCapacity
    ) {
      if (this._owned !== undefined) this._owned.destroy();
      const desc: GPUBufferDescriptor = {
        size: requiredSize,
        usage,
        ...(this.opts.label !== undefined ? { label: this.opts.label } : {}),
      };
      this._owned = this.device.createBuffer(desc);
      this._ownedCapacity = requiredSize;
    }
    // writeBuffer accepts ArrayBuffer or any ArrayBufferView.
    const data = src.data;
    if (data instanceof ArrayBuffer) {
      this.device.queue.writeBuffer(this._owned, 0, data, 0, src.sizeBytes);
    } else {
      this.device.queue.writeBuffer(
        this._owned,
        0,
        data.buffer,
        data.byteOffset,
        src.sizeBytes,
      );
    }
    return this._owned;
  }
}

function roundUp4(n: number): number {
  return (n + 3) & ~3;
}

/**
 * Wrap an `aval<IBuffer>` as a ref-counted, adaptively-uploaded
 * `AdaptiveResource<GPUBuffer>`. The resource is itself an
 * `aval<GPUBuffer>` — downstream consumers can `getValue(token)`
 * inside their own adaptive evaluation.
 */
export function prepareAdaptiveBuffer(
  device: GPUDevice,
  source: aval<IBuffer>,
  opts: PrepareAdaptiveBufferOptions,
): AdaptiveResource<GPUBuffer> {
  return new AdaptiveBuffer(device, source, opts);
}
