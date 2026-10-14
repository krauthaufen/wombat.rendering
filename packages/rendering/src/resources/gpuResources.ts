// Device-owned GPU resources that are updated IN PLACE, plus the
// region-write / readback helpers they are built on.
//
// The adaptive preparers (`prepareAdaptiveTexture` / `Buffer`) always
// replace a resource's whole contents from a host value. Streaming
// workloads (a brick atlas written one brick at a time, a per-frame
// cell list, compute scratch buffers) need the opposite: a persistent
// allocation that is written region by region and never re-uploaded.
//
//   - `writeTextureRegion` / `writeBufferRange` — `queue.write*` with the
//     row pitches / byte ranges derived from the texture format.
//   - `readBuffer` / `readTextureRegion` — copy into a staging buffer,
//     map it and hand back a tightly packed `ArrayBuffer`.
//   - `MutableTexture` / `MutableBuffer` — an owned `GPUTexture` /
//     `GPUBuffer` that IS an `aval<ITexture>` / `aval<IBuffer>`, so it
//     binds anywhere a texture/buffer aval goes (render objects, scene
//     graph uniforms, storage buffers). Writes go straight to the queue;
//     by default each write also marks the aval (coalesced into one mark
//     per microtask) so a render loop that sleeps until something
//     changes draws the new contents. Pass `notify: false` for writes
//     made during frame evaluation (they are picked up by the frame
//     being encoded anyway).
//
// These resources belong to ONE device: after a device loss they are
// dead and have to be recreated (and refilled) on the new device.

import {
  AbstractVal, markOutdated, transact,
  type AdaptiveToken,
} from "@aardworx/wombat.adaptive";
import { IBuffer, ITexture } from "../core/index.js";
import { blockInfoFor } from "./adaptiveTexture.js";
import { BufferUsage, TextureUsage } from "./webgpuFlags.js";

/** Host data accepted by the write helpers. */
export type HostData = ArrayBuffer | ArrayBufferView;

/** A box of texels in one mip level. Arrays are `[x, y, z]` / `[w, h, d]`; missing entries default. */
export interface TextureRegion {
  /** Texel origin. Default `[0, 0, 0]`. */
  readonly origin?: readonly number[];
  /** Texel extent. Default: from `origin` to the end of the mip level. */
  readonly size?: readonly number[];
  readonly mipLevel?: number;
  /** Row pitch of the HOST data in bytes. Default: tightly packed. */
  readonly bytesPerRow?: number;
  /** Rows per 2D image of the HOST data. Default: tightly packed. */
  readonly rowsPerImage?: number;
}

function mipExtent(texture: GPUTexture, level: number): [number, number, number] {
  const w = Math.max(1, texture.width >> level);
  const h = texture.dimension === "1d" ? 1 : Math.max(1, texture.height >> level);
  // array layers do not shrink with the mip level, 3D depth does
  const d = texture.dimension === "3d"
    ? Math.max(1, texture.depthOrArrayLayers >> level)
    : texture.depthOrArrayLayers;
  return [w, h, d];
}

interface ResolvedRegion {
  readonly mipLevel: number;
  readonly origin: [number, number, number];
  readonly size: [number, number, number];
  /** tightly packed host layout */
  readonly bytesPerRow: number;
  readonly rowsPerImage: number;
}

function resolveRegion(texture: GPUTexture, region: TextureRegion): ResolvedRegion {
  const mipLevel = region.mipLevel ?? 0;
  const full = mipExtent(texture, mipLevel);
  const origin: [number, number, number] = [
    region.origin?.[0] ?? 0, region.origin?.[1] ?? 0, region.origin?.[2] ?? 0,
  ];
  const size: [number, number, number] = [
    region.size?.[0] ?? full[0] - origin[0],
    region.size?.[1] ?? full[1] - origin[1],
    region.size?.[2] ?? full[2] - origin[2],
  ];
  const block = blockInfoFor(texture.format);
  return {
    mipLevel, origin, size,
    bytesPerRow: Math.ceil(size[0] / block.width) * block.bytesPerBlock,
    rowsPerImage: Math.ceil(size[1] / block.height),
  };
}

function asBytes(data: HostData): Uint8Array {
  if (Array.isArray(data)) {
    // e.g. an F# array Fable could not type (a list converted to an array)
    throw new Error("wombat.rendering: host data must be an ArrayBuffer or typed array, got a plain JS array");
  }
  return data instanceof ArrayBuffer
    ? new Uint8Array(data)
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

/**
 * Write `data` into a box of `texture` (`queue.writeTexture`). Without
 * `region` the whole mip 0 is written; row pitches default to the
 * tightly packed layout of the texture's format.
 */
export function writeTextureRegion(
  device: GPUDevice, texture: GPUTexture, data: HostData, region: TextureRegion = {},
): void {
  const r = resolveRegion(texture, region);
  device.queue.writeTexture(
    { texture, mipLevel: r.mipLevel, origin: r.origin },
    asBytes(data) as unknown as GPUAllowSharedBufferSource,
    { bytesPerRow: region.bytesPerRow ?? r.bytesPerRow, rowsPerImage: region.rowsPerImage ?? r.rowsPerImage },
    r.size,
  );
}

/**
 * Write `data` (or `byteLength` bytes of it from `srcByteOffset`) to
 * `buffer` at `byteOffset` (`queue.writeBuffer`). WebGPU requires the
 * offset and the length to be multiples of 4.
 */
export function writeBufferRange(
  device: GPUDevice, buffer: GPUBuffer, byteOffset: number, data: HostData,
  srcByteOffset = 0, byteLength?: number,
): void {
  const bytes = asBytes(data);
  const length = byteLength ?? bytes.byteLength - srcByteOffset;
  if (byteOffset % 4 !== 0 || length % 4 !== 0) {
    throw new Error(`writeBufferRange: offset ${byteOffset} and length ${length} must be multiples of 4`);
  }
  device.queue.writeBuffer(buffer, byteOffset, bytes.buffer as ArrayBuffer, bytes.byteOffset + srcByteOffset, length);
}

/**
 * Read `byteLength` bytes of `buffer` from `byteOffset` back to the CPU.
 * The buffer needs `COPY_SRC` usage. Resolves after all work submitted
 * before the call has finished.
 */
export async function readBuffer(
  device: GPUDevice, buffer: GPUBuffer, byteOffset = 0, byteLength?: number,
): Promise<ArrayBuffer> {
  const length = byteLength ?? buffer.size - byteOffset;
  // copyBufferToBuffer works on 4-byte multiples: widen, then cut
  const start = byteOffset & ~3;
  const end = Math.min(buffer.size, (byteOffset + length + 3) & ~3);
  const staging = device.createBuffer({ size: Math.max(4, end - start), usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST });
  try {
    const enc = device.createCommandEncoder({ label: "readBuffer" });
    enc.copyBufferToBuffer(buffer, start, staging, 0, end - start);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    return staging.getMappedRange().slice(byteOffset - start, byteOffset - start + length);
  } finally {
    staging.destroy();
  }
}

/**
 * Read a box of `texture` back to the CPU, tightly packed (rows of
 * `size[0]` texels, `size[1]` rows per image). The texture needs
 * `COPY_SRC` usage; block-compressed formats are not supported.
 */
export async function readTextureRegion(
  device: GPUDevice, texture: GPUTexture, region: TextureRegion = {},
): Promise<ArrayBuffer> {
  const r = resolveRegion(texture, region);
  const block = blockInfoFor(texture.format);
  if (block.width !== 1 || block.height !== 1) {
    throw new Error(`readTextureRegion: block-compressed format ${texture.format} not supported`);
  }
  const [w, h, d] = r.size;
  const tight = r.bytesPerRow;
  const padded = Math.ceil(tight / 256) * 256;
  const staging = device.createBuffer({ size: Math.max(4, padded * h * d), usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST });
  try {
    const enc = device.createCommandEncoder({ label: "readTextureRegion" });
    enc.copyTextureToBuffer(
      { texture, mipLevel: r.mipLevel, origin: r.origin },
      { buffer: staging, bytesPerRow: padded, rowsPerImage: h },
      [w, h, d],
    );
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(staging.getMappedRange());
    const out = new Uint8Array(tight * h * d);
    if (padded === tight) out.set(src.subarray(0, out.byteLength));
    else for (let row = 0; row < h * d; row++) out.set(src.subarray(row * padded, row * padded + tight), row * tight);
    return out.buffer;
  } finally {
    staging.destroy();
  }
}

/** Options of the `MutableTexture` / `MutableBuffer` writes. */
export interface WriteOptions {
  /**
   * Mark the resource's aval after the write (coalesced: one mark per
   * microtask). Default `true`. Pass `false` for writes made during a
   * frame's evaluation — that frame uploads them anyway, and a mark
   * would only schedule another frame.
   */
  readonly notify?: boolean;
}

/**
 * Coalesced "contents changed" marking shared by the mutable resources:
 * the mark runs in a microtask, i.e. after any adaptive evaluation that
 * is on the stack right now (marking during evaluation would race the
 * render loop's dirty tracking).
 */
abstract class MutableResource<T> extends AbstractVal<T> {
  private _markPending = false;
  protected _disposed = false;

  /** Schedule a mark of this aval (see `WriteOptions.notify`). */
  markChanged(): void {
    if (this._markPending || this._disposed) return;
    this._markPending = true;
    queueMicrotask(() => {
      this._markPending = false;
      transact(() => markOutdated(this));
    });
  }

  get isDisposed(): boolean { return this._disposed; }
}

export interface MutableTextureDescriptor {
  /** `[width, height?, depthOrArrayLayers?]` */
  readonly size: readonly number[];
  readonly format: GPUTextureFormat;
  /** Default `"2d"`. */
  readonly dimension?: GPUTextureDimension;
  readonly mipLevelCount?: number;
  /** Extra usage. `TEXTURE_BINDING | COPY_DST | COPY_SRC` are always set. */
  readonly usage?: GPUTextureUsageFlags;
  readonly label?: string;
}

// Formats WebGPU can render to without optional features (core spec
// table). Adding RENDER_ATTACHMENT to those lets Dawn zero-initialise
// them with a clear instead of one staging buffer of the full texture
// size — which exceeds maxBufferSize for large 3D atlases.
const COLOR_RENDERABLE = new Set<GPUTextureFormat>([
  "r8unorm", "r8uint", "r8sint", "rg8unorm", "rg8uint", "rg8sint",
  "rgba8unorm", "rgba8unorm-srgb", "rgba8uint", "rgba8sint", "bgra8unorm", "bgra8unorm-srgb",
  "r16uint", "r16sint", "r16float", "rg16uint", "rg16sint", "rg16float",
  "rgba16uint", "rgba16sint", "rgba16float",
  "r32uint", "r32sint", "r32float", "rg32uint", "rg32sint", "rg32float",
  "rgba32uint", "rgba32sint", "rgba32float", "rgb10a2unorm", "rgb10a2uint",
]);

/**
 * A device-owned texture written in place, region by region. Itself an
 * `aval<ITexture>` (always the same `gpu`-kind value): bind it as a
 * texture uniform and every write shows up in the next frame.
 */
export class MutableTexture extends MutableResource<ITexture> {
  readonly texture: GPUTexture;
  private readonly _value: ITexture;

  constructor(readonly device: GPUDevice, desc: MutableTextureDescriptor) {
    super();
    const size = [desc.size[0] ?? 1, desc.size[1] ?? 1, desc.size[2] ?? 1];
    let usage = (desc.usage ?? 0) | TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST | TextureUsage.COPY_SRC;
    if (COLOR_RENDERABLE.has(desc.format)) usage |= TextureUsage.RENDER_ATTACHMENT;
    this.texture = device.createTexture({
      size,
      format: desc.format,
      dimension: desc.dimension ?? "2d",
      mipLevelCount: desc.mipLevelCount ?? 1,
      usage,
      ...(desc.label !== undefined ? { label: desc.label } : {}),
    });
    this._value = ITexture.fromGPU(this.texture);
  }

  get width(): number { return this.texture.width; }
  get height(): number { return this.texture.height; }
  get depthOrArrayLayers(): number { return this.texture.depthOrArrayLayers; }
  get format(): GPUTextureFormat { return this.texture.format; }

  override compute(_token: AdaptiveToken): ITexture { return this._value; }

  /** Write `data` into `region` (default: all of mip 0), tightly packed unless the region says otherwise. */
  write(data: HostData, region: TextureRegion = {}, opts: WriteOptions = {}): void {
    if (this._disposed) throw new Error("MutableTexture: write after dispose");
    writeTextureRegion(this.device, this.texture, data, region);
    if (opts.notify !== false) this.markChanged();
  }

  /** Read `region` back (tightly packed). */
  read(region: TextureRegion = {}): Promise<ArrayBuffer> {
    return readTextureRegion(this.device, this.texture, region);
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this.texture.destroy();
  }
}

export interface MutableBufferDescriptor {
  /** Initial size in bytes (rounded up to 4). */
  readonly size: number;
  /** Usage. Default `STORAGE`; `COPY_DST | COPY_SRC` are always set. */
  readonly usage?: GPUBufferUsageFlags;
  readonly label?: string;
}

/**
 * A device-owned buffer written in place. Itself an `aval<IBuffer>`:
 * its value only changes when `resize` / `ensureCapacity` has to
 * reallocate, content writes keep it (and only mark it, see
 * `WriteOptions`).
 */
export class MutableBuffer extends MutableResource<IBuffer> {
  private _buffer: GPUBuffer;
  private _value: IBuffer;
  private readonly _usage: GPUBufferUsageFlags;
  private readonly _label: string | undefined;

  constructor(readonly device: GPUDevice, desc: MutableBufferDescriptor) {
    super();
    this._usage = (desc.usage ?? BufferUsage.STORAGE) | BufferUsage.COPY_DST | BufferUsage.COPY_SRC;
    this._label = desc.label;
    this._buffer = this.allocate(desc.size);
    this._value = IBuffer.fromGPU(this._buffer);
  }

  private allocate(size: number): GPUBuffer {
    return this.device.createBuffer({
      size: Math.max(4, (size + 3) & ~3),
      usage: this._usage,
      ...(this._label !== undefined ? { label: this._label } : {}),
    });
  }

  /** The current allocation. Changes on reallocation. */
  get buffer(): GPUBuffer { return this._buffer; }
  get size(): number { return this._buffer.size; }

  override compute(_token: AdaptiveToken): IBuffer { return this._value; }

  /** Write `data` (or `byteLength` bytes of it from `srcByteOffset`) at `byteOffset`. */
  write(byteOffset: number, data: HostData, srcByteOffset = 0, byteLength?: number, opts: WriteOptions = {}): void {
    if (this._disposed) throw new Error("MutableBuffer: write after dispose");
    writeBufferRange(this.device, this._buffer, byteOffset, data, srcByteOffset, byteLength);
    if (opts.notify !== false) this.markChanged();
  }

  /**
   * Reallocate to exactly `size` bytes (no-op when already that size).
   * `preserve` copies the overlapping prefix of the old contents. The
   * aval changes to the new buffer; the old one is destroyed once the
   * GPU is done with work submitted so far.
   */
  resize(size: number, preserve = true): void {
    if (this._disposed) throw new Error("MutableBuffer: resize after dispose");
    const old = this._buffer;
    if (((size + 3) & ~3) === old.size) return;
    const next = this.allocate(size);
    if (preserve) {
      const enc = this.device.createCommandEncoder({ label: "MutableBuffer.resize" });
      enc.copyBufferToBuffer(old, 0, next, 0, Math.min(old.size, next.size));
      this.device.queue.submit([enc.finish()]);
    }
    this._buffer = next;
    this._value = IBuffer.fromGPU(next);
    // a frame being evaluated right now may still bind `old`
    this.device.queue.onSubmittedWorkDone().then(() => old.destroy(), () => old.destroy());
    this.markChanged();
  }

  /**
   * Grow to at least `minSize` bytes (by 1.5x steps, contents kept).
   * Returns true when the buffer was reallocated.
   */
  ensureCapacity(minSize: number): boolean {
    if (minSize <= this._buffer.size) return false;
    this.resize(Math.max(minSize, Math.ceil(this._buffer.size * 1.5)));
    return true;
  }

  /** Read `byteLength` bytes from `byteOffset` back (default: everything). */
  read(byteOffset = 0, byteLength?: number): Promise<ArrayBuffer> {
    return readBuffer(this.device, this._buffer, byteOffset, byteLength);
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this._buffer.destroy();
  }
}
