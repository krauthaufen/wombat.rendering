// ITexture — texture source that can be either a real GPU texture
// or a host-side image. Mirrors the IBuffer pattern.
//
// Host sources cover everything WebGPU's `copyExternalImageToTexture`
// or `writeTexture` can accept:
//   - `ImageBitmap`, `HTMLImageElement`, `HTMLVideoElement`,
//     `HTMLCanvasElement`, `OffscreenCanvas`, `ImageData` —
//     uploaded via `copyExternalImageToTexture`.
//   - Raw `{ data, width, height, format }` — uploaded via
//     `queue.writeTexture` (one mip; runtime can synthesise the
//     rest if requested).

export interface RawTextureSource {
  readonly kind: "raw";
  readonly data: ArrayBuffer | ArrayBufferView;
  readonly width: number;
  readonly height: number;
  readonly depthOrArrayLayers?: number;
  readonly format: GPUTextureFormat;
  readonly mipLevelCount?: number;
}

export interface ExternalTextureSource {
  readonly kind: "external";
  readonly source:
    | ImageBitmap
    | HTMLImageElement
    | HTMLVideoElement
    | HTMLCanvasElement
    | OffscreenCanvas
    | ImageData;
  /** Override the format the runtime should allocate (default: "rgba8unorm"). */
  readonly format?: GPUTextureFormat;
  /** Generate a mip chain on upload. Default `false`. */
  readonly generateMips?: boolean;
}

export type HostTextureSource = RawTextureSource | ExternalTextureSource;

export type ITexture =
  | { readonly kind: "gpu"; readonly texture: GPUTexture }
  | { readonly kind: "host"; readonly source: HostTextureSource };

export const ITexture = {
  fromGPU(texture: GPUTexture): ITexture {
    return { kind: "gpu", texture };
  },
  fromExternal(
    source: ExternalTextureSource["source"],
    opts: { format?: GPUTextureFormat; generateMips?: boolean } = {},
  ): ITexture {
    const ext: ExternalTextureSource = {
      kind: "external",
      source,
      ...(opts.format !== undefined ? { format: opts.format } : {}),
      ...(opts.generateMips !== undefined ? { generateMips: opts.generateMips } : {}),
    };
    return { kind: "host", source: ext };
  },
  fromRaw(raw: Omit<RawTextureSource, "kind">): ITexture {
    return { kind: "host", source: { kind: "raw", ...raw } };
  },
} as const;
