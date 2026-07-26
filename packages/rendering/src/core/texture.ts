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
  /**
   * Dimension SNAPSHOT taken at `ITexture.fromExternal` time. Memory-
   * lean callers close their ImageBitmap once its pixels are uploaded;
   * a closed bitmap reports 0×0, which would flip classification
   * (atlas-eligibility, tier routing) on a later re-ingest. Measurement
   * sites must prefer these over live source reads.
   */
  readonly width?: number;
  readonly height?: number;
}

export type HostTextureSource = RawTextureSource | ExternalTextureSource;

/** One cube face's pixel source (uploaded via `copyExternalImageToTexture`). */
export type CubeFaceSource =
  | ImageBitmap
  | HTMLImageElement
  | HTMLCanvasElement
  | OffscreenCanvas
  | ImageData;

// `url` is a *deferred* host source — the runtime can't upload until the
// browser has fetched + decoded the image. Consumers that route textures
// to the GPU (adaptiveTexture, atlas pool, heap adapter) refuse to handle
// `url` directly; the wombat.dom Sg layer resolves it asynchronously,
// swapping in a placeholder until the real bitmap is ready (see
// `scene/textureResolver.ts`). Keep this variant out of the rendering
// runtime's hot paths — it exists purely as an authoring shorthand.
export type ITexture =
  | { readonly kind: "gpu"; readonly texture: GPUTexture }
  | { readonly kind: "host"; readonly source: HostTextureSource }
  | { readonly kind: "url"; readonly url: string;
      readonly format?: GPUTextureFormat; readonly generateMips?: boolean }
  // A cube map as 6 faces in order [+X,-X,+Y,-Y,+Z,-Z] (Z-up convention: +Z up).
  // The heap renders this as a 2D grid texture with cross-face gutters + a
  // face-projection sample — NO `texture_cube` (Metal/WebKit reject cube views
  // over array textures). See heapAdapter (packing) + heapEffectIR (rewrite).
  | { readonly kind: "cube"; readonly faces: readonly CubeFaceSource[]; readonly faceSize?: number };

// ─── Equality / hash protocol ──────────────────────────────────────────
//
// `ITexture` values flow through `aval<ITexture>` into identity-keyed
// caches (the atlas pool, the uniform pool). Without an explicit
// equality protocol two structurally-equal values built at distinct
// call sites — e.g. `ITexture.fromUrl("X")` twice — would key
// separately and defeat the dedup. We define:
//
//   - `url`      → equal iff `(url, format, generateMips)` match. Pure
//                  structural; cheap to hash.
//   - `gpu`      → reference equality on the `GPUTexture` (opaque
//                  resource — two distinct handles are two textures).
//   - `host`/raw → structural metadata `(width, height, format,
//                  mipLevelCount, depthOrArrayLayers)` AND reference
//                  equality on the byte payload (`Uint8Array` etc. are
//                  "non-trivial" — comparing contents is too costly and
//                  rarely what's wanted; identity is the right call).
//   - `host`/external → reference equality on the source DOM object.
//
// `getHashCode` mirrors: structural fields hashed; opaque leaves
// (GPUTexture, byte payload, DOM source) contribute an identity hash.
// So an `ITexture` is always "hashable" (works in `HashTable`-backed
// caches), but two distinct opaque-backed textures never collide by
// content.
//
// The methods are installed via a shared prototype object so each
// `ITexture` value carries no own closures; data fields remain own
// enumerable props (spreads / `Object.keys` see only data).

function hashStr(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return h;
}
let _idCounter = 0;
const _idHashes = new WeakMap<object, number>();
function idHash(o: object): number {
  let v = _idHashes.get(o);
  if (v === undefined) { v = (++_idCounter) | 0; _idHashes.set(o, v); }
  return v;
}
function mix(a: number, b: number): number { return (((a << 5) - a + b) | 0); }

function hostSourceEquals(a: HostTextureSource, b: HostTextureSource): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "raw" && b.kind === "raw") {
    return a.data === b.data && a.width === b.width && a.height === b.height
      && a.format === b.format && (a.mipLevelCount ?? 1) === (b.mipLevelCount ?? 1)
      && (a.depthOrArrayLayers ?? 1) === (b.depthOrArrayLayers ?? 1);
  }
  // external
  const ea = a as ExternalTextureSource, eb = b as ExternalTextureSource;
  return ea.source === eb.source && ea.format === eb.format
    && (ea.generateMips ?? false) === (eb.generateMips ?? false);
}
function hostSourceHash(s: HostTextureSource): number {
  if (s.kind === "raw") {
    return mix(mix(mix(mix(hashStr(s.format), s.width), s.height), s.mipLevelCount ?? 1), idHash(s.data as object));
  }
  const e = s as ExternalTextureSource;
  return mix(idHash(e.source as object), hashStr(e.format ?? "rgba8unorm"));
}

function iTextureEquals(a: ITexture, o: unknown): boolean {
  if (a === o) return true;
  if (o === null || typeof o !== "object") return false;
  const b = o as ITexture;
  if (a.kind !== b.kind) return false;
  switch (a.kind) {
    case "url":  return a.url === (b as { url: string }).url
      && a.format === (b as { format?: string }).format
      && (a.generateMips ?? false) === ((b as { generateMips?: boolean }).generateMips ?? false);
    case "gpu":  return a.texture === (b as { texture: GPUTexture }).texture;
    case "host": return hostSourceEquals(a.source, (b as { source: HostTextureSource }).source);
    case "cube": {
      const bc = b as { faces: readonly CubeFaceSource[]; faceSize?: number };
      if (a.faces.length !== bc.faces.length || a.faceSize !== bc.faceSize) return false;
      for (let i = 0; i < a.faces.length; i++) if (a.faces[i] !== bc.faces[i]) return false;
      return true;
    }
  }
}
function iTextureHash(a: ITexture): number {
  switch (a.kind) {
    case "url":  return mix(mix(hashStr("url"), hashStr(a.url)), hashStr(a.format ?? "")) | (a.generateMips ? 1 : 0);
    case "gpu":  return mix(hashStr("gpu"), idHash(a.texture));
    case "host": return mix(hashStr("host"), hostSourceHash(a.source));
    case "cube": {
      let h = mix(hashStr("cube"), a.faceSize ?? 0);
      for (const f of a.faces) h = mix(h, idHash(f as object));
      return h;
    }
  }
}
const ITEXTURE_PROTO = {
  equals(this: ITexture, o: unknown): boolean { return iTextureEquals(this, o); },
  getHashCode(this: ITexture): number { return iTextureHash(this) | 0; },
};
function withEq<T extends ITexture>(t: T): T {
  return Object.assign(Object.create(ITEXTURE_PROTO) as object, t) as T;
}

// `url` textures are interned by their structural key so two
// `ITexture.fromUrl("X")` calls return the *same* object — `===` true,
// no second atlas slot, no second fetch. Bounded by the distinct set of
// (url, format, generateMips) tuples; url descriptors are tiny.
const _urlIntern = new Map<string, ITexture>();

export const ITexture = {
  fromGPU(texture: GPUTexture): ITexture {
    return withEq({ kind: "gpu", texture });
  },
  fromExternal(
    source: ExternalTextureSource["source"],
    opts: { format?: GPUTextureFormat; generateMips?: boolean } = {},
  ): ITexture {
    // Snapshot dimensions NOW — the caller may close an ImageBitmap
    // after upload, and a closed bitmap reports 0x0 (which would flip
    // atlas-eligibility / tier classification on a later re-ingest).
    const s = source as { width?: number; height?: number; videoWidth?: number; videoHeight?: number };
    const w = (typeof HTMLVideoElement !== "undefined" && source instanceof HTMLVideoElement)
      ? (s.videoWidth ?? 0) : (s.width ?? 0);
    const h = (typeof HTMLVideoElement !== "undefined" && source instanceof HTMLVideoElement)
      ? (s.videoHeight ?? 0) : (s.height ?? 0);
    const ext: ExternalTextureSource = {
      kind: "external",
      source,
      ...(w > 0 && h > 0 ? { width: w, height: h } : {}),
      ...(opts.format !== undefined ? { format: opts.format } : {}),
      ...(opts.generateMips !== undefined ? { generateMips: opts.generateMips } : {}),
    };
    return withEq({ kind: "host", source: ext });
  },
  fromRaw(raw: Omit<RawTextureSource, "kind">): ITexture {
    return withEq({ kind: "host", source: { kind: "raw", ...raw } });
  },
  /**
   * A cube map from 6 face images in order [+X,-X,+Y,-Y,+Z,-Z] (Z-up: +Z up).
   * The heap packs these into a 2D grid with cross-face gutters and samples
   * by face-projection — no `texture_cube`, so it works on iOS/Metal.
   * `faceSize` overrides the per-face packing resolution (default: heap's F0).
   */
  fromCubeFaces(faces: readonly CubeFaceSource[], opts: { faceSize?: number } = {}): ITexture {
    if (faces.length !== 6) throw new Error(`ITexture.fromCubeFaces: expected 6 faces, got ${faces.length}`);
    return withEq({ kind: "cube", faces, ...(opts.faceSize !== undefined ? { faceSize: opts.faceSize } : {}) });
  },
  fromUrl(url: string, opts: { format?: GPUTextureFormat; generateMips?: boolean } = {}): ITexture {
    const key = `${url} ${opts.format ?? ""} ${opts.generateMips ? 1 : 0}`;
    let t = _urlIntern.get(key);
    if (t === undefined) {
      t = withEq({
        kind: "url", url,
        ...(opts.format !== undefined ? { format: opts.format } : {}),
        ...(opts.generateMips !== undefined ? { generateMips: opts.generateMips } : {}),
      });
      _urlIntern.set(key, t);
    }
    return t;
  },
} as const;
