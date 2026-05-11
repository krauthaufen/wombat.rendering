// heapAdapter — `RenderObject → HeapDrawSpec`.
//
// Reactive: returns the same `HeapDrawSpec` shape the heap path
// already accepts. Per-input wiring:
//
//   - vertexAttributes   → spec.inputs[name] = BufferView (heap path
//                          handles BufferView ingest natively, pivots
//                          the pool aval-key onto the underlying
//                          `aval<IBuffer>` for share-by-identity).
//   - uniforms           → spec.inputs[name] = aval<unknown>.
//   - textures/samplers  → folded into a HeapTextureSet (single pair
//                          v1; the eligibility classifier already
//                          rejects multi-binding ROs).
//   - indices            → BufferView's `aval<IBuffer>` mapped to
//                          `aval<Uint32Array>` for the heap path's
//                          IndexPool.
//   - effect             → passed through.
//   - pipelineState      → passed through (heap path bucket-keys it).
//
// Drawcall wedges (instanceCount > 1, baseVertex/firstIndex/
// firstInstance != 0, non-indexed) are screened out by
// `isHeapEligible`; this adapter is the partner that converts the
// heap-eligible subset. If we see something we can't handle, throw —
// that's a classifier/adapter mismatch, a bug.
//
// No multi-binding texture API yet: `ro.textures.count <= 1` and
// `ro.samplers.count <= 1` are classifier invariants.

import { type aval, type AdaptiveToken } from "@aardworx/wombat.adaptive";
import type { IBuffer, HostBufferSource } from "../core/buffer.js";
import type { BufferView } from "../core/bufferView.js";
import { ITexture, type HostTextureSource } from "../core/texture.js";
import { ISampler } from "../core/sampler.js";
import type { RenderObject } from "../core/renderObject.js";
import type { HeapDrawSpec, HeapTextureSet } from "./heapScene.js";
import { AtlasPool } from "./textureAtlas/atlasPool.js";

/**
 * View `HostBufferSource` as `Uint32Array`. Index buffers are u32 in
 * the heap path; non-u32 indices would have to be transcoded (out
 * of scope — eligibility doesn't currently inspect index format).
 */
function asUint32(data: HostBufferSource): Uint32Array {
  if (data instanceof Uint32Array) return data;
  if (ArrayBuffer.isView(data)) {
    return new Uint32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  }
  return new Uint32Array(data);
}

// IndexPool keys on aval identity → ROs sharing the same
// `ro.indices.buffer` must produce the SAME downstream
// `aval<Uint32Array>`. A fresh `.map(…)` per RO would defeat the
// pool's identity-based sharing and balloon GPU memory (e.g. 11K
// spheres each allocating their own copy of the 6144-index buffer
// → 270 MB).
const indicesAvalCache = new WeakMap<aval<IBuffer>, aval<Uint32Array>>();

// Per-aval shared "current pool ref" cell. ROs that share an
// `aval<ITexture>` (and therefore one AtlasPool entry) all read
// through the same cell; `repack` updates `cell.ref` so every
// `release` closure frees the latest sub-rect after a swap.
const currentRefCellCache = new WeakMap<aval<ITexture>, { ref: number }>();
function currentRefCellFor(av: aval<ITexture>, initial: number): { ref: number } {
  let c = currentRefCellCache.get(av);
  if (c === undefined) {
    c = { ref: initial };
    currentRefCellCache.set(av, c);
  }
  return c;
}
function indicesAvalFor(ibAval: aval<IBuffer>): aval<Uint32Array> {
  let av = indicesAvalCache.get(ibAval);
  if (av === undefined) {
    av = ibAval.map((ib: IBuffer) => {
      if (ib.kind !== "host") {
        throw new Error("heapAdapter: index buffer flipped to native GPUBuffer; classifier should have caught this");
      }
      return asUint32(ib.data);
    });
    indicesAvalCache.set(ibAval, av);
  }
  return av;
}

/**
 * Pull `(format, width, height, mipLevelCount)` out of an ITexture so
 * we can run Tier-S eligibility against it. Both variants expose this:
 *
 *   - `kind: "host"` with a `raw` source carries the fields directly
 *     (`width`, `height`, `format`, optional `mipLevelCount`).
 *   - `kind: "host"` with an `external` source (`ImageBitmap`,
 *     `HTMLCanvasElement`, `ImageData`, …) reads `.width`/`.height`
 *     off the image and uses the source's `format` override (default
 *     `rgba8unorm`); `generateMips` triggers a >1 mip count.
 *   - `kind: "gpu"` reads `.format`/`.width`/`.height`/`.mipLevelCount`
 *     off the resolved `GPUTexture`. Width/height come from the WebGPU
 *     texture surface; no host source is exposed (the pool can't
 *     CPU-downscale, so mip generation defers to the existing GPU
 *     mip chain when wantsMips is true).
 *
 * Returns `null` for sources we can't measure (e.g. an `external`
 * source without dimensions — `HTMLVideoElement` whose readyState
 * hasn't given us a video frame yet).
 */
interface TextureDescriptor {
  format: GPUTextureFormat;
  width: number;
  height: number;
  mipLevelCount: number;
  host?: HostTextureSource;
}
function describeTexture(t: ITexture): TextureDescriptor | null {
  if (t.kind === "gpu") {
    const tex = t.texture;
    return {
      format: tex.format,
      width: tex.width,
      height: tex.height,
      mipLevelCount: tex.mipLevelCount,
    };
  }
  // URL-deferred textures are resolved at the Sg layer (placeholder
  // checker until ready); they should never reach the heap adapter.
  if (t.kind === "url") return null;
  const src = t.source;
  if (src.kind === "raw") {
    return {
      format: src.format,
      width: src.width,
      height: src.height,
      mipLevelCount: src.mipLevelCount ?? 1,
      host: src,
    };
  }
  // external — read width/height off the source. Most types have it,
  // HTMLVideoElement uses `videoWidth/videoHeight`.
  const ext = src.source as unknown;
  let w = 0, h = 0;
  if (typeof HTMLVideoElement !== "undefined" && ext instanceof HTMLVideoElement) {
    w = ext.videoWidth; h = ext.videoHeight;
  } else if (
    typeof ImageData !== "undefined" && ext instanceof ImageData
  ) {
    w = ext.width; h = ext.height;
  } else {
    const any = ext as { width?: number; height?: number };
    w = any.width ?? 0;
    h = any.height ?? 0;
  }
  if (w <= 0 || h <= 0) return null;
  return {
    format: src.format ?? "rgba8unorm",
    width: w,
    height: h,
    mipLevelCount: src.generateMips ? Math.floor(Math.log2(Math.max(w, h))) + 1 : 1,
    host: src,
  };
}

/**
 * Convert a heap-eligible `RenderObject` into a `HeapDrawSpec`. The
 * caller (hybrid render task) is responsible for ensuring eligibility
 * before calling — this adapter throws on any disagreement.
 *
 * The `token` is used once to read `drawCall` and the (single) texture
 * + sampler avals at addDraw time. Per-frame data updates flow
 * through the heap path's pool/repack machinery via the returned
 * spec's avals — this function does not subscribe to anything.
 */
export function renderObjectToHeapSpec(
  ro: RenderObject,
  token: AdaptiveToken,
  pool?: AtlasPool,
): HeapDrawSpec {
  // 1. Inputs map: vertex attributes (BufferView) + uniforms.
  const inputs: { [name: string]: aval<unknown> | unknown } = {};
  ro.vertexAttributes.iter((name, bv: BufferView) => { inputs[name] = bv; });
  ro.uniforms.iter((name, av) => { inputs[name] = av; });

  // 1b. Instance attributes — same shape, threaded via the heap path's
  //     per-RO instancing fast path (one record / one drawIndirect for
  //     `instanceCount > 1`, with per-instance attribute reads indexed
  //     by the in-RO instance idx).
  let instanceAttributes: { [name: string]: aval<unknown> | unknown } | undefined;
  if (ro.instanceAttributes !== undefined && ro.instanceAttributes.count > 0) {
    instanceAttributes = {};
    ro.instanceAttributes.iter((name, bv: BufferView) => {
      instanceAttributes![name] = bv;
    });
  }

  // 2. Indices: BufferView → aval<Uint32Array>. Map the underlying
  //    IBuffer aval; the heap path's IndexPool will key on this aval.
  if (ro.indices === undefined) {
    throw new Error("heapAdapter: RenderObject without indices not supported (heap is indexed-only)");
  }
  const indices: aval<Uint32Array> = indicesAvalFor(ro.indices.buffer);

  // 3. Texture/sampler: single-pair v1. Classifier already capped
  //    counts at 1 each.
  let textures: HeapTextureSet | undefined;
  if (ro.textures.count === 1 && ro.samplers.count === 1) {
    let texAval: aval<unknown> | undefined;
    let texVal: ITexture | undefined;
    let samplerAval: aval<unknown> | undefined;
    let samplerVal: ISampler | undefined;
    ro.textures.iter((_n, av) => {
      texAval = av;
      texVal = av.getValue(token) as ITexture;
    });
    ro.samplers.iter((_n, av) => {
      samplerAval = av;
      samplerVal = av.getValue(token) as ISampler;
    });
    void samplerAval;
    if (texVal === undefined || samplerVal === undefined) {
      throw new Error("heapAdapter: missing texture/sampler value");
    }

    // Inspect format/dimensions/mipLevels from ITexture. Both
    // variants expose enough to classify:
    //   - kind: "host" → carries width/height/format on the source
    //     descriptor (raw) or implicit (external; default rgba8unorm).
    //   - kind: "gpu"  → resolved GPUTexture exposes .format/.width/
    //     .height/.mipLevelCount.
    const dims = describeTexture(texVal);

    // Tier-S routing: pool present, format eligible, dims ≤ cap. Mips
    // are honoured if the source advertises them.
    const atlasFormat = dims === null ? null : AtlasPool.eligibleFormat(dims.format);
    if (
      pool !== undefined &&
      dims !== null &&
      atlasFormat !== null &&
      AtlasPool.eligibleSize(dims.width, dims.height)
    ) {
      const wantsMips = dims.mipLevelCount > 1;
      const acq = pool.acquire(
        atlasFormat,
        texAval as aval<ITexture>,
        dims.width,
        dims.height,
        {
          wantsMips,
          ...(dims.host !== undefined ? { source: { width: dims.width, height: dims.height, host: dims.host } } : {}),
        },
      );
      const texAvalTyped = texAval as aval<ITexture>;
      // Per-aval shared current-ref cell so all ROs sharing a single
      // `aval<ITexture>` see the same "live" pool ref. After a repack
      // every release closure reads the cell and frees the LATEST
      // sub-rect — no leaks even when multiple ROs are bucket-shared.
      const cell = currentRefCellFor(texAvalTyped, acq.ref);
      textures = {
        kind: "atlas",
        format: atlasFormat,
        pageId: acq.pageId,
        origin: acq.origin,
        size: acq.size,
        numMips: acq.numMips,
        sampler: samplerVal,
        page: acq.page,
        poolRef: acq.ref,
        release: () => pool.release(cell.ref),
        // Reactivity hooks: heapScene subscribes to `sourceAval`; on
        // mark, drains via `repack(newTex)` — the pool frees the old
        // sub-rect, acquires a new one, and the heap path rewrites
        // the drawHeader fields. Mirrors UniformPool.repack(av, val).
        sourceAval: texAvalTyped,
        repack: (newTex: ITexture) => {
          const next = pool.repack(texAvalTyped, newTex, {
            wantsMips: dims.mipLevelCount > 1,
          });
          cell.ref = next.ref;
          return next;
        },
      };
    } else {
      // Tier-L fallback: keep the existing standalone path, which
      // requires a resolved GPUTexture/GPUSampler.
      if (texVal.kind !== "gpu") {
        throw new Error("heapAdapter: standalone path requires ITexture.kind === 'gpu'");
      }
      if (samplerVal.kind !== "gpu") {
        throw new Error("heapAdapter: standalone path requires ISampler.kind === 'gpu'");
      }
      textures = {
        kind: "standalone",
        texture: ITexture.fromGPU(texVal.texture),
        sampler: ISampler.fromGPU(samplerVal.sampler),
      };
    }
  } else if (ro.textures.count > 0 || ro.samplers.count > 0) {
    throw new Error(
      `heapAdapter: RO has ${ro.textures.count} texture(s) and ${ro.samplers.count} sampler(s); ` +
      `single-pair only in v1 (classifier should have caught this)`,
    );
  }

  // 4. DrawCall: read once. Classifier validates fields are heap-
  //    compatible (indexed, instanceCount=1, zero offsets).
  const dc = ro.drawCall.getValue(token);
  if (dc.kind !== "indexed") {
    throw new Error("heapAdapter: non-indexed drawCall; classifier should have caught this");
  }

  return {
    effect: ro.effect,
    pipelineState: ro.pipelineState,
    inputs,
    ...(instanceAttributes !== undefined ? { instanceAttributes } : {}),
    ...(dc.instanceCount > 1 ? { instanceCount: dc.instanceCount } : {}),
    indices,
    ...(textures !== undefined ? { textures } : {}),
  };
}
