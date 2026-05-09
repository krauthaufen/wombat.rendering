// heapEligibility — reactive predicate over a RenderObject deciding
// whether it can ride the heap-bucket fast path.
//
// Three classes of conditions:
//   1. STATIC HARD WEDGES — features the heap path doesn't ingest
//      yet (instanceAttributes, storageBuffers, > 1 texture or
//      sampler). Decided once at classifier time.
//   2. STATIC SOFT WEDGES — supported with limits (indexed-only,
//      instanceCount=1, baseVertex/firstIndex/firstInstance=0).
//      Read once from the drawCall snapshot.
//   3. REACTIVE — every `aval<IBuffer>` (vertex attribs + indices)
//      must currently resolve to `kind: "host"`. Every `aval<ITexture>`
//      must be heap-servable: `kind: "host"` (atlasable / Tier-L
//      host upload), 2D dimension, single array layer, and within
//      `LEGACY_MAX_DIM` on each side. Anything failing those rules
//      escalates to the legacy ScenePass — the user is managing a
//      backend resource (render-target, video, environment probe,
//      streaming tile) or asking for a cubemap/array/volume that
//      the heap shaders don't support, or a texture too large for
//      heap-side handling to be a win. Every `aval<ISampler>` must
//      resolve to `kind: "gpu"`. When any of these mark and flip,
//      the result aval marks → hybrid task repartitions.
//
// Out-of-scope blockers (return false unconditionally if present):
//   - `instanceAttributes`     — heap path doesn't ingest these yet.
//   - `storageBuffers`         — same.
//   - >1 texture or sampler    — multi-binding API not in v1.
//
// These are conservative wedges, not permanent: as the heap path
// grows feature support each becomes a per-buffer eligibility
// question instead of a blanket "no".

import { AVal, type aval } from "@aardworx/wombat.adaptive";
import type { IBuffer } from "../core/buffer.js";
import type { BufferView } from "../core/bufferView.js";
import type { ITexture } from "../core/texture.js";
import type { ISampler } from "../core/sampler.js";
import type { RenderObject } from "../core/renderObject.js";

/**
 * Maximum texture extent (per side) the heap path is willing to
 * ingest. Anything wider/taller escalates to the legacy renderer
 * which gives each RO its own dedicated GPUTexture — that's a
 * better fit than wasting an atlas page on a single ~4K image
 * (and it sidesteps device-limit risk near `maxTextureDimension2D`).
 */
const LEGACY_MAX_DIM = 4096;

const isHostBuffer = (b: IBuffer): boolean => b.kind === "host";
const isGpuSampler = (s: ISampler): boolean => s.kind === "gpu";

/**
 * Heap-servability check for a single texture value. Returns `true`
 * when the texture fits the heap path's ingest envelope:
 *   - `kind === "host"` (the user isn't managing a backend
 *     resource — those go to legacy untouched).
 *   - 2D, single array layer (heap shaders are 2D-only).
 *   - Both extents `<= LEGACY_MAX_DIM`.
 *
 * For `kind: "gpu"` we currently always reject (rule 1) — even a
 * plain 2D rgba8unorm GPUTexture goes legacy, on the assumption
 * that the user attached a `GPUTexture` for a reason (render target,
 * external import, streamed tile) and shouldn't be silently
 * rebadged. The dimension/layer/size checks are still implemented
 * for `kind: "gpu"` for clarity / future use if we relax rule 1.
 *
 * For `kind: "host"`, by construction `RawTextureSource` and
 * `ExternalTextureSource` are 2D single-layer — but we read
 * `depthOrArrayLayers` from raw sources defensively (it's optional
 * on the descriptor and could in principle be set).
 */
function isHeapServableTexture(t: ITexture): boolean {
  if (t.kind === "gpu") return false;
  const src = t.source;
  if (src.kind === "raw") {
    if ((src.depthOrArrayLayers ?? 1) !== 1) return false;
    if (src.width > LEGACY_MAX_DIM || src.height > LEGACY_MAX_DIM) return false;
    return true;
  }
  // external — read width/height off the source. Most types have
  // it; HTMLVideoElement uses videoWidth/videoHeight. Sources
  // without measurable dimensions are deferred (eligible until
  // they resolve — heapAdapter handles undimensioned gracefully).
  const ext = src.source as unknown;
  let w = 0, h = 0;
  if (typeof HTMLVideoElement !== "undefined" && ext instanceof HTMLVideoElement) {
    w = ext.videoWidth; h = ext.videoHeight;
  } else if (typeof ImageData !== "undefined" && ext instanceof ImageData) {
    w = ext.width; h = ext.height;
  } else {
    const any = ext as { width?: number; height?: number };
    w = any.width ?? 0;
    h = any.height ?? 0;
  }
  if (w > LEGACY_MAX_DIM || h > LEGACY_MAX_DIM) return false;
  return true;
}

function bufferAvals(ro: RenderObject): aval<IBuffer>[] {
  const out: aval<IBuffer>[] = [];
  ro.vertexAttributes.iter((_k, v: BufferView) => { out.push(v.buffer); });
  if (ro.indices !== undefined) out.push(ro.indices.buffer);
  return out;
}

function textureAvals(ro: RenderObject): aval<ITexture>[] {
  const out: aval<ITexture>[] = [];
  ro.textures.iter((_k, av) => { out.push(av); });
  return out;
}

function samplerAvals(ro: RenderObject): aval<ISampler>[] {
  const out: aval<ISampler>[] = [];
  ro.samplers.iter((_k, av) => { out.push(av); });
  return out;
}

/**
 * `true` when the RO is heap-bucketable, `false` when it must fall
 * back to the legacy per-RO path. Reactive on every contributing
 * `aval`: rewires the partition automatically when any input flips.
 */
export function isHeapEligible(ro: RenderObject): aval<boolean> {
  // 1. Static hard wedges.
  if (ro.instanceAttributes !== undefined && ro.instanceAttributes.count > 0) {
    return AVal.constant(false);
  }
  if (ro.storageBuffers !== undefined && ro.storageBuffers.count > 0) {
    return AVal.constant(false);
  }
  if (ro.textures.count > 1 || ro.samplers.count > 1) {
    return AVal.constant(false);
  }

  // 2. BufferView stride wedge — only tight per-vertex layouts (and
  //    broadcasts via `singleValue` / stride 0) are ingested. The
  //    shader-side cyclic addressing (`vid % length`) makes broadcasts
  //    a degenerate per-vertex case (length 1), so we don't reject
  //    them here — but interleaved strides remain unsupported.
  let strideBlocked = false;
  ro.vertexAttributes.iter((_k, v: BufferView) => {
    const stride = v.stride ?? v.elementType.byteSize;
    const isBroadcast = v.singleValue !== undefined || stride === 0;
    if (!isBroadcast && stride !== v.elementType.byteSize) strideBlocked = true;
  });
  if (strideBlocked) return AVal.constant(false);

  // 3. Index-format wedge — heap stores indices as u32.
  if (ro.indices !== undefined && ro.indices.elementType.indexFormat !== "uint32") {
    return AVal.constant(false);
  }

  const buffers  = bufferAvals(ro);
  const textures = textureAvals(ro);
  const samplers = samplerAvals(ro);

  // Reactive AND-fold over all participating avals. We also subscribe
  // to drawCall (instanceCount/baseVertex/firstIndex/firstInstance
  // can flip; if they violate heap constraints the RO routes to the
  // legacy path that frame).
  return AVal.custom(token => {
    for (const av of buffers)  if (!isHostBuffer(av.getValue(token)))         return false;
    for (const av of textures) if (!isHeapServableTexture(av.getValue(token))) return false;
    for (const av of samplers) if (!isGpuSampler(av.getValue(token)))          return false;
    const dc = ro.drawCall.getValue(token);
    if (dc.kind !== "indexed") return false;
    if (dc.instanceCount !== 1) return false;
    if (dc.baseVertex !== 0) return false;
    if (dc.firstIndex !== 0) return false;
    if (dc.firstInstance !== 0) return false;
    return true;
  });
}
