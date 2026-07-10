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
//   - `storageBuffers`         — same.
//   - >1 texture or sampler    — multi-binding API not in v1.
//
// Per-RO instancing IS supported: `instanceAttributes` are accepted as
// long as every entry passes the same tight-stride / stride-0-broadcast
// rule used for `vertexAttributes`, and `dc.instanceCount` may be any
// positive value (with `firstInstance === 0`).
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
import { asAttributeProvider } from "../core/provider.js";

/**
 * Maximum texture extent (per side) the heap path is willing to
 * ingest. Anything wider/taller escalates to the legacy renderer
 * which gives each RO its own dedicated GPUTexture — that's a
 * better fit than wasting an atlas page on a single ~4K image
 * (and it sidesteps device-limit risk near `maxTextureDimension2D`).
 */
const LEGACY_MAX_DIM = 4096;

/**
 * Total per-RO payload (attribute + instance + index bytes) above
 * which the heap path ejects the RO to the legacy renderer. The
 * heap path's value is collapsing many small ROs into one
 * drawIndirect; a single-RO bucket emits the same drawIndexed it
 * would have under legacy, AND a multi-MB allocation forces the
 * arena's pow2 GrowBuffer to allocate (and re-copy on grow) a
 * giant contiguous GPUBuffer. Default 16 MB — picked so typical
 * UI/SG geometry (a few hundred kB) sails through and only
 * terrain/photogrammetry/dense-pointcloud meshes eject.
 */
const HEAP_PAYLOAD_EJECT_BYTES = 16 * 1024 * 1024;

const isHostBuffer = (b: IBuffer): boolean => b.kind === "host";
// Samplers: accept both descriptor-form (`kind: "host"`) and resolved
// (`kind: "gpu"`). The heap atlas path uses ONE shared GPU sampler for
// all atlas reads — per-RO sampler state (filter modes, wrap modes)
// is packed into the drawHeader's formatBits and consumed by the
// shader, not bound as a per-RO GPUSampler. So the heap path doesn't
// care whether the user handed in a descriptor or a pre-resolved
// sampler; both are equivalent at this level.
const isHeapServableSampler = (_s: ISampler): boolean => true;

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
  // URL-deferred — the Sg layer hands us a placeholder while it loads.
  if (t.kind === "url") return false;
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

// Materialise every attribute view a provider knows about. Eligibility
// is decided before any shader is compiled, so we can't pull
// shader-driven here — we enumerate. Attribute providers are map-backed
// in practice, so `names()` is the full set and `tryGet` is a map hit.
function attrViews(p: { names(): Iterable<string>; tryGet(n: string): BufferView | undefined }): BufferView[] {
  const out: BufferView[] = [];
  for (const n of p.names()) { const v = p.tryGet(n); if (v !== undefined) out.push(v); }
  return out;
}

function bufferAvals(ro: RenderObject): aval<IBuffer>[] {
  const out: aval<IBuffer>[] = [];
  for (const v of attrViews(asAttributeProvider(ro.vertexAttributes))) out.push(v.buffer);
  if (ro.instanceAttributes !== undefined) {
    for (const v of attrViews(asAttributeProvider(ro.instanceAttributes))) out.push(v.buffer);
  }
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
  if (ro.storageBuffers !== undefined && ro.storageBuffers.count > 0) {
    return AVal.constant(false);
  }
  // Heap path serves at most one distinct texture + sampler per bucket
  // (the megacall draw has one bind group). The Sg compile layer
  // over-binds each texture under both `name` and `${name}_view` (so
  // whichever name the WGSL schema lands on resolves), so dedupe by
  // aval identity before applying the single-binding rule — otherwise
  // every textured leaf would be `count > 1` and fall to the legacy
  // per-RO path, which at scale (thousands of textured ROs ⇒ thousands
  // of draw calls + bind-group switches) is far slower than the heap
  // megacall sampling one shared atlas page.
  // NOTE identity-dedup WITHOUT a Set: the long-lived AVal.custom
  // closure below shares this function's V8 scope context, so any
  // temporary Set here would be retained per-RO for the scene's
  // lifetime (measured: 1 Set/RO of pure ballast at heap scale).
  if (countDistinct(ro.textures) > 1) {
    return AVal.constant(false);
  }
  if (countDistinct(ro.samplers) > 1) {
    return AVal.constant(false);
  }

  // 2. Interleaved / offset attributes are handled by de-interleaving at
  //    ingest (heapAdapter.tightenBufferView gathers a tight, offset-0
  //    copy keyed by (buffer, offset, stride) so sharing is preserved), so
  //    no stride/offset wedge remains. Broadcasts (singleValue / stride 0)
  //    still pass through as a degenerate length-1 case.

  // 3. Index-format wedge — heap stores indices as u32; u16 is widened
  //    to u32 at ingest (heapAdapter.widenU16). Anything else is rejected.
  if (ro.indices !== undefined) {
    const fmt = ro.indices.elementType.indexFormat;
    if (fmt !== "uint32" && fmt !== "uint16") return AVal.constant(false);
  }

  const buffers  = bufferAvals(ro);
  const textures = textureAvals(ro);
  const samplers = samplerAvals(ro);

  // Reactive AND-fold over all participating avals. We also subscribe
  // to drawCall (instanceCount/baseVertex/firstIndex/firstInstance
  // can flip; if they violate heap constraints the RO routes to the
  // legacy path that frame).
  return AVal.custom(token => {
    let payloadBytes = 0;
    for (const av of buffers) {
      const b = av.getValue(token);
      if (!isHostBuffer(b)) return false;
      payloadBytes += b.sizeBytes;
      // Early-out if any single RO already busts the budget — avoids
      // walking the rest of the buffer list for a guaranteed reject.
      if (payloadBytes > HEAP_PAYLOAD_EJECT_BYTES) return false;
    }
    for (const av of textures) if (!isHeapServableTexture(av.getValue(token))) return false;
    for (const av of samplers) if (!isHeapServableSampler(av.getValue(token))) return false;
    const dc = ro.drawCall.getValue(token);
    if (dc.instanceCount < 1) return false;
    if (dc.firstInstance !== 0) return false;
    if (dc.kind === "indexed") {
      if (ro.indices === undefined) return false;
      // Offsets and shared sub-buffer slices are all eligible now: the heap
      // still ingests the WHOLE index + vertex buffers (one shared arena
      // allocation per buffer aval — sharing is the heap's whole point), but
      // the record now honours the drawCall slice. firstIndex folds into the
      // record's indexStart, indexCount becomes the slice length (so a single
      // glyph of a multi-glyph run reads only its run of indices), and
      // baseVertex is baked into the transcoded index values at ingest
      // (heapAdapter.indicesAvalFor). No drawCall wedge remains here.
    } else {
      // Non-indexed: the megacall uses the local vertex index directly, so a
      // prefix read is safe (vid = 0..vertexCount-1). firstVertex must be 0 —
      // a non-indexed vertex offset would need a per-record field (the index
      // bake-in trick doesn't apply with no index buffer), so it's deferred.
      if (ro.indices !== undefined) return false;
      if (dc.firstVertex !== 0) return false;
    }
    return true;
  });
}


/** Count distinct avals in a name→aval map by identity, allocation-free
 *  for the 0/1/2-distinct cases that matter (we only compare > 1). */
function countDistinct(m: { iter(f: (n: string, av: aval<unknown>) => void): void }): number {
  let first: aval<unknown> | undefined;
  let second: aval<unknown> | undefined;
  let n = 0;
  m.iter((_n, av) => {
    if (av === first || av === second) return;
    if (first === undefined) { first = av; n = 1; }
    else if (second === undefined) { second = av; n = 2; }
    else n++;
  });
  return n;
}
