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
//      to `kind: "gpu"` (heap path doesn't run host-texture upload),
//      every `aval<ISampler>` to `kind: "gpu"`. When any of these
//      mark and flip, the result aval marks → hybrid task repartitions.
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

const isHostBuffer = (b: IBuffer): boolean => b.kind === "host";
const isGpuTexture = (t: ITexture): boolean => t.kind === "gpu";
const isGpuSampler = (s: ISampler): boolean => s.kind === "gpu";

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
    for (const av of buffers)  if (!isHostBuffer(av.getValue(token)))  return false;
    for (const av of textures) if (!isGpuTexture(av.getValue(token)))  return false;
    for (const av of samplers) if (!isGpuSampler(av.getValue(token)))  return false;
    const dc = ro.drawCall.getValue(token);
    if (dc.kind !== "indexed") return false;
    if (dc.instanceCount !== 1) return false;
    if (dc.baseVertex !== 0) return false;
    if (dc.firstIndex !== 0) return false;
    if (dc.firstInstance !== 0) return false;
    return true;
  });
}
