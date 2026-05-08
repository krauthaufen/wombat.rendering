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
import type { ITexture } from "../core/texture.js";
import type { ISampler } from "../core/sampler.js";
import type { RenderObject } from "../core/renderObject.js";
import type { HeapDrawSpec, HeapTextureSet } from "./heapScene.js";

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
 * Convert a heap-eligible `RenderObject` into a `HeapDrawSpec`. The
 * caller (hybrid render task) is responsible for ensuring eligibility
 * before calling — this adapter throws on any disagreement.
 *
 * The `token` is used once to read `drawCall` and the (single) texture
 * + sampler avals at addDraw time. Per-frame data updates flow
 * through the heap path's pool/repack machinery via the returned
 * spec's avals — this function does not subscribe to anything.
 */
export function renderObjectToHeapSpec(ro: RenderObject, token: AdaptiveToken): HeapDrawSpec {
  // 1. Inputs map: vertex attributes (BufferView) + uniforms.
  const inputs: { [name: string]: aval<unknown> | unknown } = {};
  ro.vertexAttributes.iter((name, bv: BufferView) => { inputs[name] = bv; });
  ro.uniforms.iter((name, av) => { inputs[name] = av; });

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
    let texture: GPUTexture | undefined;
    let sampler: GPUSampler | undefined;
    ro.textures.iter((_n, av) => {
      const t = av.getValue(token) as ITexture;
      if (t.kind !== "gpu") {
        throw new Error("heapAdapter: texture kind != 'gpu'; classifier should have caught this");
      }
      texture = t.texture;
    });
    ro.samplers.iter((_n, av) => {
      const s = av.getValue(token) as ISampler;
      if (s.kind !== "gpu") {
        throw new Error("heapAdapter: sampler kind != 'gpu'; classifier should have caught this");
      }
      sampler = s.sampler;
    });
    textures = { texture: texture!, sampler: sampler! };
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
    indices,
    ...(textures !== undefined ? { textures } : {}),
  };
}
