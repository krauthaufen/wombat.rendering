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

import { type aval, type AdaptiveToken, AVal } from "@aardworx/wombat.adaptive";
import { IBuffer, type HostBufferSource } from "../core/buffer.js";
import { BufferView } from "../core/bufferView.js";
import { ITexture, type HostTextureSource } from "../core/texture.js";
import { ISampler } from "../core/sampler.js";

/** Map shader-defined sampler state (FShade Filter/WrapMode names) to a
 *  GPUSamplerDescriptor (heap path). */
function samplerDescriptorFromState(
  state: {
    readonly filter: string;
    readonly addressU: string;
    readonly addressV: string;
    readonly addressW?: string;
    readonly comparison?: string;
    readonly maxAnisotropy?: number;
    readonly minLod?: number;
    readonly maxLod?: number;
    readonly mipLodBias?: number;
  },
): GPUSamplerDescriptor {
  // FShade WrapMode names. WebGPU has no Border / MirrorOnce — map them to
  // their closest modes (clamp-to-edge / mirror-repeat).
  const addr = (a: string): GPUAddressMode =>
    a === "Clamp" || a === "Border" ? "clamp-to-edge"
    : a === "Mirror" || a === "MirrorOnce" ? "mirror-repeat"
    : "repeat";
  // Full FShade Filter vocabulary: the name spells Min/Mag/Mip in order;
  // a missing Mip part means "use the trailing mode for mip too".
  // Anisotropic → all-linear + maxAnisotropy.
  let min: GPUFilterMode = "linear";
  let mag: GPUFilterMode = "linear";
  let mip: GPUMipmapFilterMode = "linear";
  let aniso = 1;
  switch (state.filter) {
    case "Anisotropic": aniso = 16; break;
    case "MinMagMipLinear": break;
    case "MinMagMipPoint": min = "nearest"; mag = "nearest"; mip = "nearest"; break;
    case "MinMagLinearMipPoint": mip = "nearest"; break;
    case "MinMagPointMipLinear": min = "nearest"; mag = "nearest"; break;
    case "MinLinearMagMipPoint": mag = "nearest"; mip = "nearest"; break;
    case "MinLinearMagPointMipLinear": mag = "nearest"; break;
    case "MinPointMagLinearMipPoint": min = "nearest"; mip = "nearest"; break;
    case "MinPointMagMipLinear": min = "nearest"; break;
    case "MinMagPoint": min = "nearest"; mag = "nearest"; mip = "nearest"; break;
    case "MinMagLinear": break;
    case "MinPointMagLinear": min = "nearest"; break;
    case "MinLinearMagPoint": mag = "nearest"; break;
    default: break; // unknown → all-linear
  }
  // Explicit maxAnisotropy overrides the Anisotropic-filter default.
  // WebGPU validation: maxAnisotropy > 1 requires all-linear filtering.
  if (state.maxAnisotropy !== undefined) aniso = state.maxAnisotropy;
  if (min !== "linear" || mag !== "linear" || mip !== "linear") aniso = 1;
  const desc: GPUSamplerDescriptor = {
    magFilter: mag, minFilter: min, mipmapFilter: mip,
    addressModeU: addr(state.addressU), addressModeV: addr(state.addressV),
    maxAnisotropy: aniso,
  };
  if (state.addressW !== undefined) (desc as { addressModeW?: GPUAddressMode }).addressModeW = addr(state.addressW);
  if (state.minLod !== undefined) (desc as { lodMinClamp?: number }).lodMinClamp = state.minLod;
  if (state.maxLod !== undefined) (desc as { lodMaxClamp?: number }).lodMaxClamp = state.maxLod;
  // `comparison` is applied by the caller when the binding is a
  // sampler_comparison (a compare function on a filtering sampler would
  // fail WebGPU validation). `mipLodBias` has no WebGPU counterpart.
  return desc;
}

/** FShade ComparisonFunction case name → GPUCompareFunction. */
function compareFunctionOf(name: string): GPUCompareFunction {
  switch (name) {
    case "Never": return "never";
    case "Less": return "less";
    case "Equal": return "equal";
    case "LessOrEqual": return "less-equal";
    case "Greater": return "greater";
    case "GreaterOrEqual": return "greater-equal";
    case "NotEqual": return "not-equal";
    default: return "always";
  }
}
import type { RenderObject } from "../core/renderObject.js";
import { asAttributeProvider, asUniformProvider } from "../core/provider.js";
import type { HeapDrawSpec, HeapTextureSet } from "./heapScene.js";
import { isBufferView } from "./heapScene/pools.js";
import { compileHeapEffect } from "./heapEffect.js";
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

// Widen 16-bit indices to u32 (the heap's indexStorage is u32). Copy, not a
// reinterpret view — each u16 index becomes a u32 element.
function widenU16(data: HostBufferSource): Uint32Array {
  const u16 = data instanceof Uint16Array
    ? data
    : ArrayBuffer.isView(data)
      ? new Uint16Array(data.buffer, data.byteOffset, data.byteLength / 2)
      : new Uint16Array(data);
  return Uint32Array.from(u16);
}

// IndexPool keys on aval identity → ROs sharing the same
// `ro.indices.buffer` must produce the SAME downstream
// `aval<Uint32Array>`. A fresh `.map(…)` per RO would defeat the
// pool's identity-based sharing and balloon GPU memory (e.g. 11K
// spheres each allocating their own copy of the 6144-index buffer
// → 270 MB).
// Keyed by (buffer aval, baseVertex). baseVertex is baked into the index
// VALUES (index[i] + baseVertex), so a buffer drawn with two different
// baseVertex offsets needs two distinct transcoded arrays — but the common
// case (baseVertex 0, or per-primitive index buffers that aren't shared)
// keeps one entry per buffer and preserves the pool's identity sharing.
const indicesAvalCache = new WeakMap<aval<IBuffer>, Map<number, aval<Uint32Array>>>();

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
function indicesAvalFor(ibAval: aval<IBuffer>, indexFormat: GPUIndexFormat, baseVertex: number): aval<Uint32Array> {
  let byBase = indicesAvalCache.get(ibAval);
  if (byBase === undefined) {
    byBase = new Map();
    indicesAvalCache.set(ibAval, byBase);
  }
  let av = byBase.get(baseVertex);
  if (av === undefined) {
    av = ibAval.map((ib: IBuffer) => {
      if (ib.kind !== "host") {
        throw new Error("heapAdapter: index buffer flipped to native GPUBuffer; classifier should have caught this");
      }
      const u32 = indexFormat === "uint16" ? widenU16(ib.data) : asUint32(ib.data);
      if (baseVertex === 0) return u32;
      // Fold baseVertex into the index values so the megacall decode
      // (vid = indexStorage[...]) lands on the right vertex with no
      // extra record field. Copy (don't mutate the shared view).
      const out = new Uint32Array(u32.length);
      for (let i = 0; i < u32.length; i++) out[i] = u32[i]! + baseVertex;
      return out;
    });
    byBase.set(baseVertex, av);
  }
  return av;
}

// Byte view over a host buffer source (for the de-interleave gather).
function bytesOf(data: HostBufferSource): Uint8Array {
  if (data instanceof Uint8Array) return data;
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return new Uint8Array(data);
}

// De-interleaved (tight, offset-0) views, keyed by (source buffer aval,
// offset, stride) so ROs sharing one interleaved buffer + the same attribute
// share a single gathered allocation.
const deinterleaveCache = new WeakMap<aval<IBuffer>, Map<string, aval<IBuffer>>>();

/**
 * Make a tight, offset-0 BufferView the arena can ingest. Tight/broadcast
 * views pass through unchanged; interleaved or offset views are gathered
 * into a fresh packed host buffer (every `stride` bytes from `offset`,
 * `byteSize` each) wrapped as a new whole-buffer view.
 */
function tightenBufferView(bv: BufferView): BufferView {
  if (bv.singleValue !== undefined) return bv; // broadcast → lowered to a uniform
  const byteSize = bv.elementType.byteSize;
  const offset = bv.offset ?? 0;
  const stride = (bv.stride !== undefined && bv.stride > 0) ? bv.stride : byteSize;
  if (offset === 0 && stride === byteSize) return bv; // already tight
  let byKey = deinterleaveCache.get(bv.buffer);
  if (byKey === undefined) { byKey = new Map(); deinterleaveCache.set(bv.buffer, byKey); }
  const key = `${offset}:${stride}:${byteSize}`;
  let tight = byKey.get(key);
  if (tight === undefined) {
    tight = bv.buffer.map((ib: IBuffer): IBuffer => {
      if (ib.kind !== "host") {
        throw new Error("heapAdapter: interleaved attribute buffer flipped to native GPUBuffer; classifier should have caught this");
      }
      const src = bytesOf(ib.data);
      const count = Math.max(0, Math.floor((src.byteLength - offset - byteSize) / stride) + 1);
      const out = new Uint8Array(count * byteSize);
      for (let i = 0; i < count; i++) {
        const s = offset + i * stride;
        out.set(src.subarray(s, s + byteSize), i * byteSize);
      }
      return IBuffer.fromHost(out);
    });
    byKey.set(key, tight);
  }
  return BufferView.ofBuffer(tight, bv.elementType);
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
  // 1. Inputs map: vertex attributes (BufferView) + uniforms, pulled
  //    SHADER-DRIVEN from the providers. We compile `ro.effect` (cached
  //    per Effect by the module-level compile cache) to discover the
  //    names it declares, then `tryGet` exactly those — so a lazy
  //    uniform provider (the Sg layer's auto-injected derived trafos)
  //    only ever materialises the ~2-4 trafos an effect actually reads,
  //    not all ~15. Compiling without a `fragmentOutputLayout` yields
  //    the *unreduced* interface — a superset of what the heap
  //    drawHeader ends up with — so we never under-provide; the
  //    over-provided handful (uniforms that feed only pruned outputs)
  //    is harmless (no drawHeader field → ignored).
  const vAttr = asAttributeProvider(ro.vertexAttributes);
  const uProv = asUniformProvider(ro.uniforms);
  // The names this effect declares (post link + DCE) — via the
  // localStorage-backed `compileHeapEffect` cache rather than a raw
  // `effect.compile()`, so a warm reload skips the optimiser pipeline
  // here too. `schema.uniforms` already merges uniform-block fields +
  // loose uniforms and drops names shadowed by an attribute.
  const schema = compileHeapEffect(ro.effect).schema;
  const inputs: { [name: string]: aval<unknown> | unknown } = {};
  for (const a of schema.attributes) {
    const bv = vAttr.tryGet(a.name);
    if (bv !== undefined) inputs[a.name] = isBufferView(bv) ? tightenBufferView(bv) : bv;
  }
  const pullUniform = (name: string): void => {
    if (Object.prototype.hasOwnProperty.call(inputs, name)) return;
    const av = uProv.tryGet(name);
    if (av !== undefined) inputs[name] = av;
  };
  for (const u of schema.uniforms) pullUniform(u.name);
  // §7 derived-uniforms constituents — cheap (these are the raw
  // `state.model/view/proj` avals, no `compose`/`inverse`) and the
  // compute pre-pass needs them even when the effect itself doesn't
  // declare them. No-ops if the provider doesn't carry them.
  for (const n of ["ModelTrafo", "ViewTrafo", "ProjTrafo"]) pullUniform(n);

  // 1b. Instance attributes — provider is map-backed (user-supplied via
  //     `Sg.instanced({attributes})`), so enumerate its names. Threaded
  //     via the heap path's per-RO instancing fast path.
  let instanceAttributes: { [name: string]: aval<unknown> | unknown } | undefined;
  if (ro.instanceAttributes !== undefined) {
    const iAttr = asAttributeProvider(ro.instanceAttributes);
    const names = [...iAttr.names()];
    if (names.length > 0) {
      instanceAttributes = {};
      for (const n of names) {
        const bv = iAttr.tryGet(n);
        if (bv !== undefined) instanceAttributes[n] = isBufferView(bv) ? tightenBufferView(bv) : bv;
      }
    }
  }

  // DrawCall: read once (snapshot — like instanceCount/vertexCount).
  //    Slice offsets fold in here: firstIndex → record indexStart,
  //    indexCount → record slice length, baseVertex → baked into the
  //    transcoded index values (so it needs no record field).
  const dc = ro.drawCall.getValue(token);
  const baseVertex = dc.kind === "indexed" ? dc.baseVertex : 0;

  // 2. Indices: BufferView → aval<Uint32Array>. Map the underlying
  //    IBuffer aval; the heap path's IndexPool will key on this aval
  //    (and on baseVertex, since it's baked into the values).
  //    Non-indexed ROs (no `ro.indices`) carry no index aval — the spec
  //    then sets `vertexCount` and the megacall decodes the vertex directly.
  const indices: aval<Uint32Array> | undefined =
    ro.indices !== undefined ? indicesAvalFor(ro.indices.buffer, ro.indices.elementType.indexFormat ?? "uint32", baseVertex) : undefined;

  // 3. Texture/sampler: single-pair v1. The Sg compile layer binds
  //    each texture aval under both `name` and `${name}_view` so the
  //    WGSL schema's binding shape doesn't matter at scene time — that
  //    leaves us with two HashMap entries pointing at the same aval.
  //    Dedupe by identity before applying the single-pair rule.
  // `let` + nulled after use: the spec's long-lived closures share
  // this scope context — a leftover Set here is per-RO ballast.
  let distinctTexAvals: Set<aval<ITexture>> | undefined = new Set<aval<ITexture>>();
  ro.textures.iter((_n, av) => { distinctTexAvals!.add(av as aval<ITexture>); });
  let distinctSamplerAvals: Set<aval<ISampler>> | undefined = new Set<aval<ISampler>>();
  // Shader-defined sampler state (from a `sampler2d { filter …; addressU … }`
  // builder, carried through the IR) overrides the scene's default sampler.
  const stateBinding = schema.samplers.find(b => b.state !== undefined);
  if (stateBinding?.state !== undefined) {
    const desc = samplerDescriptorFromState(stateBinding.state);
    if (stateBinding.wgslType === "sampler_comparison") {
      // Shadow sampler: the binding requires a compare function. WebGPU
      // also forbids anisotropy unless all filters are linear; keep the
      // state's choice (samplerDescriptorFromState already clamps).
      (desc as { compare?: GPUCompareFunction }).compare =
        compareFunctionOf(stateBinding.state.comparison ?? "LessOrEqual");
    }
    distinctSamplerAvals.add(
      AVal.constant(ISampler.fromDescriptor(desc)) as aval<ISampler>,
    );
  } else {
    ro.samplers.iter((_n, av) => { distinctSamplerAvals!.add(av as aval<ISampler>); });
  }
  let textures: HeapTextureSet | undefined;
  if (distinctTexAvals.size === 1 && distinctSamplerAvals.size === 1) {
    const texAval = [...distinctTexAvals][0]!;
    const samplerAval = [...distinctSamplerAvals][0]!;
    const texVal = texAval.getValue(token);
    const samplerVal = samplerAval.getValue(token);
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
      // The pool.acquire above bumped refcount to 1; release it
      // immediately so the entry's refcount tracks ACTUAL drawHeader
      // usage, not adapter-call count. heapScene.addDraw bumps refcount
      // via incRef per cycle (paired with our release closure). Without
      // this drop, every cached HeapDrawSpec (re-introduced across
      // aset add/remove cycles) leaves the atlas entry pinned forever.
      pool.release(acq.ref);
      // Heap-side re-acquire hook (attached after the literal so the
      // strict HeapTextureSet typing stays clean): if the entry was
      // evicted between heap remove/add cycles, addDraw acquires a
      // fresh sub-rect and uses this to redirect our per-aval cell to
      // the new ref, so the release closure stops freeing the long-
      // evicted one.
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
        // Captured for the heap-side re-acquire path (see addDraw atlas
        // bucket: when the pool entry was evicted while this spec sat
        // idle, the pool re-acquires a fresh sub-rect and MUST also
        // re-upload the texture pixels — otherwise the drawHeader
        // points at uninitialized atlas memory (or worse, leftover
        // pixels from the just-evicted neighbor).
        ...(dims.host !== undefined ? { host: dims.host } : {}),
      };
      (textures as unknown as { __retarget: (r: number) => void }).__retarget = (r: number) => { cell.ref = r; };
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
  } else if (distinctTexAvals.size > 0 || distinctSamplerAvals.size > 0) {
    throw new Error(
      `heapAdapter: RO has ${distinctTexAvals.size} distinct texture aval(s) and ${distinctSamplerAvals.size} distinct sampler aval(s); ` +
      `single-pair only in v1 (classifier should have caught this)`,
    );
  }
  // Drop the dedup scratch — see the `let` note above.
  distinctTexAvals = undefined;
  distinctSamplerAvals = undefined;

  // 4. DrawCall geometry. Classifier validates fields are heap-
  //    compatible (instanceCount≥1, firstInstance=0, non-indexed firstVertex=0).
  if (dc.kind === "indexed" && indices === undefined) {
    throw new Error("heapAdapter: indexed drawCall without indices");
  }
  if (dc.kind === "non-indexed" && indices !== undefined) {
    throw new Error("heapAdapter: non-indexed drawCall with an index buffer");
  }
  // Indexed → carry the index aval + the firstIndex/indexCount slice (folded
  // into the record's indexStart/indexCount; baseVertex is already baked into
  // the index values). Non-indexed → carry the vertex count; the megacall uses
  // the local vertex index directly (no index lookup).
  const geom = dc.kind === "indexed"
    ? { indices: indices!, firstIndex: dc.firstIndex, indexCount: dc.indexCount }
    : { vertexCount: dc.vertexCount };

  return {
    effect: ro.effect,
    ...(ro.pickId !== undefined ? { pickId: ro.pickId } : {}),
    pipelineState: ro.pipelineState,
    inputs,
    ...(instanceAttributes !== undefined ? { instanceAttributes } : {}),
    ...(dc.instanceCount > 1 ? { instanceCount: dc.instanceCount } : {}),
    ...geom,
    ...(textures !== undefined ? { textures } : {}),
    ...(ro.modeRules !== undefined ? { modeRules: ro.modeRules } : {}),
    ...(ro.modelChain !== undefined ? { modelChain: ro.modelChain } : {}),
    // Pass `active` through unforced so the heap path can subscribe
    // and react to flips without re-running the spec adapter.
    ...(ro.active !== undefined ? { active: ro.active } : {}),
  };
}
