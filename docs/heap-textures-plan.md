# Heap renderer — texture management plan

The single highest-value remaining feature for the heap path. Goal:
*support heterogeneous-texture scenes without ballooning GPU memory
or fragmenting buckets back to one-per-texture.*

This supersedes the "per-size `texture_2d_array` binning" sketch in
`heap-future-work.md` §9. That approach wastes memory in two ways:

- A 200×200 icon binned into a 256² array layer wastes ~38% per
  layer.
- A 1100×800 photo binned into a 2048² layer wastes ~78%.

Across hundreds of textures the waste is hundreds of MB. Not
acceptable on mobile, and a real concern even on desktop.

## Design principles

1. **Pack, don't bin.** Many textures share a single physical
   `GPUTexture` via 2D rectangle packing inside it. The waste per
   page is the residual after packing, not the next-pow2 round-up.
2. **Tier by size and capability.** Small/non-mippable textures pack
   into shared atlas pages. Large or mip-needing textures stay
   standalone. The threshold and the tier rules are explicit, not a
   single one-size-fits-all policy.
3. **Reactive residency.** Just as RO membership is reactive (aset
   delta-driven), texture residency is too. When the live set
   changes, packed pages get sub-rect freed and reused; standalone
   textures get destroyed when refcount hits zero. No periodic
   garbage collection if the reactive system handles it correctly.
4. **Spirit-aligned with arena allocation.** The atlas page is a
   `GrowBuffer`-equivalent with rectangle freelist instead of byte
   freelist. `pageBuf.acquire(width, height)` returns a sub-rect
   ref; `pageBuf.release(ref)` frees it. Same identity-driven
   sharing pattern (key on the source `aval<ImageData>`), same
   refcount-on-aval pattern.
5. **Bucket key folds atlas page identity, not individual texture
   identity.** ROs whose textures live on the same atlas page share a
   bucket. ROs whose textures are standalone bucket per texture
   (today's behavior, fine for the few large ones).

## Tiering

Three tiers, decided at addDraw time from the source texture's
descriptor:

### Tier S (small, packed, no mips)

- Source dimensions ≤ `ATLAS_MAX_DIM` (e.g. 1024×1024).
- Source has no mip chain OR caller declares it doesn't need one.
- Common formats: `rgba8unorm`, `rgba8unorm-srgb`, `bc7-rgba-unorm`
  (depending on compression policy).

**Storage:** one or more "atlas pages." Each page is a fixed-size
2D texture (e.g. 4096×4096 in `rgba8unorm`). Pages allocated on
demand when no existing page can fit a new sub-rect.

**Per-page bookkeeping:**
- 2D rectangle allocator. **Reference implementation already exists
  in Aardvark**: `Aardvark.Geometry.TexturePacking<'a>` at
  `aardvark.base/src/Aardvark.Geometry/TexturePacking.fs`. Public
  API (`TryAdd(id, size) → Option<packing'>`, `Remove(id)`,
  `Free` / `Used` / `Occupancy`) is exactly the shape we need:
  - **Persistent / immutable** — `TryAdd` returns a new packer.
    Fits wombat's reactive model naturally; each scene snapshot can
    hold its packer state in an aval.
  - **Removal supported** — handles aset deltas as expected.
  - **BvhTree2d-backed free-space queries** — O(log N) lookups
    instead of the linear-freelist scan typical of MaxRects
    implementations elsewhere. Pays off at thousands of sub-rects.
  - `allowRotate` flag for 90° auto-rotation during packing.

  Plan: port the algorithm + structure to TS. Tests are
  FsCheck-property-based at
  `aardvark.base/src/Tests/Aardvark.Geometry.Tests/TexturePacking.fs`
  — port those too (vitest's property-based testing via fast-check)
  to lock in the same correctness guarantees.

  This saves ~1–2 weeks of implementation + bug-hunting compared to
  writing MaxRects from scratch. Also keeps the algorithm consistent
  between native Aardvark and wombat — future cross-checks and
  feature parity become trivial.

  TileRenderer has a near-clone at
  `TileRenderer/src/TileRenderer.Base/TexturePacking.fs` — useful as
  a secondary reference for "what consumers actually needed."

- Refcount per allocated sub-rect (multiple ROs may share a
  texture identity → one sub-rect, N references).

**RO-side:**
- DrawHeader gains `(atlasPageId, layerIdx=0, uvScale: vec2,
  uvBias: vec2)` for each texture binding.
- Bucket key includes `(atlasPageIdSet)` rather than individual
  texture identities. Two ROs whose textures live on the same set
  of pages share a bucket.

**No mips.** Use cases: icons, glyphs, markers, UI elements,
diagrammatic features. Anisotropic and trilinear filtering are
disabled at the sampler level for atlas-page samplers; bilinear
only.

### Tier M (medium, packed, custom mip chain)

- Source dimensions ≤ `ATLAS_MAX_DIM`.
- Caller wants mip filtering.

**Storage:** packed into atlas pages, but the page has a
pre-computed mip pyramid baked alongside the base layer. Each
sub-rect's mip chain is *also* sub-packed: at mip level k, the
sub-rect lives at `(x >> k, y >> k)` with `(w >> k, h >> k)` within
the page's mip-k image.

**Sub-rect padding:** N-pixel border around each sub-rect (mirror
or clamp), to prevent mip downsampling from bleeding neighbors. N
= pad size for mip count we want; typically 4 pixels for ~5 mip
levels.

**Sampling shader-side:** standard `textureSampleLevel(...)` with
hardware mip filter, BUT the sampling needs to clamp UVs to the
sub-rect at each mip level. Either:
- Software clamp in WGSL: `uv = clamp(uv, sub.minUv + 0.5/dim,
  sub.maxUv - 0.5/dim)`. One extra ALU per sample.
- OR rely on padding to absorb the bleed; risky for high
  anisotropy.

**Cost:** per-sub-rect mip pyramid pre-computation (CPU or compute
shader) at upload time. Standard mip-gen + careful boundary
handling.

### Tier L (large, standalone)

- Source dimensions > `ATLAS_MAX_DIM`, OR
- Format incompatible with atlas pages (e.g. floating-point HDR
  while atlas is 8-bit), OR
- Mutable texture (frequent CPU updates) where atlas churn would
  thrash packing.

**Storage:** one `GPUTexture` per source. Same as today's RO
texture handling.

**RO-side:** bucket key includes the texture identity. Each
distinct standalone texture → distinct bucket. Acceptable because
there are few of these per scene (typically <20).

## Atlas page format and capacity

**Page size:** 4096×4096. Reasonable balance between packing
efficiency (large enough to fit big textures) and not requiring
exotic device limits (4096 is universal in WebGPU).

**Format policy — per-format page sets, with limited promotion:**

The naive answer ("everything → rgba8unorm") wastes memory and
breaks sRGB. A 256² r8unorm mask is 64 KB; promoted to rgba8unorm
that's 256 KB — 4× waste. At thousands of small single-channel
textures (glyph SDFs, point splats, brush stamps) this adds up to
hundreds of MB of pure padding. And mixing sRGB content into a
linear page loses the hardware gamma decode.

So: **each page format gets its own page set.** ROs whose textures
reference different page formats still work — the bucket binds
multiple page sets, and the shader declares each as a separate
`binding_array<texture_2d<f32>, N>`.

Page format types:

1. `rgba8unorm-srgb` — color textures, the common case. Most
   icons, photos, color decals.
2. `rgba8unorm` — linear color data, normal maps stored as 4-ch.
3. `r8unorm` — single-channel masks, alpha sprites, glyph SDFs.
4. `rg8unorm` — two-channel data (normal-map XY, custom encodings).
5. (v2) `rgba16float` / `rg16float` for HDR. Almost certainly
   Tier L (large + unique) anyway.

**Promotion rules (limited):**

- `rgb8` (no actual WebGPU format) → promoted to rgba8unorm or
  rgba8unorm-srgb at upload. 33% memory penalty per such texture;
  unavoidable.
- `bgra8unorm` → swizzled to rgba8unorm at upload. Same memory.
- Otherwise: source format → same-format page.

**Format inference at addDraw:**

- Default policy: PNG/JPEG → rgba8unorm-srgb. Color is the common
  case; getting hardware gamma for free is the win.
- Explicit override: `ITexture.fromUrl(url, { format: "linear" })`,
  `{ format: "r8" }`, etc. The user knows when their texture is
  data, not color.
- Auto-detect 1-channel grayscale PNGs → r8unorm if the user hasn't
  overridden.

**MVP scope (phase 1–3): rgba8unorm-srgb + rgba8unorm only.**
Two page sets. Covers 90%+ of real scene texture content (color
images dominate). r8/rg8 page support comes in phase 5 alongside
other format optimizations. For MVP, a 64×64 alpha-only icon DOES
get bloated to 4 channels — acceptable trade-off; profile real
workloads to decide whether the extra format complexity is worth
the per-channel memory savings.

**Capacity per page:** 4096² × 4 bytes = 64 MB per color page.
With mips: 64 MB × 1.33 = ~85 MB. A scene with ~5 atlas pages →
~425 MB texture memory. Bounded.

**Number of pages:** unbounded in principle; allocate on demand.
Each page = one element in the bucket key's `atlasPageIdSet`. If
the bucket joins ROs whose textures span 3 pages, the bind group
binds 3 textures + N samplers. Bind group capacity: WebGPU's
default is 4 textures per stage; we have 8 in fragment. Plenty.

**Cap to bind-group size:** if a bucket would need more atlas
pages than fit in one bind group, split into multiple buckets.
Fine for v1; in practice the same set of pages serves many ROs.

## Reactive residency

Each atlas page has its own `aval<unknown>` representing its content
state. Sub-rect allocations are keyed on the source `aval<ImageData>`
identity.

**Pool wiring** (mirrors UniformPool / IndexPool):

```ts
class AtlasPool {
  acquire(
    page: AtlasPage,
    sourceAval: aval<ImageData>,
    width: number, height: number,
  ): { rect: SubRect; ref: number };

  release(ref: number): void;  // refcount--
}
```

Identity sharing: 100 ROs all referencing `marker-icon.png` →
one acquire returns one sub-rect, refcount = 100. Removal of any RO
decrements; sub-rect freed when refcount reaches 0.

**Eviction policy:** LRU when total atlas memory exceeds budget.
Released sub-rects are immediately available for reuse. If a page
becomes entirely empty, it's destroyed and its ID retired (rare —
usually a page is partially live).

**Repacking:** as fragmentation builds up over long sessions, free
space gets non-contiguous. Two responses:
1. **Don't repack.** New allocations fail-over to a new page. Page
   count grows.
2. **Periodic compaction.** When fragmentation ratio (= used /
   capacity) drops below threshold (e.g. 60%), compact the page:
   allocate a fresh page, copy live sub-rects via
   `copyTextureToTexture`, update all refs in drawHeaders, retire
   the old page.

V1: option 1. V2: add option 2 if profiling shows it matters.

## RO drawHeader extensions

Per-texture binding in the schema becomes:

```
{ name: "albedo", kind: "texture-ref", wgslType: "texture_2d<f32>" }
```

Compiles to drawHeader fields:
- `albedoPageRef: u32` — per-bucket atlasPageId
- `albedoUvScale: vec2<f32>` — sub-rect dimensions normalized
- `albedoUvBias: vec2<f32>` — sub-rect origin normalized

Total: 5 × 4 = 20 bytes per textured RO per binding. For most ROs
with one albedo texture, +20 bytes per drawHeader. Negligible.

VS prelude reads these from the drawHeader (same as uniform refs).
FS samples:

```wgsl
let uv = clamp(in.uv, vec2(0.0), vec2(1.0));
let atlasUv = uv * uvScale + uvBias;
let color = textureSampleLevel(atlas[pageRef], sampler, atlasUv, lod);
```

`atlas[pageRef]` requires either:
- Bindless (not in WebGPU 1.0).
- A small statically-bound array of textures + dynamic indexing.
  WebGPU 1.0 supports `binding_array<texture_2d<f32>, N>` with
  uniform-control-flow indexing. Works for up to N atlas pages
  in one bind group.

For v1 we use the second: bind group declares
`binding_array<texture_2d<f32>, 8>` (or whatever the FS allows).
Each page assigned a slot in the array. Bucket key includes the
*page assignment* — which atlas pages are bound to which slots.
Slot count = `min(numPages, FS texture limit)`.

If a bucket would need more pages than slots, split. Or fall back
to standalone texture handling for the offending ROs.

## API surface (user-facing)

Existing wombat texture API stays unchanged. The runtime decides
the tiering at addDraw time:

```ts
const albedo: ITexture = ITexture.fromUrl("marker-icon.png");
const sampler: ISampler = ISampler.bilinear();

const ro = makeRO({
  effect: surface,
  textures: { albedo, sampler },
  ...
});
```

The runtime classifies `albedo`:
- Loaded image data fits Tier S / M criteria → atlas-pack.
- Otherwise → standalone.

User can override via texture-creation flags:
- `ITexture.fromUrl(url, { hint: "atlas" })` — force packing.
- `ITexture.fromUrl(url, { hint: "standalone" })` — force standalone
  (e.g. for a large background image known to be unique).

## Memory budget for a representative scene

- 200 marker icons (64–256², avg 128²): **fit in one 4096²
  atlas page** with room to spare. ~200 × 64 KB = 12.8 MB used,
  64 MB allocated. After a few hundred more icons, page fills,
  spawn page #2. (This is the "good case" the architecture
  optimizes for.)
- 30 mid-resolution decals/textures (512×512 with mips): **second
  atlas page** (Tier M with mip pyramid baking). ~30 × 1 MB
  = 30 MB used, 85 MB allocated.
- 5 large unique textures (4096² each, mipped): **standalone**, 5 ×
  85 MB = 425 MB.

Total: ~575 MB. Iphone with `requiredLimits` raised to adapter max
(1 GB on most modern iPhones) handles this. Default desktop
allocation has plenty of headroom.

Compare to the per-size-array scheme:
- 200 icons binned into 256² array layers: 200 × 256 KB = 50 MB
  *if perfectly packed*. In practice each layer is a full 256²,
  so 50 MB used out of 200 × 256² × 4 = 50 MB (no extra waste —
  layers are exactly 256²). But: array layer count limit (2048 in
  WebGPU). For 5 size classes × 256 layers each = 1280 layers
  total. Fits, barely.
- Per-size-array works ONLY for mostly-uniform-size collections.
  Real-world heterogeneity (the icon set has 64²-512² variance,
  the decal set has 256²-1024²) doesn't bin cleanly. Each
  texture binned to nearest-larger pow2 wastes 25–75%.

Atlas packing wins on memory for heterogeneous workloads. It also
wins on bind-group simplicity (one texture binding per atlas page,
not per size class).

## Implementation phasing

**Phase 1 — atlas infrastructure** (~2 weeks)
- Atlas page abstraction: `GPUTexture` + 2D rectangle allocator
  (MaxRects-BSSF, well-tested algorithm).
- `AtlasPool`: aval-keyed pool with refcounting; same shape as
  UniformPool/IndexPool.
- Image data ingest: from `ImageBitmap`, `ImageData`, fetch'd PNG/
  JPEG. Use `device.queue.copyExternalImageToTexture` for upload.
- Sub-rect to UV transform: stored in drawHeader as `(scale, bias)`.

**Phase 2 — Tier S integration** (~1 week)
- isHeapEligible accepts ROs whose texture fits Tier S criteria.
- heapAdapter classifies texture; routes to AtlasPool.
- DrawHeader schema extension for `(pageRef, uvScale, uvBias)` per
  texture binding.
- Bucket key includes atlas page set.
- Sampling in shader: WGSL prelude includes the
  `binding_array<texture_2d<f32>, N>` decl + indexing logic.

**Phase 3 — Tier L integration** (~3 days)
- Standalone path: existing per-RO bucket logic, just gated on
  Tier L classification.
- Eligibility: no change; bucket key already includes texture
  identity.

**Phase 4 — Tier M (mip support)** (~2 weeks)
- Padding logic in rectangle allocator (allocate w + 2*pad,
  h + 2*pad).
- Per-sub-rect mip pyramid baking via compute shader.
- Sampling shader: clamp-to-sub-rect logic at each mip.

**Phase 5 — Eviction + repacking** (~1 week)
- LRU policy on AtlasPool.
- Refcount-driven release path (fall out of UniformPool pattern).
- Optional: page compaction when fragmentation crosses threshold.

**Phase 6 — Polish** (~1–2 weeks)
- Tests covering all three tiers.
- Examples in heap-demo.
- Documentation update.
- Benchmark: same scene with N textures, before/after atlas.

**Total: ~7–9 weeks of focused work for the full plan.**

For an MVP that proves the architecture: **Phases 1–3 only, ~3
weeks.** No mips (Tier M deferred), simple page allocation, no
eviction (until you hit the budget). Already covers the most
impactful case (markers, icons, glyphs, simple textured props) and
unlocks heterogeneous-texture scenes for the heap path.

## Open questions

- **Compression**: should we accept BC7 / ASTC source textures and
  pack them into compressed atlas pages? More complex packing
  (block-aligned), but ~4× memory savings. Defer to v2.
- **Format negotiation**: when ROs request both linear and sRGB
  for the same logical texture, do we duplicate or do we pick one
  and gamma-correct in shader? Probably pick one and shader-gamma;
  costs one ALU.
- **Texture array sub-allocation**: instead of independent atlas
  pages, use a `texture_2d_array` where each layer is a
  full-resolution atlas page. Saves binding slots (one binding
  for many pages) but constrains all pages to identical
  size/format. Probably worth doing in v2; lower priority than
  the basic tiering work.
- **CPU-side cache of decoded image data** for fast re-upload after
  eviction. Trades RAM for re-decode cost. Sane defaults: keep
  decoded data for textures < 1 MB; evict from RAM for larger.
