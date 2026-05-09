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

### Tier 0 — escalates to legacy

Some ROs never enter the heap subset at all. `isHeapEligible` rejects
them up front and `HybridScene` routes them to `ScenePass` (the
per-RO legacy renderer with full user control). This is *before* the
Tier-S/M/L decision below — Tier 0 ROs are simply not classified.

Escalation rules (any one is sufficient):

- **`ITexture.kind === "gpu"`.** The user is managing a backend
  resource themselves — render target, video frame, environment
  probe, streaming photogrammetry tile. Heap-side handling
  (atlas-packing, rebadging, bucket-key folding) doesn't apply.
- **Texture extent > `LEGACY_MAX_DIM` on either side**
  (`LEGACY_MAX_DIM = 4096`). Above this threshold, even the heap's
  Tier-L "standalone" path stops being a win: better to give the RO
  its own dedicated GPUTexture in legacy than to host-upload a ~16M
  pixel image into heap-managed memory and risk bumping
  `maxTextureDimension2D`.
- **Texture is not 2D / has more than one array layer.** Cubemaps,
  2D arrays, 3D volumes. Heap shaders are 2D-only. (For
  `kind: "gpu"` this is read off the underlying `GPUTexture`'s
  `dimension` / `depthOrArrayLayers`. For `kind: "host"` the
  source descriptors are 2D-by-construction; the rule still
  defensively rejects raw sources with `depthOrArrayLayers > 1`.)

Note: rule 1 already covers most rule-3 cases — non-2D / arrayed
textures essentially only exist via user-supplied `fromGPU`. Rule 3
is belt-and-suspenders for any future host-side variant.

The remaining heap-eligible ROs feed into Tier S/M/L below.

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

### Tier M (medium, packed, embedded 1.5× mip pyramid)

- Source dimensions ≤ `ATLAS_MAX_DIM`.
- Caller wants mip filtering.

**Storage — embedded 1.5W × H mip pyramid per sub-rect:**

A previous version of this plan packed each mip level into a
separate atlas image (mip 0 in the page's mip-0 image, mip 1 in
the page's mip-1 image, etc.). That requires hardware mip filter
and careful per-mip clamp logic, AND it forces all sub-rects in
the atlas to the same number of mip levels. Two problems: per-
sub-rect mip count is a real workload variation we want to honor,
AND it makes the cost of "an atlas containing N textures, only
one of which wants mips" needlessly large.

The right layout: the classic **Iliffe / id Tech "1.5×1" mip
pyramid**, embedded in the atlas as a single rect per texture.

```
+---------------+-------+
|               |  m=1  |
|               |       |
|     m=0       +---+---+
|     (W × H)   |m=2|m=3|...
|               +---+
|               |
+---------------+-------+
       W            W/2 = W/2
       <--- 1.5W ---->
```

- Mip 0 occupies the left W×H block.
- Mip 1 (W/2 × H/2) occupies the top-right.
- Mip 2 (W/4 × H/4) below mip 1, then mip 3 below mip 2, etc.
- All mips combined fit in **1.5W × H**.

Total atlas footprint per texture: 1.5W × H instead of W × H. ~50%
storage overhead for mips, contiguous within the atlas, single
sub-rect per texture. The packer (TexturePacking) gets the rect
size 1.5W × H for mipped sources and W × H for non-mipped ones —
no other change to the packer itself.

Per-texture mip count is freely variable: a texture wanting only
3 mip levels packs `ceil(W * (1 + 0.5 + 0.25 + 0.125))` etc.; a
texture not wanting mips packs W × H. Hetereogeneous textures
coexist in one atlas page.

**Sub-rect padding:** 1–2 pixels of mirror/clamp bleed around
each sub-rect to absorb hardware bilinear filter at sub-rect
boundaries. Padding lives outside the 1.5W × H footprint
(packer asks for 1.5W+2pad × H+2pad).

**Sampling: software mip filter, hardware bilinear within mip:**

Hardware mip filter cannot be used (the GPU would walk the
texture's own mip chain, not our embedded pyramid). Instead the
shader does mip selection + lerp between two adjacent mips
manually, with hardware bilinear within each mip:

```wgsl
fn atlasSample(
  pageRef: u32, format: u32,
  origin: vec2<f32>,    // top-left of mip 0 in atlas, normalized
  size: vec2<f32>,      // mip 0 size in atlas, normalized
  numMips: u32,
  uv: vec2<f32>,
  addrU: u32, addrV: u32, // wrap modes (per binding, see below)
) -> vec4<f32> {
  // 1. Wrap mode applied in shader, before atlas-coord transform.
  let uvW = applyWrap(uv, addrU, addrV);

  // 2. LOD from screen-space derivatives.
  //    mip 0 is `size` in atlas-normalized units; we want LOD in
  //    "source-texture pixels per screen pixel" terms.
  let dx = dpdx(uvW * size);
  let dy = dpdy(uvW * size);
  let rho = max(length(dx), length(dy));
  let lod = clamp(log2(max(rho * f32(atlasPx), 1e-6)),
                  0.0, f32(numMips - 1));

  // 3. Two adjacent mip levels, lerp.
  let lo = u32(floor(lod));
  let hi = min(lo + 1u, numMips - 1u);
  let t  = lod - f32(lo);

  let a = sampleMip(pageRef, format, origin, size, lo, uvW);
  let b = sampleMip(pageRef, format, origin, size, hi, uvW);
  return mix(a, b, t);
}

fn sampleMip(pageRef, format, origin, size, k, uvW) -> vec4<f32> {
  // Mip k's region in the 1.5W × H pyramid:
  //   k == 0: at (origin, origin + size)
  //   k >= 1: at (origin + (size.x, sumPrev), size / 2^k)
  //   where sumPrev = sum_{j=1..k-1} size.y / 2^j
  let mipSize = size / pow(2.0, f32(k));
  let mipOrigin = mipOriginInPyramid(origin, size, k);
  let atlasUv = mipOrigin + uvW * mipSize;
  // hardware bilinear within mip; format-dispatch as planned.
  return select(
    textureSampleLevel(atlasLinear[pageRef], atlasSampler, atlasUv, 0.0),
    textureSampleLevel(atlasSrgb[pageRef],   atlasSampler, atlasUv, 0.0),
    format == 1u,
  );
}
```

`mipOriginInPyramid` is a closed-form expression — for the 1.5×1
layout it's a simple `select`-y formula. Roughly:
`mipOrigin.x = origin.x + size.x` for k≥1; `mipOrigin.y = origin.y +
size.y * (1 - 1/2^(k-1))` for k≥1. No table lookup needed.

**Cost vs hardware mip filter:**
- 2 `textureSampleLevel` calls instead of 1 `textureSample`.
- ~5–10 extra ALU ops for LOD + region computation.
- One `dpdx` + `dpdy` (which we'd issue anyway under hardware
  filter — `textureSample` does this internally).
- For non-mipped textures (numMips == 1): skip the lerp entirely;
  one sample, branch on `numMips`.

For the user, this is invisible. Their `textureSample(t, smp, uv)`
call gets rewritten to `atlasSample(...)` by the IR pass; the
shader they wrote just works.

**Mip pyramid generation:**

CPU- or compute-side, depending on what's available. For MVP, CPU
generation: download to ImageBitmap, downscale via canvas-2d at
each level, upload via `copyExternalImageToTexture` to the
appropriate sub-region. Slow on first upload but acceptable for
infrequent texture loads.

For v2: a compute shader that takes the mip-0 region as input and
writes mips 1..N into the pyramid sub-regions in-place. Much
faster, no CPU round-trip.

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

Compiles to drawHeader fields (the *full* mip+sampler-aware set):

- `albedoPageRef: u32` — index into the format's binding_array
- `albedoFormatBits: u32` — packed: bit 0 = format (0=linear, 1=srgb),
  bits 1..3 = numMips (0..7, where 0 means "no mips, single sample"),
  bits 4..7 = addrU (clamp/repeat/mirror), bits 8..11 = addrV,
  bits 12..15 = mag/min/mip filter (linear/nearest)
- `albedoOrigin: vec2<f32>` — top-left of mip-0 in atlas, normalized
- `albedoSize: vec2<f32>` — mip-0 size in atlas, normalized

Total: 4 + 4 + 8 + 8 = 24 bytes per textured RO per binding. With
the standalone-Tier-L path needing none of these (resolved
texture is bound directly), this only costs Tier-S ROs.

The packed `formatBits` u32 covers all "sampler state per RO"
needs without requiring a separate sampler binding per
configuration. The shader unpacks via bit-ops; one extra ALU per
field. Sub-millisecond aggregate impact.

VS prelude reads these from the drawHeader (same as uniform refs).
The IR rewriter substitutes user-shader `textureSample(albedo, smp,
uv)` calls with `atlasSample(albedo, uv)` calls that thread the
drawHeader fields through the helper defined earlier (with the
1.5×1 mip pyramid layout):

```wgsl
fn atlasSample(
  pageRef: u32, formatBits: u32,
  origin: vec2<f32>, size: vec2<f32>,
  uv: vec2<f32>,
) -> vec4<f32> {
  let format = formatBits & 0x1u;
  let numMips = (formatBits >> 1u) & 0x7u;
  let addrU   = (formatBits >> 4u) & 0xFu;
  let addrV   = (formatBits >> 8u) & 0xFu;
  // ... apply wrap, compute LOD, lerp two adjacent mips ...
}
```

User does not see this. They write
`textureSample(albedo, smp, uv)` and the atlas mechanism is
invisible — including wrap modes (clamp / repeat / mirror) and
mip filtering. The "sampler" in the user's source is reduced to
*sampler state*, which gets packed into formatBits at addDraw
time from the `ISampler` they handed in.

`atlas[pageRef]` indexing: WebGPU 1.0 supports
`binding_array<texture_2d<f32>, N>` with uniform-control-flow
indexing. Works for up to N atlas pages in one bind group.

For v1 we use this: bind group declares
`binding_array<texture_2d<f32>, 8>` per format (linear + srgb).
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

## Future work — non-2D textures

The current plan handles 2D textures only. Cubemaps, volumes
(3D), and texture arrays escalate to the legacy per-RO path via
`isHeapEligible`. Of these, **cubemaps will eventually want
atlasing** — environment maps, IBL prefilter chains, and shadow
cubes are common-enough cases that bucket-per-cubemap will start
hurting at scale.

**Sketch (when we get there):**

- Each face packs as its own 1.5W × H mip pyramid (same layout as
  Tier M today). All 6 faces of one cubemap allocate in one
  contiguous block — most natural is a 6-wide row of pyramids
  (9W × H per cubemap) or 2×3 grid (3W × 2H), pick whichever the
  packer is happier with.
- The drawHeader carries the cubemap's `origin` and per-face
  `size` (or one face size + a deterministic face layout offset
  formula). One extra `cubeFaceLayout: u32` field tells the
  shader which arrangement was chosen, so the math is uniform.
- Shader: `textureSampleCube(cube, sampler, dir)` rewrites to
  `atlasSampleCube(...)` which:
  1. Computes face index + face-local UV from `dir` (standard
     cubemap math: dominant axis selection, then 2D project).
  2. Selects the face's region in the atlas via the layout
     formula.
  3. Sample as a Tier-M 1.5×1 mip pyramid within that face.

For prefiltered IBL cubemaps with per-mip roughness baking, the
existing per-face mip pyramid already serves — each mip level is
a separately-baked roughness layer.

This pulls in shader complexity (cubemap projection math) but no
new pool/packer infrastructure beyond what Tier M already has.

**Volumes (3D textures) and 2D arrays** stay legacy-only. Volumes
have no clean atlas story (slicing a 3D dataset into 2D tiles
breaks trilinear filtering), and 2D arrays are typically used by
the user precisely because they want array semantics — atlasing
flattens that abstraction in ways that defeat the purpose.
