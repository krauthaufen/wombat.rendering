# Large host textures on the heap: a page of their own

Status: design (2026-11-12). Asked for by PoEtry (2432² sprite sheets under 9,000 sprites) and the
user: "ideally the heap would just accept large textures and make them a chunk on their own since
they often are atlases already".

## Today

- A host texture up to `ATLAS_MAX_DIM` (1024) a side is packed into the atlas: per format one
  `texture_2d_array` of `pageSize²` layers (4096² on desktop), bound at a fixed binding of every
  atlas bucket; the draw header carries the placement (layer, origin, size, mip count, sampler
  bits) and the shader samples through it (gutters, wrap emulated, mips in an Iliffe column).
- A larger one, or a format without an atlas page, makes `isHeapEligible` reject the draw: it goes
  to the legacy renderer, one draw per RenderObject. Since 95e531b they share the GPU texture, but
  every draw is still a legacy draw (PoEtry: thousands).
- The Tier-L "standalone" branch (`kind: "gpu"` textures, bucket keyed by texture, its own
  standalone shader variant) exists in heapScene but is not reached by host textures: a hybrid
  scene compiles textured effects atlas-sampled.

## Design

A large host texture becomes a **page of its own**: an atlas-shaped texture sized to it.

- **Storage.** `AtlasPool` gets a second page kind, a *solo page*: a `texture_2d_array` with one
  layer, sized to the one entry it holds with the atlas layout (2 px gutters; with mips the Iliffe
  column beside mip 0: `(W+4) + (W/2+4)` wide). It is created when the first draw of that source
  arrives and filled with the same upload and gutter/mip kernels as an atlas entry. One solo page
  per source (keyed like atlas entries: by source, so draws that only share the value share it),
  refcounted, destroyed when its last draw goes (no LRU: they are big).
- **Draws.** The draw header is the atlas placement: layer 0, the whole page's inner rectangle,
  mips and sampler bits as for any atlas entry. The shader is the atlas variant unchanged.
- **Binding.** A solo page cannot be a layer of the format's array (its size differs), so the
  bucket binds it *in place of* the format array at the same binding. The bucket key becomes
  (effect family, pipeline, chunk, solo page id) for draws on a solo page; atlas draws keep their
  key. Decided when the draw is added: no per-frame case distinction, no shader change.
- **Wrap and filtering.** Kept as the atlas does it (shader emulation inside the placement, filter
  and wrap from the header's format bits), so one shader serves both. Real sampler addressing would
  need a second sampler binding per bucket and a shader branch; the emulation already works at the
  borders thanks to the gutters.
- **Standalone branch.** Left as it is for `kind: "gpu"` textures (user-managed resources keep
  their own path); solo pages replace legacy for host textures only.
- **Reactivity.** A texture aval that changes repacks as atlas entries do; when its size moves it
  across the cap, the draw moves between an atlas entry and a solo page (remove + add, as a repack
  that changes bucket).
- **Out of memory.** The solo page is allocated in the same device-error scope as atlas growth;
  if it fails the draw falls back to legacy for that texture (as atlas-full does today: the draw
  stays visible), retried when memory frees.
- **Eligibility** then rejects only what cannot be sampled this way: cube (packed elsewhere),
  array, 3D, external/video, and sizes whose solo page exceeds `maxTextureDimension2D`.

## Tests

Real GPU: a texture over the cap under many draws (one GPU texture, heap draws, right pixels); two
large textures in one scene (two buckets); large and atlas textures mixed; a texture that changes
(and one that crosses the cap); removal (page destroyed); allocation failure falls back. Unit:
routing (atlas / solo / legacy) by size, format and kind. Then PoEtry's full tree before/after.
