// Megacall GPU prefix-sum compute shader + record-layout constants.
//
// Four-pass scan over the bucket's drawTable:
//   1. scanTile         — per-tile Blelloch scan, writes per-tile sums
//   2. scanBlocks       — exclusive scan over the per-tile sums →
//                         tile offsets + total emit count
//   3. addOffsets       — add tile offsets back into per-record
//                         firstEmit values
//   4. buildTileIndex   — per-tile binary-search-fold for the VS
//                         prelude (mapping vertex_index → drawIdx)
//
// All four entry points share one WGSL module + one binding layout
// (defined below). The host runs them in sequence with the same bind
// group; only the dispatch shape and entry point change per pass.

export const SCAN_TILE_SIZE = 512;
export const SCAN_WG_SIZE   = 256;
/** numBlocks ≤ TILE_SIZE — single-pass blockOffsets fits in shared memory. */
export const SCAN_MAX_RECORDS = SCAN_TILE_SIZE * SCAN_TILE_SIZE;

/** Tile size for the firstDrawInTile binary-search-fold index. */
export const TILE_K = 64;

/** drawTable record width: (firstEmit, drawIdx, indexStart, indexCount, instanceCount). */
export const RECORD_U32   = 5;
export const RECORD_BYTES = RECORD_U32 * 4;

export const HEAP_SCAN_WGSL = `
struct Params {
  numRecords: u32,
  numBlocks:  u32,
  _pad0:      u32,
  _pad1:      u32,
};

struct Record {
  firstEmit:     u32,
  drawIdx:       u32,
  indexStart:    u32,
  indexCount:    u32,
  instanceCount: u32,
};

@group(0) @binding(0) var<storage, read_write> drawTable:        array<Record>;
@group(0) @binding(1) var<storage, read_write> blockSums:        array<u32>;
@group(0) @binding(2) var<storage, read_write> blockOffsets:     array<u32>;
@group(0) @binding(3) var<storage, read_write> indirect:         array<u32>;
@group(0) @binding(4) var<uniform>             params:           Params;
@group(0) @binding(5) var<storage, read_write> firstDrawInTile:  array<u32>;

const TILE_SIZE: u32 = 512u;
const WG_SIZE:   u32 = 256u;
const TILE_K:    u32 = 64u;

var<workgroup> sdata: array<u32, 512>;

fn blellochScan(tid: u32) {
  var offset: u32 = 1u;
  for (var d: u32 = TILE_SIZE >> 1u; d > 0u; d = d >> 1u) {
    workgroupBarrier();
    if (tid < d) {
      let ai = offset * (2u * tid + 1u) - 1u;
      let bi = offset * (2u * tid + 2u) - 1u;
      sdata[bi] = sdata[bi] + sdata[ai];
    }
    offset = offset * 2u;
  }
  if (tid == 0u) { sdata[TILE_SIZE - 1u] = 0u; }
  for (var d: u32 = 1u; d < TILE_SIZE; d = d * 2u) {
    offset = offset >> 1u;
    workgroupBarrier();
    if (tid < d) {
      let ai = offset * (2u * tid + 1u) - 1u;
      let bi = offset * (2u * tid + 2u) - 1u;
      let t = sdata[ai];
      sdata[ai] = sdata[bi];
      sdata[bi] = sdata[bi] + t;
    }
  }
  workgroupBarrier();
}

@compute @workgroup_size(WG_SIZE)
fn scanTile(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
  let tid = lid.x;
  let blockOff = wgid.x * TILE_SIZE;
  let n = params.numRecords;
  let i0 = blockOff + tid;
  let i1 = blockOff + tid + WG_SIZE;
  var v0: u32 = 0u;
  var v1: u32 = 0u;
  if (i0 < n) { v0 = drawTable[i0].indexCount * drawTable[i0].instanceCount; }
  if (i1 < n) { v1 = drawTable[i1].indexCount * drawTable[i1].instanceCount; }
  sdata[tid]           = v0;
  sdata[tid + WG_SIZE] = v1;
  workgroupBarrier();
  blellochScan(tid);
  if (i0 < n) { drawTable[i0].firstEmit = sdata[tid]; }
  if (i1 < n) { drawTable[i1].firstEmit = sdata[tid + WG_SIZE]; }
  if (tid == WG_SIZE - 1u) {
    blockSums[wgid.x] = sdata[tid + WG_SIZE] + v1;
  }
}

@compute @workgroup_size(WG_SIZE)
fn scanBlocks(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let n = params.numBlocks;
  let i0 = tid;
  let i1 = tid + WG_SIZE;
  var v0: u32 = 0u;
  var v1: u32 = 0u;
  if (i0 < n) { v0 = blockSums[i0]; }
  if (i1 < n) { v1 = blockSums[i1]; }
  sdata[tid]           = v0;
  sdata[tid + WG_SIZE] = v1;
  workgroupBarrier();
  blellochScan(tid);
  if (i0 < n) { blockOffsets[i0] = sdata[tid]; }
  if (i1 < n) { blockOffsets[i1] = sdata[tid + WG_SIZE]; }
  workgroupBarrier();
  if (tid == 0u) {
    if (n > 0u) {
      let lastIdx = n - 1u;
      let total = blockOffsets[lastIdx] + blockSums[lastIdx];
      indirect[0] = total;
    } else {
      indirect[0] = 0u;
    }
    indirect[1] = 1u;
    indirect[2] = 0u;
    indirect[3] = 0u;
  }
}

@compute @workgroup_size(WG_SIZE)
fn addOffsets(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
  let tid = lid.x;
  let blockOff = wgid.x * TILE_SIZE;
  let n = params.numRecords;
  let off = blockOffsets[wgid.x];
  let i0 = blockOff + tid;
  let i1 = blockOff + tid + WG_SIZE;
  if (i0 < n) { drawTable[i0].firstEmit = drawTable[i0].firstEmit + off; }
  if (i1 < n) { drawTable[i1].firstEmit = drawTable[i1].firstEmit + off; }
}

@compute @workgroup_size(WG_SIZE)
fn buildTileIndex(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tileIdx = gid.x;
  // totalEmit is computed by scanBlocks into indirect[0]; reading it
  // from indirect avoids a separate uniform/storage round-trip.
  let totalEmit = indirect[0];
  let numTiles = (totalEmit + TILE_K - 1u) / TILE_K;
  if (tileIdx > numTiles) { return; }
  if (params.numRecords == 0u) {
    if (tileIdx == 0u) { firstDrawInTile[0] = 0u; }
    return;
  }
  if (tileIdx == numTiles) {
    // Sentinel for the open upper bound — the LAST VALID SLOT, not
    // numRecords. The render VS uses
    //     hi = firstDrawInTile[_tileIdx + 1u]
    // and the binary search treats hi as INCLUSIVE. If the sentinel
    // were numRecords (one past last), the search would drag lo into
    // the OOB slot for emits in the last tile, since drawTable reads
    // past recordCount return 0 (binding size clamping) and 0 ≤ emit
    // is always true. Visible symptom: the LAST few emits in the
    // bucket land on slot=numRecords (drawIdx=0, indexCount=0 → /-by-
    // zero) → degenerate / cross-RO triangle stitched to slot 0.
    firstDrawInTile[tileIdx] = params.numRecords - 1u;
    return;
  }
  let tileStart = tileIdx * TILE_K;
  var lo: u32 = 0u;
  var hi: u32 = params.numRecords - 1u;
  loop {
    if (lo >= hi) { break; }
    let mid = (lo + hi + 1u) >> 1u;
    if (drawTable[mid].firstEmit <= tileStart) { lo = mid; } else { hi = mid - 1u; }
  }
  firstDrawInTile[tileIdx] = lo;
}
`;
