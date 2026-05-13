// Phase 5c.3 partition kernel — GPU-driven record routing.
//
// Each frame the kernel runs over a bucket's master record pool. For
// each record, it evaluates the rule (currently hard-coded: flip
// cullMode by sign of det(upperLeft3x3) of the record's ModelTrafo),
// picks the destination slot index (0 = declared, 1 = flipped),
// atomically reserves a write offset in that slot, and copies the
// record into `slotDrawTable[slotIdx][offset]`.
//
// Buffer layout (one bucket):
//
//   masterRecords: array<Record>            // CPU populates at addRO
//   numRecords:    u32 (uniform)
//   slot0Indirect: array<u32, 4>            // (indexCount, instanceCount=1, 0, 0)
//   slot1Indirect: array<u32, 4>            // ditto
//   slot0DrawTable: array<Record>           // partition output
//   slot1DrawTable: array<Record>           // partition output
//   slot0ScanParams: { numRecords: u32 ... }
//   slot1ScanParams: { numRecords: u32 ... }
//
// The existing scan (HEAP_SCAN_WGSL in heapScene/scanKernel.ts) runs
// per slot AFTER partition: it reads the slot's drawTable, computes
// firstEmit prefix sums, and writes the final indirect[0] = totalEmit.
//
// What this kernel does NOT do:
//   - Generalize beyond the determinant-flip-cull rule (v1 only).
//   - Handle >2 slots per bucket. (For multi-output rules, future
//     work: dynamic slot count, modeKey-table lookup.)
//   - Manage record removal — that's CPU at removeDraw time (swap-pop
//     the master pool).

import type { CullMode } from "../pipelineCache/index.js";
import { CULL_TO_U32 } from "./gpuKernel.js";

/** Record layout: 6 × u32 = 24 bytes per record. Matches the heap
 *  scan's record format (5 × u32) plus a `modelRef` byte offset. */
export const PARTITION_RECORD_U32   = 6;
export const PARTITION_RECORD_BYTES = PARTITION_RECORD_U32 * 4;

/** Per-bucket partition kernel WGSL. Two output slots: declared
 *  + flipped. */
export const PARTITION_FLIP_CULL_BY_DET_WGSL = /* wgsl */ `
struct Record {
  // firstEmit is GPU-written by the scan; leave as 0 here.
  firstEmit:     u32,
  drawIdx:       u32,
  indexStart:    u32,
  indexCount:    u32,
  instanceCount: u32,
  modelRef:      u32,
};
struct PartitionParams {
  numRecords:     u32,
  declaredCull:   u32,   // 0=none, 1=front, 2=back
  _pad0:          u32,
  _pad1:          u32,
};

@group(0) @binding(0) var<storage, read>        arena:           array<u32>;
@group(0) @binding(1) var<storage, read>        masterRecords:   array<Record>;
@group(0) @binding(2) var<uniform>              params:          PartitionParams;
// Slot 0 = declared cull mode. Slot 1 = flipped cull mode.
@group(0) @binding(3) var<storage, read_write>  slot0Count:      array<atomic<u32>>;  // length 1
@group(0) @binding(4) var<storage, read_write>  slot1Count:      array<atomic<u32>>;
// Slot draw tables hold scan-format records (5 × u32 per record:
// firstEmit, drawIdx, indexStart, indexCount, instanceCount). We
// drop modelRef on scatter — only the partition kernel needs it,
// and including it would force the scan to read a 6-u32 stride.
@group(0) @binding(5) var<storage, read_write>  slot0DrawTable:  array<u32>;
@group(0) @binding(6) var<storage, read_write>  slot1DrawTable:  array<u32>;

const SCAN_REC_U32: u32 = 5u;

fn loadUpper3x3(refBytes: u32) -> array<f32, 9> {
  let baseU32 = (refBytes + 16u) >> 2u;
  return array<f32, 9>(
    bitcast<f32>(arena[baseU32 + 0u]),  bitcast<f32>(arena[baseU32 + 1u]),  bitcast<f32>(arena[baseU32 + 2u]),
    bitcast<f32>(arena[baseU32 + 4u]),  bitcast<f32>(arena[baseU32 + 5u]),  bitcast<f32>(arena[baseU32 + 6u]),
    bitcast<f32>(arena[baseU32 + 8u]),  bitcast<f32>(arena[baseU32 + 9u]),  bitcast<f32>(arena[baseU32 + 10u]),
  );
}

fn det3x3(m: array<f32, 9>) -> f32 {
  return m[0] * (m[4] * m[8] - m[5] * m[7])
       - m[1] * (m[3] * m[8] - m[5] * m[6])
       + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
  // One thread total clears both atomics. Cheap pass.
  if (gid.x != 0u) { return; }
  atomicStore(&slot0Count[0], 0u);
  atomicStore(&slot1Count[0], 0u);
}

@compute @workgroup_size(64)
fn partition(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numRecords) { return; }
  let r = masterRecords[i];
  let m = loadUpper3x3(r.modelRef);
  let flipped = det3x3(m) < 0.0;
  // slotIdx 0 = declared (not flipped), 1 = flipped.
  // Atomic-add into the appropriate slot's count, then scatter the
  // record into the slot's drawTable at the reserved offset.
  if (flipped) {
    let off = atomicAdd(&slot1Count[0], 1u);
    let base = off * SCAN_REC_U32;
    slot1DrawTable[base + 0u] = 0u;              // firstEmit (scan rewrites)
    slot1DrawTable[base + 1u] = r.drawIdx;
    slot1DrawTable[base + 2u] = r.indexStart;
    slot1DrawTable[base + 3u] = r.indexCount;
    slot1DrawTable[base + 4u] = r.instanceCount;
  } else {
    let off = atomicAdd(&slot0Count[0], 1u);
    let base = off * SCAN_REC_U32;
    slot0DrawTable[base + 0u] = 0u;
    slot0DrawTable[base + 1u] = r.drawIdx;
    slot0DrawTable[base + 2u] = r.indexStart;
    slot0DrawTable[base + 3u] = r.indexCount;
    slot0DrawTable[base + 4u] = r.instanceCount;
  }
}
`;

/** WGSL helper: pack a Trafo3d's upper-left 3×3 + a CullMode enum
 *  into the partition kernel's expected uniform layout. */
export function cullModeToU32(c: CullMode): number {
  return CULL_TO_U32[c];
}
