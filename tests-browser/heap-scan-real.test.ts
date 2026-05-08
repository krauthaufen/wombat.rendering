// Validates the megacall prefix-sum compute shader on real WebGPU.
// Builds a drawTable with known indexCounts, dispatches scanTile +
// scanBlocks + addOffsets, reads firstEmit and indirect back, and
// compares against the CPU reference scan.

import { describe, expect, it } from "vitest";
import { requestRealDevice } from "./_realGpu.js";

const TILE_SIZE = 512;
const WG_SIZE = 256;

const HEAP_SCAN_WGSL = `
struct Params {
  numRecords: u32,
  numBlocks:  u32,
  _pad0:      u32,
  _pad1:      u32,
};

struct Record {
  firstEmit:  u32,
  drawIdx:    u32,
  indexStart: u32,
  indexCount: u32,
};

@group(0) @binding(0) var<storage, read_write> drawTable:    array<Record>;
@group(0) @binding(1) var<storage, read_write> blockSums:    array<u32>;
@group(0) @binding(2) var<storage, read_write> blockOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> indirect:     array<u32>;
@group(0) @binding(4) var<uniform>             params:       Params;

const TILE_SIZE: u32 = 512u;
const WG_SIZE:   u32 = 256u;

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
  if (i0 < n) { v0 = drawTable[i0].indexCount; }
  if (i1 < n) { v1 = drawTable[i1].indexCount; }
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
`;

interface ScanRig {
  device: GPUDevice;
  drawTableBuf: GPUBuffer;
  indirectBuf: GPUBuffer;
  paramsBuf: GPUBuffer;
  blockSumsBuf: GPUBuffer;
  blockOffsetsBuf: GPUBuffer;
  bg: GPUBindGroup;
  pTile: GPUComputePipeline;
  pBlocks: GPUComputePipeline;
  pAdd: GPUComputePipeline;
}

async function buildRig(device: GPUDevice, capRecords: number): Promise<ScanRig> {
  const numBlocksCap = Math.max(1, Math.ceil(capRecords / TILE_SIZE));
  const drawTableBuf = device.createBuffer({
    size: capRecords * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const indirectBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const paramsBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const blockSumsBuf = device.createBuffer({
    size: Math.max(16, numBlocksCap * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const blockOffsetsBuf = device.createBuffer({
    size: Math.max(16, numBlocksCap * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const module = device.createShaderModule({ code: HEAP_SCAN_WGSL });
  const pTile = device.createComputePipeline({ layout, compute: { module, entryPoint: "scanTile" } });
  const pBlocks = device.createComputePipeline({ layout, compute: { module, entryPoint: "scanBlocks" } });
  const pAdd = device.createComputePipeline({ layout, compute: { module, entryPoint: "addOffsets" } });
  const bg = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: drawTableBuf } },
      { binding: 1, resource: { buffer: blockSumsBuf } },
      { binding: 2, resource: { buffer: blockOffsetsBuf } },
      { binding: 3, resource: { buffer: indirectBuf } },
      { binding: 4, resource: { buffer: paramsBuf } },
    ],
  });
  return { device, drawTableBuf, indirectBuf, paramsBuf, blockSumsBuf, blockOffsetsBuf, bg, pTile, pBlocks, pAdd };
}

async function readU32(device: GPUDevice, buf: GPUBuffer, byteCount: number): Promise<Uint32Array> {
  const staging = device.createBuffer({ size: byteCount, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, byteCount);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

function cpuExclusiveScan(indexCounts: number[]): { firstEmit: number[]; total: number } {
  const firstEmit: number[] = [];
  let acc = 0;
  for (const c of indexCounts) { firstEmit.push(acc); acc += c; }
  return { firstEmit, total: acc };
}

async function runScan(rig: ScanRig, indexCounts: number[]): Promise<{ firstEmit: Uint32Array; indirect: Uint32Array }> {
  const n = indexCounts.length;
  const numBlocks = Math.max(1, Math.ceil(n / TILE_SIZE));
  const records = new Uint32Array(n * 4);
  for (let i = 0; i < n; i++) {
    records[i * 4 + 0] = 0xdeadbeef;        // firstEmit — should be overwritten
    records[i * 4 + 1] = i;                  // drawIdx
    records[i * 4 + 2] = 0;                  // indexStart
    records[i * 4 + 3] = indexCounts[i]!;    // indexCount
  }
  rig.device.queue.writeBuffer(rig.drawTableBuf, 0, records.buffer, records.byteOffset, records.byteLength);
  rig.device.queue.writeBuffer(rig.paramsBuf, 0, new Uint32Array([n, numBlocks, 0, 0]));
  // Pre-fill indirect with sentinel so we can see if the shader wrote it.
  rig.device.queue.writeBuffer(rig.indirectBuf, 0, new Uint32Array([0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef]));

  const enc = rig.device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setBindGroup(0, rig.bg);
  pass.setPipeline(rig.pTile);
  pass.dispatchWorkgroups(numBlocks, 1, 1);
  pass.setPipeline(rig.pBlocks);
  pass.dispatchWorkgroups(1, 1, 1);
  pass.setPipeline(rig.pAdd);
  pass.dispatchWorkgroups(numBlocks, 1, 1);
  pass.end();
  rig.device.queue.submit([enc.finish()]);

  const back = await readU32(rig.device, rig.drawTableBuf, n * 16);
  const firstEmit = new Uint32Array(n);
  for (let i = 0; i < n; i++) firstEmit[i] = back[i * 4 + 0]!;
  const indirect = await readU32(rig.device, rig.indirectBuf, 16);
  return { firstEmit, indirect };
}

describe("heap-scan prefix sum (real GPU)", () => {
  it("scans 1 record correctly", async () => {
    const device = await requestRealDevice();
    const rig = await buildRig(device, 16);
    const { firstEmit, indirect } = await runScan(rig, [42]);
    expect(Array.from(firstEmit)).toEqual([0]);
    expect(indirect[0]).toBe(42);
    expect(indirect[1]).toBe(1);
  });

  it("scans 10 records correctly", async () => {
    const device = await requestRealDevice();
    const rig = await buildRig(device, 32);
    const counts = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
    const { firstEmit, indirect } = await runScan(rig, counts);
    const ref = cpuExclusiveScan(counts);
    expect(Array.from(firstEmit)).toEqual(ref.firstEmit);
    expect(indirect[0]).toBe(ref.total);
  });

  it("scans full single tile (512 records) correctly", async () => {
    const device = await requestRealDevice();
    const rig = await buildRig(device, 512);
    const counts: number[] = [];
    for (let i = 0; i < 512; i++) counts.push((i * 7 + 3) % 41 + 1);
    const { firstEmit, indirect } = await runScan(rig, counts);
    const ref = cpuExclusiveScan(counts);
    expect(Array.from(firstEmit)).toEqual(ref.firstEmit);
    expect(indirect[0]).toBe(ref.total);
  });

  it("scans across multiple tiles (3000 records)", async () => {
    const device = await requestRealDevice();
    const rig = await buildRig(device, 3072);
    const counts: number[] = [];
    for (let i = 0; i < 3000; i++) counts.push(((i * 13) ^ 0x5a) % 37 + 1);
    const { firstEmit, indirect } = await runScan(rig, counts);
    const ref = cpuExclusiveScan(counts);
    expect(Array.from(firstEmit)).toEqual(ref.firstEmit);
    expect(indirect[0]).toBe(ref.total);
  });

  it("scans 10000 records across many tiles", async () => {
    const device = await requestRealDevice();
    const rig = await buildRig(device, 10240);
    const counts: number[] = [];
    for (let i = 0; i < 10000; i++) counts.push(((i * 31) ^ 0x9e3) % 53 + 1);
    const { firstEmit, indirect } = await runScan(rig, counts);
    const ref = cpuExclusiveScan(counts);
    expect(Array.from(firstEmit)).toEqual(ref.firstEmit);
    expect(indirect[0]).toBe(ref.total);
  });
});
