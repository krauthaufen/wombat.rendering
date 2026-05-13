// GPU-evaluated derived-mode rules — minimum viable kernel.
//
// CPU eval (Task 2 Phase 1-3, 0.9.17) runs rule closures in JS. For
// rules whose inputs live GPU-side (per-RO trafos, etc.) JS evaluation
// fans out CPU work that's wasted: the trafo bytes are right there in
// the arena. This module runs the rule on the GPU instead, in one
// dispatch over all dirty ROs per frame.
//
// v1 scope is deliberately narrow: ONE hard-coded WGSL kernel for
// "flip cullMode by sign of det(upper3x3 of an arena mat4)". Picked
// because it's the motivating case from docs/derived-modes.md and
// proves the GPU-eval path end-to-end. Generalizes via a traced IR
// builder in v2 (see derived-mode-rules-plan.md Phase 5).
//
// Architecture
// ============
//
//   - One **module-wide** `GpuDerivedModesScene` per heap scene. Holds:
//       * `roOutputBuf`  — `array<u32>` sized to scene.numROs.
//                          Each entry = the rule's modeKey contribution.
//       * `roInputBuf`   — `array<u32>` (one ref-into-arena per RO)
//                          telling the kernel where to read the input.
//       * `stagingBuf`   — MAP_READ + COPY_DST mirror of roOutputBuf,
//                          for the per-frame readback.
//   - Dispatch each frame in `update()`, ONLY when something dirty.
//   - `mapAsync(staging)` resolves on a microtask; the CPU diffs the
//     buffer against `roLastKey[]`. Any change → mark drawId dirty;
//     the existing rebucket flow does the rest.
//
// The dirty gate keeps steady-state cost ≈ 0 — when no input aval
// marks AND no RO add/remove, we skip the dispatch + readback.

export const GPU_FLIP_CULL_BY_DET_WGSL = /* wgsl */ `
// Per-RO record. Index i corresponds to drawId i in the scene's
// flat RO table.
struct GpuRuleParams {
  numROs:     u32,
  declared:   u32,   // 0 = none, 1 = front, 2 = back  (must match CULL_MODE[] order in bitfield.ts)
  _pad0:      u32,
  _pad1:      u32,
};

@group(0) @binding(0) var<storage, read>       arena:        array<u32>;
// One per-RO entry: byte offset within the arena of the ROs ModelTrafo
// allocation header. Data is at offset + 16 (ALLOC_HEADER_PAD_TO).
// 0xFFFFFFFFu means "this RO does NOT have a flip-cull-by-det rule".
@group(0) @binding(1) var<storage, read>       roInputRefs:  array<u32>;
@group(0) @binding(2) var<storage, read_write> roOutputs:    array<u32>;
@group(0) @binding(3) var<uniform>             uIn:          GpuRuleParams;

// Read a row-major mat4 from the arena at byte offset \`refBytes + 16\`
// (header is 16 bytes wide due to ALLOC_HEADER_PAD_TO). Returns the
// 3×3 upper-left in row-major order: [m00,m01,m02, m10,m11,m12, m20,m21,m22].
fn loadUpper3x3(refBytes: u32) -> array<f32, 9> {
  let baseU32 = (refBytes + 16u) >> 2u;
  // The arena is a Float32 storage of mat4 in row-major, 16 floats.
  // Read 9 of them.
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

// Mirror of bitfield.ts CULL_MODE[] order: 0=none, 1=front, 2=back.
fn flipCull(c: u32) -> u32 {
  if (c == 1u) { return 2u; } // front -> back
  if (c == 2u) { return 1u; } // back  -> front
  return c;                   // none
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= uIn.numROs) { return; }
  let refByte = roInputRefs[i];
  if (refByte == 0xFFFFFFFFu) {
    // No rule on this RO; leave existing output untouched. Allows a
    // single kernel dispatch to skip non-participating ROs.
    return;
  }
  let m = loadUpper3x3(refByte);
  let d = det3x3(m);
  let declared = uIn.declared;
  let outVal = select(declared, flipCull(declared), d < 0.0);
  roOutputs[i] = outVal;
}
`;

/** Map between bitfield.ts's CULL_MODE order and the kernel's u32 enum. */
export const CULL_TO_U32 = { "none": 0, "front": 1, "back": 2 } as const;
export const U32_TO_CULL = ["none", "front", "back"] as const;
