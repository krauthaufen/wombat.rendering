// Canonical pipeline-state descriptor — the data needed to bake a
// `GPURenderPipeline` minus the bind-group / shader-module / vertex-layout
// pieces (those come from the bucket's effect + family). This struct is
// what the bitfield encoder/decoder round-trips and what the pipeline
// cache keys on.
//
// All fields are plain (non-aval) values — descriptors are produced by
// snapshotting the user-authored aval PipelineState at scene-build or
// at modeKey-recompute time.
//
// Fields excluded by design (see derived-modes.md "What can and cannot
// be derived"):
//   - depthBias triplet (continuous baked field; stays per-effect or
//     per-SG-scope static, set when the descriptor is constructed)
//   - blendConstant     (dynamic state; set via setBlendConstant)
//   - stencilReference  (dynamic state; set via setStencilReference)
//   - render-pass-constant fields (color/depth formats, sample count;
//     known per bucket from the framebuffer signature)

export type CullMode  = "none" | "front" | "back";
export type FrontFace = "ccw" | "cw";
export type Topology  = GPUPrimitiveTopology;

export interface DepthSlice {
  readonly write:   boolean;
  readonly compare: GPUCompareFunction;
  /** unclippedDepth — requires the WebGPU adapter feature. */
  readonly clamp:   boolean;
}

export interface BlendComponent {
  readonly srcFactor: GPUBlendFactor;
  readonly dstFactor: GPUBlendFactor;
  readonly operation: GPUBlendOperation;
}

export interface AttachmentBlend {
  /** Whether blending is enabled for this attachment. */
  readonly enabled:   boolean;
  /** Only valid if `enabled` is true. */
  readonly color:     BlendComponent;
  readonly alpha:     BlendComponent;
  /** RGBA channel write mask (bitmask: R=1, G=2, B=4, A=8). */
  readonly writeMask: number;
}

export interface StencilFace {
  readonly compare:     GPUCompareFunction;
  readonly failOp:      GPUStencilOperation;
  readonly depthFailOp: GPUStencilOperation;
  readonly passOp:      GPUStencilOperation;
}

export interface StencilSlice {
  readonly readMask:  number;
  readonly writeMask: number;
  readonly front:     StencilFace;
  readonly back:      StencilFace;
}

export interface PipelineStateDescriptor {
  readonly topology:         Topology;
  readonly stripIndexFormat: GPUIndexFormat | undefined;
  readonly frontFace:        FrontFace;
  readonly cullMode:         CullMode;
  /** Present only if the depth attachment is in use. */
  readonly depth?:           DepthSlice;
  /** Present only if stencil testing is enabled. */
  readonly stencil?:         StencilSlice;
  /** One entry per color attachment. v1: at most one; v2 expands. */
  readonly attachments:      readonly AttachmentBlend[];
  readonly alphaToCoverage:  boolean;
}

export const DEFAULT_ATTACHMENT_BLEND: AttachmentBlend = {
  enabled:   false,
  color:     { srcFactor: "one", dstFactor: "zero", operation: "add" },
  alpha:     { srcFactor: "one", dstFactor: "zero", operation: "add" },
  writeMask: 0xF,
};

export const DEFAULT_DESCRIPTOR: PipelineStateDescriptor = {
  topology:         "triangle-list",
  stripIndexFormat: undefined,
  frontFace:        "ccw",
  cullMode:         "back",
  depth:            { write: true, compare: "less", clamp: false },
  attachments:      [DEFAULT_ATTACHMENT_BLEND],
  alphaToCoverage:  false,
};

/**
 * Structural equality on two descriptors. Used as the collision-safety
 * fallback if a bitfield key ever needs disambiguation (not expected
 * with the current layout; the bitfield is collision-free by
 * construction).
 */
export function descriptorEquals(
  a: PipelineStateDescriptor,
  b: PipelineStateDescriptor,
): boolean {
  if (a.topology !== b.topology) return false;
  if (a.stripIndexFormat !== b.stripIndexFormat) return false;
  if (a.frontFace !== b.frontFace) return false;
  if (a.cullMode !== b.cullMode) return false;
  if ((a.depth === undefined) !== (b.depth === undefined)) return false;
  if (a.depth !== undefined && b.depth !== undefined) {
    if (a.depth.write   !== b.depth.write)   return false;
    if (a.depth.compare !== b.depth.compare) return false;
    if (a.depth.clamp   !== b.depth.clamp)   return false;
  }
  if ((a.stencil === undefined) !== (b.stencil === undefined)) return false;
  if (a.stencil !== undefined && b.stencil !== undefined) {
    if (!stencilEquals(a.stencil, b.stencil)) return false;
  }
  if (a.alphaToCoverage !== b.alphaToCoverage) return false;
  if (a.attachments.length !== b.attachments.length) return false;
  for (let i = 0; i < a.attachments.length; i++) {
    if (!attachmentEquals(a.attachments[i]!, b.attachments[i]!)) return false;
  }
  return true;
}

function attachmentEquals(a: AttachmentBlend, b: AttachmentBlend): boolean {
  if (a.enabled !== b.enabled) return false;
  if (a.writeMask !== b.writeMask) return false;
  if (!a.enabled) return true; // color/alpha fields irrelevant
  return (
    a.color.srcFactor === b.color.srcFactor &&
    a.color.dstFactor === b.color.dstFactor &&
    a.color.operation === b.color.operation &&
    a.alpha.srcFactor === b.alpha.srcFactor &&
    a.alpha.dstFactor === b.alpha.dstFactor &&
    a.alpha.operation === b.alpha.operation
  );
}

function stencilFaceEquals(a: StencilFace, b: StencilFace): boolean {
  return (
    a.compare === b.compare &&
    a.failOp === b.failOp &&
    a.depthFailOp === b.depthFailOp &&
    a.passOp === b.passOp
  );
}

function stencilEquals(a: StencilSlice, b: StencilSlice): boolean {
  if (a.readMask !== b.readMask || a.writeMask !== b.writeMask) return false;
  return stencilFaceEquals(a.front, b.front) && stencilFaceEquals(a.back, b.back);
}
