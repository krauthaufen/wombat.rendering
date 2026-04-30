// PipelineState — the fixed-function bits the user picks per
// RenderObject. Vertex layout, blending, depth/stencil, raster.
// Combined with a CompiledEffect + FramebufferSignature this is
// enough to bake a GPURenderPipeline.

import type { HashMap } from "@aardworx/wombat.adaptive";

export type CullMode = "none" | "front" | "back";
export type FrontFace = "ccw" | "cw";
export type Topology = GPUPrimitiveTopology;

export interface RasterizerState {
  readonly topology: Topology;
  readonly cullMode: CullMode;
  readonly frontFace: FrontFace;
  /** "none" disables polygon offset. */
  readonly depthBias?: { readonly constant: number; readonly slopeScale: number; readonly clamp: number };
}

export interface DepthState {
  readonly write: boolean;
  readonly compare: GPUCompareFunction;
}

export interface StencilFaceState {
  readonly compare: GPUCompareFunction;
  readonly failOp: GPUStencilOperation;
  readonly depthFailOp: GPUStencilOperation;
  readonly passOp: GPUStencilOperation;
}

export interface StencilState {
  readonly readMask: number;
  readonly writeMask: number;
  readonly front: StencilFaceState;
  readonly back: StencilFaceState;
}

export interface BlendComponentState {
  readonly operation: GPUBlendOperation;
  readonly srcFactor: GPUBlendFactor;
  readonly dstFactor: GPUBlendFactor;
}

export interface BlendState {
  readonly color: BlendComponentState;
  readonly alpha: BlendComponentState;
  /** RGBA channel write mask (bitmask: R=1, G=2, B=4, A=8). */
  readonly writeMask: number;
}

export interface PipelineState {
  readonly rasterizer: RasterizerState;
  readonly depth?: DepthState;
  readonly stencil?: StencilState;
  /**
   * Per-color-attachment blend state, keyed by the attachment name
   * from the `FramebufferSignature`. Missing entries default to
   * "no blending, write all channels".
   */
  readonly blends?: HashMap<string, BlendState>;
  /** Multi-sample alpha-to-coverage. Default `false`. */
  readonly alphaToCoverage?: boolean;
}
