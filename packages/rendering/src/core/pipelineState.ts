// PipelineState — the fixed-function bits the user picks per
// RenderObject. Vertex layout, blending, depth/stencil, raster.
// Combined with a CompiledEffect + FramebufferSignature this is
// enough to bake a GPURenderPipeline.
//
// All fields are `aval<T>` so wombat.dom's render-state scopes can
// flow through without forcing. The shader (`Effect`) stays a plain
// non-aval value; only uniforms and pipeline state are reactive.

import { AVal, HashMap, type aval } from "@aardworx/wombat.adaptive";

export type CullMode = "none" | "front" | "back";
export type FrontFace = "ccw" | "cw";
export type Topology = GPUPrimitiveTopology;

export interface DepthBiasState {
  readonly constant: number;
  readonly slopeScale: number;
  readonly clamp: number;
}

export interface RasterizerState {
  readonly topology:  aval<Topology>;
  readonly cullMode:  aval<CullMode>;
  readonly frontFace: aval<FrontFace>;
  /** `undefined` value disables polygon offset. */
  readonly depthBias?: aval<DepthBiasState | undefined>;
}

export interface DepthState {
  readonly write:   aval<boolean>;
  readonly compare: aval<GPUCompareFunction>;
  /** unclippedDepth — requires the WebGPU adapter feature. */
  readonly clamp?:  aval<boolean>;
}

export interface StencilFaceState {
  readonly compare:     aval<GPUCompareFunction>;
  readonly failOp:      aval<GPUStencilOperation>;
  readonly depthFailOp: aval<GPUStencilOperation>;
  readonly passOp:      aval<GPUStencilOperation>;
}

export interface StencilState {
  readonly enabled:   aval<boolean>;
  /** Per-frame, NOT pipeline rebuild. */
  readonly reference: aval<number>;
  readonly readMask:  aval<number>;
  readonly writeMask: aval<number>;
  readonly front:     StencilFaceState;
  readonly back:      StencilFaceState;
}

export interface BlendComponentState {
  readonly operation: aval<GPUBlendOperation>;
  readonly srcFactor: aval<GPUBlendFactor>;
  readonly dstFactor: aval<GPUBlendFactor>;
}

export interface BlendState {
  readonly color:     BlendComponentState;
  readonly alpha:     BlendComponentState;
  /** RGBA channel write mask (bitmask: R=1, G=2, B=4, A=8). */
  readonly writeMask: aval<number>;
}

export interface PipelineState {
  readonly rasterizer:       RasterizerState;
  readonly depth?:           DepthState;
  readonly stencil?:         StencilState;
  /**
   * Per-color-attachment blend state, keyed by the attachment name
   * from the `FramebufferSignature`. Missing entries default to
   * "no blending, write all channels".
   */
  readonly blends?:          aval<HashMap<string, BlendState>>;
  /** Multi-sample alpha-to-coverage. Default `false`. */
  readonly alphaToCoverage?: aval<boolean>;
  /** Per-frame, not pipeline-rebuild. */
  readonly blendConstant?:   aval<{ r: number; g: number; b: number; a: number }>;
}

// ---------------------------------------------------------------------------
// Plain (non-aval) mirror types — handy for the `PipelineState.constant`
// helper. Callers that already hold avals construct `PipelineState`
// directly; callers that have plain values use the helper to wrap.
// ---------------------------------------------------------------------------

export interface PlainRasterizerState {
  readonly topology: Topology;
  readonly cullMode: CullMode;
  readonly frontFace: FrontFace;
  readonly depthBias?: DepthBiasState;
}

export interface PlainDepthState {
  readonly write: boolean;
  readonly compare: GPUCompareFunction;
  readonly clamp?: boolean;
}

export interface PlainStencilFaceState {
  readonly compare: GPUCompareFunction;
  readonly failOp: GPUStencilOperation;
  readonly depthFailOp: GPUStencilOperation;
  readonly passOp: GPUStencilOperation;
}

export interface PlainStencilState {
  readonly enabled: boolean;
  readonly reference: number;
  readonly readMask: number;
  readonly writeMask: number;
  readonly front: PlainStencilFaceState;
  readonly back: PlainStencilFaceState;
}

export interface PlainBlendComponentState {
  readonly operation: GPUBlendOperation;
  readonly srcFactor: GPUBlendFactor;
  readonly dstFactor: GPUBlendFactor;
}

export interface PlainBlendState {
  readonly color: PlainBlendComponentState;
  readonly alpha: PlainBlendComponentState;
  readonly writeMask: number;
}

export interface PlainPipelineState {
  readonly rasterizer: PlainRasterizerState;
  readonly depth?: PlainDepthState;
  readonly stencil?: PlainStencilState;
  readonly blends?: HashMap<string, PlainBlendState>;
  readonly alphaToCoverage?: boolean;
  readonly blendConstant?: { r: number; g: number; b: number; a: number };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function blendStateFromPlain(b: PlainBlendState): BlendState {
  return {
    color: {
      operation: AVal.constant(b.color.operation),
      srcFactor: AVal.constant(b.color.srcFactor),
      dstFactor: AVal.constant(b.color.dstFactor),
    },
    alpha: {
      operation: AVal.constant(b.alpha.operation),
      srcFactor: AVal.constant(b.alpha.srcFactor),
      dstFactor: AVal.constant(b.alpha.dstFactor),
    },
    writeMask: AVal.constant(b.writeMask),
  };
}

function stencilFaceFromPlain(f: PlainStencilFaceState): StencilFaceState {
  return {
    compare: AVal.constant(f.compare),
    failOp: AVal.constant(f.failOp),
    depthFailOp: AVal.constant(f.depthFailOp),
    passOp: AVal.constant(f.passOp),
  };
}

/**
 * Wrap a plain `PlainPipelineState` into the aval-shaped
 * `PipelineState` by lifting every leaf with `AVal.constant`. Useful
 * for tests and for callers that don't carry avals yet — wombat.dom's
 * `derivePipelineState` constructs the aval-shape directly.
 */
export const PipelineState = {
  constant(plain: PlainPipelineState): PipelineState {
    const rast: RasterizerState = {
      topology: AVal.constant(plain.rasterizer.topology),
      cullMode: AVal.constant(plain.rasterizer.cullMode),
      frontFace: AVal.constant(plain.rasterizer.frontFace),
      ...(plain.rasterizer.depthBias !== undefined
        ? { depthBias: AVal.constant<DepthBiasState | undefined>(plain.rasterizer.depthBias) }
        : {}),
    };
    let blends: aval<HashMap<string, BlendState>> | undefined;
    if (plain.blends !== undefined) {
      let m = HashMap.empty<string, BlendState>();
      for (const [k, v] of plain.blends) m = m.add(k, blendStateFromPlain(v));
      blends = AVal.constant(m);
    }
    return {
      rasterizer: rast,
      ...(plain.depth !== undefined
        ? {
            depth: {
              write: AVal.constant(plain.depth.write),
              compare: AVal.constant(plain.depth.compare),
              ...(plain.depth.clamp !== undefined ? { clamp: AVal.constant(plain.depth.clamp) } : {}),
            },
          }
        : {}),
      ...(plain.stencil !== undefined
        ? {
            stencil: {
              enabled: AVal.constant(plain.stencil.enabled),
              reference: AVal.constant(plain.stencil.reference),
              readMask: AVal.constant(plain.stencil.readMask),
              writeMask: AVal.constant(plain.stencil.writeMask),
              front: stencilFaceFromPlain(plain.stencil.front),
              back: stencilFaceFromPlain(plain.stencil.back),
            },
          }
        : {}),
      ...(blends !== undefined ? { blends } : {}),
      ...(plain.alphaToCoverage !== undefined ? { alphaToCoverage: AVal.constant(plain.alphaToCoverage) } : {}),
      ...(plain.blendConstant !== undefined ? { blendConstant: AVal.constant(plain.blendConstant) } : {}),
    };
  },
};
