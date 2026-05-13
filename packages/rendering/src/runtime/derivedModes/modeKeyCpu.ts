// Per-RO modeKey production (CPU side).
//
// Task 1 of the derived-modes work: turn a `PipelineState` (the
// aval-shaped struct from core/pipelineState.ts) plus the surrounding
// `FramebufferSignature` into a canonical `PipelineStateDescriptor`
// and its bitfield-encoded modeKey. Subscribes to every input aval
// so that mutating a `cval` for cullMode / blendFactor / etc. fires
// a dirty callback, the bucket flushes the changed RO's modeKey to
// GPU on the next frame, and the partition kernel re-buckets the
// record into its new slot. No rule-IR machinery yet — Task 2 layers
// `derivedMode(...)` on top of this.

import type {
  DepthState,
  PipelineState,
  RasterizerState,
  StencilState,
  BlendState as PsBlendState,
  BlendComponentState as PsBlendComponentState,
} from "../../core/pipelineState.js";
import type { FramebufferSignature } from "../../core/framebufferSignature.js";
import type { aval, IDisposable } from "@aardworx/wombat.adaptive";
import { addMarkingCallback } from "@aardworx/wombat.adaptive";

import {
  encodeModeKey,
  DEFAULT_DESCRIPTOR,
  DEFAULT_ATTACHMENT_BLEND,
  type PipelineStateDescriptor,
  type AttachmentBlend,
  type BlendComponent,
  type DepthSlice,
} from "../pipelineCache/index.js";

/**
 * One-shot snapshot of the current values of every leaf aval in a
 * `PipelineState`, mapped onto a `PipelineStateDescriptor` whose
 * `attachments` array follows the framebuffer's `colorNames` order
 * (missing entries default to non-blended / 0xF writeMask, matching
 * `BLEND_DEFAULT` semantics in heapScene).
 */
export function snapshotDescriptor(
  ps: PipelineState | undefined,
  signature: FramebufferSignature,
): PipelineStateDescriptor {
  if (ps === undefined) {
    return buildAttachmentsFor(DEFAULT_DESCRIPTOR, signature);
  }
  const r = ps.rasterizer;
  const depth = ps.depth !== undefined ? snapshotDepth(ps.depth, signature) : undefined;
  const atc = ps.alphaToCoverage !== undefined ? ps.alphaToCoverage.force(/* allow-force */) : false;
  const blends = ps.blends !== undefined ? ps.blends.force(/* allow-force */) : undefined;
  const attachments: AttachmentBlend[] = signature.colorNames.map((name) => {
    const b = blends?.tryFind(name);
    return b !== undefined ? snapshotAttachment(b) : DEFAULT_ATTACHMENT_BLEND;
  });
  const out: PipelineStateDescriptor = {
    topology:         r.topology.force(/* allow-force */),
    stripIndexFormat: stripFormatFor(r.topology.force(/* allow-force */)),
    frontFace:        r.frontFace.force(/* allow-force */),
    cullMode:         r.cullMode.force(/* allow-force */),
    ...(depth !== undefined ? { depth } : {}),
    attachments,
    alphaToCoverage: atc,
  };
  return out;
}

function buildAttachmentsFor(
  base: PipelineStateDescriptor,
  signature: FramebufferSignature,
): PipelineStateDescriptor {
  // When `ps` is undefined, every named attachment uses defaults.
  return {
    ...base,
    attachments: signature.colorNames.map(() => DEFAULT_ATTACHMENT_BLEND),
  };
}

function snapshotDepth(d: DepthState, signature: FramebufferSignature): DepthSlice | undefined {
  if (signature.depthStencil === undefined) return undefined;
  return {
    write:   d.write.force(/* allow-force */),
    compare: d.compare.force(/* allow-force */),
    clamp:   d.clamp !== undefined ? d.clamp.force(/* allow-force */) : false,
  };
}

function snapshotAttachment(b: PsBlendState): AttachmentBlend {
  const color = snapshotBlendComponent(b.color);
  const alpha = snapshotBlendComponent(b.alpha);
  const writeMask = b.writeMask.force(/* allow-force */) & 0xF;
  // "Blend enabled" derives from whether the components differ from
  // the no-op (src=one, dst=zero, op=add) shape. This mirrors WebGPU's
  // contract: a blend descriptor MUST be present for blending to occur,
  // and the no-op shape is the canonical disabled state.
  const enabled =
    !(color.srcFactor === "one" && color.dstFactor === "zero" && color.operation === "add" &&
      alpha.srcFactor === "one" && alpha.dstFactor === "zero" && alpha.operation === "add");
  return { enabled, color, alpha, writeMask };
}

function snapshotBlendComponent(c: PsBlendComponentState): BlendComponent {
  return {
    srcFactor: c.srcFactor.force(/* allow-force */),
    dstFactor: c.dstFactor.force(/* allow-force */),
    operation: c.operation.force(/* allow-force */),
  };
}

function stripFormatFor(topology: GPUPrimitiveTopology): GPUIndexFormat | undefined {
  return topology === "line-strip" || topology === "triangle-strip" ? "uint32" : undefined;
}

// ─── Reactive tracker ──────────────────────────────────────────────────

/**
 * Subscribes to every leaf aval in a `PipelineState` and invokes
 * `onDirty` when any of them marks. Use one tracker per RO; call
 * `recompute()` after a dirty signal to refresh the descriptor +
 * modeKey, and `dispose()` when the RO is removed.
 *
 * Reactive cvals are the typical case; constant avals also "subscribe"
 * — `addMarkingCallback` is a no-op for an aval that never marks, so
 * no special-casing required.
 */
export class ModeKeyTracker implements IDisposable {
  readonly ps: PipelineState | undefined;
  readonly signature: FramebufferSignature;
  private readonly onDirty: () => void;
  private readonly subs: IDisposable[] = [];
  private cachedDescriptor: PipelineStateDescriptor;
  private cachedModeKey: bigint;

  constructor(
    ps: PipelineState | undefined,
    signature: FramebufferSignature,
    onDirty: () => void,
  ) {
    this.ps        = ps;
    this.signature = signature;
    this.onDirty   = onDirty;
    this.cachedDescriptor = snapshotDescriptor(ps, signature);
    this.cachedModeKey    = encodeModeKey(this.cachedDescriptor);
    this.subscribeAll();
  }

  get descriptor(): PipelineStateDescriptor { return this.cachedDescriptor; }
  get modeKey(): bigint { return this.cachedModeKey; }

  /**
   * Snapshot current aval values and rebuild descriptor + modeKey.
   * Returns true iff the modeKey changed (so the bucket only needs to
   * upload + repartition when something actually moved).
   */
  recompute(): boolean {
    const next = snapshotDescriptor(this.ps, this.signature);
    const nextKey = encodeModeKey(next);
    if (nextKey === this.cachedModeKey) return false;
    this.cachedDescriptor = next;
    this.cachedModeKey    = nextKey;
    return true;
  }

  dispose(): void {
    for (const s of this.subs) s.dispose();
    this.subs.length = 0;
  }

  private subscribeAll(): void {
    if (this.ps === undefined) return;
    const ps = this.ps;
    this.sub(ps.rasterizer.topology);
    this.sub(ps.rasterizer.cullMode);
    this.sub(ps.rasterizer.frontFace);
    if (ps.rasterizer.depthBias !== undefined) this.sub(ps.rasterizer.depthBias);
    if (ps.depth !== undefined) {
      this.sub(ps.depth.write);
      this.sub(ps.depth.compare);
      if (ps.depth.clamp !== undefined) this.sub(ps.depth.clamp);
    }
    if (ps.stencil !== undefined) {
      // Stencil enabled/reference/masks contribute to neither modeKey
      // nor cache key (v1 carves stencil out — only the "enabled"
      // flag matters and we surface it via snapshotDescriptor). We
      // still subscribe so callers can repackage if a future change
      // flips the enable bit.
      this.subStencilFaces(ps.stencil);
    }
    if (ps.blends !== undefined) {
      this.sub(ps.blends);
      const map = ps.blends.force(/* allow-force */);
      for (const [, bs] of map) this.subBlendState(bs);
    }
    if (ps.alphaToCoverage !== undefined) this.sub(ps.alphaToCoverage);
    // blendConstant is dynamic state; not in the modeKey, no subscription needed.
  }

  private sub(a: aval<unknown>): void {
    this.subs.push(addMarkingCallback(a, this.onDirty));
  }

  private subBlendState(b: PsBlendState): void {
    this.sub(b.color.srcFactor); this.sub(b.color.dstFactor); this.sub(b.color.operation);
    this.sub(b.alpha.srcFactor); this.sub(b.alpha.dstFactor); this.sub(b.alpha.operation);
    this.sub(b.writeMask);
  }

  private subStencilFaces(_s: StencilState): void {
    // v1: stencil isn't in the modeKey; deliberately not subscribing
    // to face avals here. Wire up in v2 when stencil enters the key.
  }
}
