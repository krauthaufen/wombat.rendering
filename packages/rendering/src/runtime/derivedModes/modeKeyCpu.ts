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
import { addMarkingCallback, AdaptiveToken } from "@aardworx/wombat.adaptive";

import {
  encodeModeKey,
  DEFAULT_DESCRIPTOR,
  DEFAULT_ATTACHMENT_BLEND,
  type PipelineStateDescriptor,
  type AttachmentBlend,
  type BlendComponent,
  type DepthSlice,
} from "../pipelineCache/index.js";
import type { DerivedModeRule } from "./rule.js";

/**
 * Per-axis derived-mode rules for one RO. Same shape as
 * `HeapDrawSpec.modeRules`.
 */
export interface RoModeRules {
  readonly cull?:            DerivedModeRule<"cull">;
  readonly frontFace?:       DerivedModeRule<"frontFace">;
  readonly topology?:        DerivedModeRule<"topology">;
  readonly depthCompare?:    DerivedModeRule<"depthCompare">;
  readonly depthWrite?:      DerivedModeRule<"depthWrite">;
  readonly alphaToCoverage?: DerivedModeRule<"alphaToCoverage">;
}


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
  options: {
    readonly modeRules?: RoModeRules;
    readonly uniformAvals?: ReadonlyMap<string, aval<unknown>>;
    readonly token?: AdaptiveToken;
    readonly recordDep?: (av: aval<unknown>) => void;
  } = {},
): PipelineStateDescriptor {
  if (ps === undefined) {
    return applyRules(buildAttachmentsFor(DEFAULT_DESCRIPTOR, signature), options);
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
  return applyRules(out, options);
}

function applyRules(
  d: PipelineStateDescriptor,
  options: {
    readonly modeRules?: RoModeRules;
    readonly uniformAvals?: ReadonlyMap<string, aval<unknown>>;
    readonly token?: AdaptiveToken;
    readonly recordDep?: (av: aval<unknown>) => void;
  },
): PipelineStateDescriptor {
  // Phase 5c.3 — rules are IR-traced and run on the GPU. We never
  // CPU-evaluate them here; the descriptor returned reflects the
  // raw PipelineState aval values (= the "declared" value for each
  // axis). The heap runtime creates one bucket slot per
  // `rule.domain` entry and the partition kernel routes per record.
  void options;
  return d;
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
  private readonly modeRules: RoModeRules | undefined;
  private readonly uniformAvals: ReadonlyMap<string, aval<unknown>> | undefined;
  /** Avals discovered while evaluating rules; merged into the leaf
   *  set returned by `forEachLeaf` so the heap scene subscribes once. */
  private readonly ruleDeps = new Set<aval<unknown>>();

  constructor(
    ps: PipelineState | undefined,
    signature: FramebufferSignature,
    onDirty: () => void,
    /**
     * Skip the per-instance addMarkingCallback subscriptions. The caller
     * is responsible for invoking `onDirty` (or recomputing directly)
     * when an input aval marks. Use this when the heap scene shares one
     * subscription across N tracker instances (the 20k-ROs-share-one-
     * cullCval case) — saves N subscriptions and N callback dispatches
     * per mark.
     */
    options: {
      skipSubscribe?: boolean;
      modeRules?: RoModeRules;
      uniformAvals?: ReadonlyMap<string, aval<unknown>>;
    } = {},
  ) {
    this.ps        = ps;
    this.signature = signature;
    this.onDirty   = onDirty;
    if (options.modeRules    !== undefined) this.modeRules    = options.modeRules;
    if (options.uniformAvals !== undefined) this.uniformAvals = options.uniformAvals;
    this.cachedDescriptor = this.snapshot();
    this.cachedModeKey    = encodeModeKey(this.cachedDescriptor);
    if (options.skipSubscribe !== true) this.subscribeAll();
  }

  private snapshot(): PipelineStateDescriptor {
    return snapshotDescriptor(this.ps, this.signature, {
      ...(this.modeRules    !== undefined ? { modeRules:    this.modeRules    } : {}),
      ...(this.uniformAvals !== undefined ? { uniformAvals: this.uniformAvals } : {}),
      recordDep: (av) => { this.ruleDeps.add(av); },
    });
  }

  /**
   * Walk every leaf aval in this tracker's PipelineState and invoke
   * `visit(aval)`. Used by the heap scene to register ONE
   * addMarkingCallback per unique aval across many trackers.
   */
  forEachLeaf(visit: (a: aval<unknown>) => void): void {
    const ps = this.ps;
    if (ps !== undefined) {
      visit(ps.rasterizer.topology);
      visit(ps.rasterizer.cullMode);
      visit(ps.rasterizer.frontFace);
      if (ps.rasterizer.depthBias !== undefined) visit(ps.rasterizer.depthBias);
      if (ps.depth !== undefined) {
        visit(ps.depth.write);
        visit(ps.depth.compare);
        if (ps.depth.clamp !== undefined) visit(ps.depth.clamp);
      }
      if (ps.blends !== undefined) {
        visit(ps.blends);
        const map = ps.blends.force(/* allow-force */);
        for (const [, bs] of map) {
          visit(bs.color.srcFactor); visit(bs.color.dstFactor); visit(bs.color.operation);
          visit(bs.alpha.srcFactor); visit(bs.alpha.dstFactor); visit(bs.alpha.operation);
          visit(bs.writeMask);
        }
      }
      if (ps.alphaToCoverage !== undefined) visit(ps.alphaToCoverage);
    }
    // Phase 5c.3: rule-input deps will be walked from the rule's
    // shader-IR body (analyseInputUniforms — pending M6/M7 wire-up).
    // For now we only visit `declared` if it's an aval — leaves
    // referenced from the rule body don't trigger partition re-
    // dispatch until the heap side reads the IR. Hidden behind the
    // initGpuRouting stub that throws on attached rules.
    if (this.modeRules !== undefined) {
      for (const rule of Object.values(this.modeRules) as DerivedModeRule[]) {
        if (rule === undefined) continue;
        const d = rule.declared;
        if (typeof d === "object" && d !== null && "getValue" in (d as object)) {
          visit(d as aval<unknown>);
        }
      }
    }
    for (const av of this.ruleDeps) visit(av);
  }

  get descriptor(): PipelineStateDescriptor { return this.cachedDescriptor; }
  get modeKey(): bigint { return this.cachedModeKey; }

  /**
   * Snapshot current aval values and rebuild descriptor + modeKey.
   * Returns true iff the modeKey changed (so the bucket only needs to
   * upload + repartition when something actually moved).
   *
   * Also rediscovers rule-input dependencies on every call — a
   * conditional rule body may read different uniforms in different
   * branches. New deps surface in `forEachLeaf` so the heap scene's
   * subscription-dedupe layer picks them up.
   */
  recompute(): boolean {
    this.ruleDeps.clear();
    const next = this.snapshot();
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
