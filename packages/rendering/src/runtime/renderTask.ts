// RenderTask — compile an `alist<Command>` into a runnable
// IRenderTask.
//
// Per `Render` command we lazily build a `ScenePass` that holds a
// stateful tree-walker subscribed to the dynamic subtrees
// (`Adaptive` / `OrderedFromList` / `UnorderedFromSet`). Each frame:
//   - The walker pulls deltas from its readers (O(deltas) work).
//   - It produces the ordered list of `PreparedRenderObject`s.
//   - We open the render pass and call `record(pass, token)` per leaf.
//
// The previous per-frame `collectLeaves` recursion through the tree
// is gone. Static subtrees prep their leaves once at construction;
// dynamic subtrees splice on delta — see scenePass.ts.

import {
  RenderContext,
  type ClearValues,
  type Command,
  type CompiledEffect,
  type Effect,
  type FramebufferSignature,
  type IFramebuffer,
  type IRenderTask,
  type RenderTree,
} from "../core/index.js";
import { beginPassDescriptor, clear } from "../commands/index.js";
import { AdaptiveToken, type alist, type aval } from "@aardworx/wombat.adaptive";
import { copy } from "./copy.js";
import { compileHybridScene, type HybridScene } from "./hybridScene.js";

export interface RuntimeContext {
  readonly device: GPUDevice;
  /**
   * Compile an `Effect` against the target framebuffer signature.
   * The signature determines the canonical fragment-output layout
   * (`signature.colorNames[i]` ↔ `@location(i)`); the shader's
   * `linkFragmentOutputs` pass uses it to re-pin and DCE outputs.
   */
  readonly compileEffect: (effect: Effect, signature: FramebufferSignature) => CompiledEffect;
  /**
   * Global heap on/off toggle propagated to every `HybridScene`
   * compiled against this context. Flipping it causes every RO to
   * migrate between the heap subset and the legacy subset.
   * Useful for A/B perf comparisons.
   */
  readonly heapEnabled?: aval<boolean>;
  /**
   * §6 family-merge opt-in — propagated to every HybridScene
   * compiled against this context. Default: merge off.
   */
  readonly enableFamilyMerge?: boolean;
  /**
   * §7 derived-uniforms opt-in — propagated to every HybridScene
   * compiled against this context.
   */
  readonly enableDerivedUniforms?: boolean;
}

class RenderTask implements IRenderTask {
  /**
   * Cache of `HybridScene`s keyed on the `Render` command itself.
   * Each command compiles to a hybrid scene composing the heap-
   * bucket fast path with the legacy per-RO path; reuse across
   * frames preserves both backends' incremental state.
   */
  private readonly _scenes = new Map<unknown, HybridScene>();
  private _disposed = false;

  constructor(
    private readonly ctx: RuntimeContext,
    readonly signature: FramebufferSignature,
    private readonly commands: alist<Command>,
  ) {}

  run(framebuffer: IFramebuffer, token: AdaptiveToken): void {
    if (this._disposed) throw new Error("RenderTask: run after dispose");
    const enc = this.ctx.device.createCommandEncoder();
    this.encode(enc, framebuffer, token);
    this.ctx.device.queue.submit([enc.finish()]);
  }

  encode(enc: GPUCommandEncoder, framebuffer: IFramebuffer, token: AdaptiveToken): void {
    if (this._disposed) throw new Error("RenderTask: encode after dispose");
    RenderContext.withEncoder(enc, () => {
      const arr: Command[] = [];
      for (const c of this.commands.content.getValue(token)) arr.push(c);
      for (let i = 0; i < arr.length; i++) {
        const c = arr[i]!;
        // Coalesce a Clear immediately followed by a Render: both
        // share the run-arg framebuffer now (no per-cmd output to
        // disambiguate), so the merge is unconditional when the
        // pair is adjacent.
        if (c.kind === "Clear") {
          const next = arr[i + 1];
          if (next !== undefined && next.kind === "Render") {
            this.encodeRenderCommand(enc, next, framebuffer, token, c.values);
            i++;
            continue;
          }
        }
        this.encodeCommand(enc, c, framebuffer, token);
      }
    });
  }

  heapBucketCount(): number {
    let total = 0;
    for (const s of this._scenes.values()) total += s.heapBucketCount();
    return total;
  }

  heapDerivedTimings(): {
    pullMs: number; uploadMs: number; encodeMs: number; records: number;
  } {
    let pullMs = 0, uploadMs = 0, encodeMs = 0, records = 0;
    for (const s of this._scenes.values()) {
      const t = s.heapDerivedTimings();
      pullMs   += t.pullMs;
      uploadMs += t.uploadMs;
      encodeMs += t.encodeMs;
      records  += t.records;
    }
    return { pullMs, uploadMs, encodeMs, records };
  }

  async validateHeap(): Promise<{
    arenaBytes: number; issues: string[];
    okRefs: number; badRefs: number;
    drawTableRows: number; drawTableErrs: number; prefixSumErrs: number;
    attrAllocsChecked: number; attrAllocsBad: number;
    tilesChecked: number; tilesBad: number;
    vidChecks: number; vidBad: number;
    indicesHash: string;
  }> {
    let arenaBytes = 0, okRefs = 0, badRefs = 0;
    let drawTableRows = 0, drawTableErrs = 0, prefixSumErrs = 0;
    let attrAllocsChecked = 0, attrAllocsBad = 0;
    let tilesChecked = 0, tilesBad = 0;
    let vidChecks = 0, vidBad = 0;
    const indicesHashes: string[] = [];
    const issues: string[] = [];
    for (const s of this._scenes.values()) {
      const r = await s.validateHeap();
      arenaBytes += r.arenaBytes;
      okRefs += r.okRefs;
      badRefs += r.badRefs;
      drawTableRows += r.drawTableRows;
      drawTableErrs += r.drawTableErrs;
      prefixSumErrs += r.prefixSumErrs;
      attrAllocsChecked += r.attrAllocsChecked;
      attrAllocsBad += r.attrAllocsBad;
      tilesChecked += r.tilesChecked;
      tilesBad += r.tilesBad;
      vidChecks += r.vidChecks;
      vidBad += r.vidBad;
      indicesHashes.push(r.indicesHash);
      for (const i of r.issues) issues.push(i);
    }
    return {
      arenaBytes, issues, okRefs, badRefs,
      drawTableRows, drawTableErrs, prefixSumErrs,
      attrAllocsChecked, attrAllocsBad,
      tilesChecked, tilesBad,
      vidChecks, vidBad,
      indicesHash: indicesHashes.join(","),
    };
  }

  async simulateDraws(samples?: number): Promise<{
    emitsChecked: number; oob: number; issues: string[];
  }> {
    let emitsChecked = 0, oob = 0;
    const issues: string[] = [];
    for (const s of this._scenes.values()) {
      const r = await s.simulateDraws(samples);
      emitsChecked += r.emitsChecked;
      oob += r.oob;
      for (const i of r.issues) issues.push(i);
    }
    return { emitsChecked, oob, issues };
  }

  async probeBinarySearch(samples?: number): Promise<{
    emitsChecked: number; gpuMismatches: number; issues: string[];
  }> {
    let emitsChecked = 0, gpuMismatches = 0;
    const issues: string[] = [];
    for (const s of this._scenes.values()) {
      const r = await s.probeBinarySearch(samples);
      emitsChecked += r.emitsChecked;
      gpuMismatches += r.gpuMismatches;
      for (const i of r.issues) issues.push(i);
    }
    return { emitsChecked, gpuMismatches, issues };
  }

  async checkTriangleCoherence(samples?: number): Promise<{
    trianglesChecked: number; crossSlot: number; issues: string[];
  }> {
    let trianglesChecked = 0, crossSlot = 0;
    const issues: string[] = [];
    for (const s of this._scenes.values()) {
      const r = await s.checkTriangleCoherence(samples);
      trianglesChecked += r.trianglesChecked;
      crossSlot += r.crossSlot;
      for (const i of r.issues) issues.push(i);
    }
    return { trianglesChecked, crossSlot, issues };
  }

  dispose(): void {
    if (this._disposed) return;
    for (const s of this._scenes.values()) s.dispose();
    this._scenes.clear();
    this._disposed = true;
  }

  private encodeCommand(
    enc: GPUCommandEncoder, c: Command, framebuffer: IFramebuffer, token: AdaptiveToken,
  ): void {
    switch (c.kind) {
      case "Clear":  clear(enc, framebuffer, c.values); return;
      case "Copy":   copy(enc, c.copy); return;
      case "Custom": c.encode(enc); return;
      case "Render": this.encodeRenderCommand(enc, c, framebuffer, token); return;
    }
  }

  private encodeRenderCommand(
    enc: GPUCommandEncoder,
    cmd: Extract<Command, { kind: "Render" }>,
    framebuffer: IFramebuffer,
    token: AdaptiveToken,
    clearValues?: ClearValues,
  ): void {
    const scene = this.sceneFor(cmd, cmd.tree);
    scene.update(token);
    if (!scene.hasDraws() && clearValues === undefined) return;
    // Compute prep (drawTable prefix-sum) must happen outside any
    // render pass — encode it before we open the pass.
    scene.encodeComputePrep(enc, token);
    // Either we have draws or we need to clear — open a single pass.
    const pass = enc.beginRenderPass(beginPassDescriptor(framebuffer, clearValues));
    scene.encodeIntoPass(pass, token);
    pass.end();
  }

  private sceneFor(
    cmd: Extract<Command, { kind: "Render" }>,
    tree: RenderTree,
  ): HybridScene {
    let s = this._scenes.get(cmd);
    if (s === undefined) {
      s = compileHybridScene(this.ctx.device, this.signature, tree, {
        compileEffect: this.ctx.compileEffect,
        ...(this.ctx.heapEnabled !== undefined ? { heapEnabled: this.ctx.heapEnabled } : {}),
        ...(this.ctx.enableFamilyMerge === true ? { enableFamilyMerge: true } : {}),
        ...(this.ctx.enableDerivedUniforms === false ? { enableDerivedUniforms: false } : {}),
      });
      this._scenes.set(cmd, s);
    }
    return s;
  }
}

export function compileRenderTask(
  ctx: RuntimeContext,
  signature: FramebufferSignature,
  commands: alist<Command>,
): IRenderTask & {
  encode(enc: GPUCommandEncoder, framebuffer: IFramebuffer, token: AdaptiveToken): void;
} {
  return new RenderTask(ctx, signature, commands);
}

// `RenderTree` carries an `aval<RenderTree>` inside its `Adaptive`
// variant; this `void` reference keeps the `aval` import live.
const _avalGuard: ((x: aval<unknown>) => void) | undefined = undefined;
void _avalGuard;
