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
import { ScenePass } from "./scenePass.js";

export interface RuntimeContext {
  readonly device: GPUDevice;
  /**
   * Compile an `Effect` against the target framebuffer signature.
   * The signature determines the canonical fragment-output layout
   * (`signature.colorNames[i]` ↔ `@location(i)`); the shader's
   * `linkFragmentOutputs` pass uses it to re-pin and DCE outputs.
   */
  readonly compileEffect: (effect: Effect, signature: FramebufferSignature) => CompiledEffect;
}

class RenderTask implements IRenderTask {
  /**
   * Cache of `ScenePass`es keyed on the `Render` command itself.
   * Holding ScenePass per command means the same `RenderTree`
   * reused across frames keeps its incrementally-maintained
   * walker state.
   */
  private readonly _scenes = new Map<unknown, ScenePass>();
  private _disposed = false;

  constructor(
    private readonly ctx: RuntimeContext,
    private readonly commands: alist<Command>,
  ) {}

  run(token: AdaptiveToken): void {
    if (this._disposed) throw new Error("RenderTask: run after dispose");
    const enc = this.ctx.device.createCommandEncoder();
    this.encode(enc, token);
    this.ctx.device.queue.submit([enc.finish()]);
  }

  encode(enc: GPUCommandEncoder, token: AdaptiveToken): void {
    if (this._disposed) throw new Error("RenderTask: encode after dispose");
    RenderContext.withEncoder(enc, () => {
      const arr: Command[] = [];
      for (const c of this.commands.content.getValue(token)) arr.push(c);
      for (let i = 0; i < arr.length; i++) {
        const c = arr[i]!;
        if (c.kind === "Clear") {
          const next = arr[i + 1];
          if (next !== undefined && next.kind === "Render" && next.output === c.output) {
            this.encodeRenderCommand(enc, next, token, c.values);
            i++;
            continue;
          }
        }
        this.encodeCommand(enc, c, token);
      }
    });
  }

  dispose(): void {
    if (this._disposed) return;
    for (const s of this._scenes.values()) s.dispose();
    this._scenes.clear();
    this._disposed = true;
  }

  private encodeCommand(enc: GPUCommandEncoder, c: Command, token: AdaptiveToken): void {
    switch (c.kind) {
      case "Clear":  clear(enc, c.output.getValue(token), c.values); return;
      case "Copy":   copy(enc, c.copy); return;
      case "Custom": c.encode(enc); return;
      case "Render": this.encodeRenderCommand(enc, c, token); return;
    }
  }

  private encodeRenderCommand(
    enc: GPUCommandEncoder,
    cmd: Extract<Command, { kind: "Render" }>,
    token: AdaptiveToken,
    clearValues?: ClearValues,
  ): void {
    const output = cmd.output.getValue(token);
    const scene = this.sceneFor(cmd, output, cmd.tree);
    const leaves = scene.resolve(token);
    if (leaves.length === 0 && clearValues === undefined) return;
    // Either we have draws or we need to clear — open a single pass.
    const pass = enc.beginRenderPass(beginPassDescriptor(output, clearValues));
    for (const leaf of leaves) leaf.record(pass, token);
    pass.end();
  }

  private sceneFor(
    cmd: Extract<Command, { kind: "Render" }>,
    output: IFramebuffer,
    tree: RenderTree,
  ): ScenePass {
    let s = this._scenes.get(cmd);
    if (s === undefined) {
      s = new ScenePass(this.ctx.device, output.signature, tree, this.ctx.compileEffect);
      this._scenes.set(cmd, s);
    }
    return s;
    // Note: `output.signature` is read here only to seed the
    // ScenePass; subsequent frames continue to use this signature
    // even if the framebuffer aval emits a different sig. That
    // matches the `(RenderObject, signature)` cache invariant —
    // changing signature requires a fresh Render command.
  }
}

export function compileRenderTask(ctx: RuntimeContext, commands: alist<Command>): IRenderTask & {
  encode(enc: GPUCommandEncoder, token: AdaptiveToken): void;
} {
  return new RenderTask(ctx, commands);
}

// `RenderTree` carries an `aval<RenderTree>` inside its `Adaptive`
// variant; this `void` reference keeps the `aval` import live.
const _avalGuard: ((x: aval<unknown>) => void) | undefined = undefined;
void _avalGuard;
