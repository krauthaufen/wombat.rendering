// ScenePass — delta-driven walker for one `Render` command's
// `RenderTree`. Built once at task-compile time; subscribes to the
// dynamic subtrees (`Adaptive` / `OrderedFromList` /
// `UnorderedFromSet`); incrementally maintains its set of
// PreparedRenderObjects.
//
// Per-frame cost = O(deltas) + O(leaves) for emission. The static
// recursion through Empty / Leaf / Ordered / Unordered nodes that
// the previous walker did each frame is gone — those are folded
// into a stable NodeWalker tree at construction.
//
// Lifted from Aardvark.Rendering's CommandTask delta handling.
// We keep the *adaptive bookkeeping* pattern (per-RO AdaptiveObject,
// reader-driven splice) and drop the native command-stream linked-
// list — that was a .NET↔native amortisation in the F# build with
// no analogue in WebGPU/JS.

import {
  type AdaptiveToken,
  type IIndexListReader,
  type IHashSetReader,
  MapExt,
  type Index,
  type aval,
  type alist,
  type aset,
} from "@aardworx/wombat.adaptive";
import {
  type CompiledEffect,
  type Effect,
  type FramebufferSignature,
  type RenderObject,
  type RenderTree,
} from "@aardworx/wombat.rendering-core";
import {
  prepareRenderObject,
  type PreparedRenderObject,
} from "@aardworx/wombat.rendering-resources";

interface BuildContext {
  readonly device: GPUDevice;
  readonly compileEffect: (e: Effect) => CompiledEffect;
  readonly signature: FramebufferSignature;
  readonly stats: WalkerStats;
}

/**
 * Counters surfaced for tests. `prepareCount` only goes up when
 * `prepareRenderObject` is called for a fresh RenderObject; it
 * does not increment per frame on an unchanged scene.
 */
export interface WalkerStats {
  prepareCount: number;
}

abstract class NodeWalker {
  /** Pull deltas from any subscribed readers and propagate to children. */
  abstract update(token: AdaptiveToken): void;
  /** Push this subtree's current leaves into `out`, in render order. */
  abstract emit(out: PreparedRenderObject[]): void;
  /** Release every PreparedRenderObject this walker owns. */
  abstract dispose(): void;
}

class EmptyWalker extends NodeWalker {
  update(): void {}
  emit(): void {}
  dispose(): void {}
}

class LeafWalker extends NodeWalker {
  private readonly prepared: PreparedRenderObject;
  constructor(ctx: BuildContext, obj: RenderObject) {
    super();
    const eff = ctx.compileEffect(obj.effect);
    this.prepared = prepareRenderObject(ctx.device, obj, eff, ctx.signature, {
      effectId: obj.effect.id,
    });
    this.prepared.acquire();
    ctx.stats.prepareCount++;
  }
  update(): void {}
  emit(out: PreparedRenderObject[]): void { out.push(this.prepared); }
  dispose(): void { this.prepared.release(); }
}

class OrderedWalker extends NodeWalker {
  constructor(private readonly children: readonly NodeWalker[]) { super(); }
  update(token: AdaptiveToken): void { for (const c of this.children) c.update(token); }
  emit(out: PreparedRenderObject[]): void { for (const c of this.children) c.emit(out); }
  dispose(): void { for (const c of this.children) c.dispose(); }
}

// Same identity-rank counter used by the previous walker. Sorting
// Unordered subtrees by pipeline-then-group0-layout to minimise
// state changes inside the render pass.
const sortRanks = new WeakMap<object, number>();
let nextSortRank = 1;
function rankOf(o: object): number {
  let r = sortRanks.get(o);
  if (r === undefined) { r = nextSortRank++; sortRanks.set(o, r); }
  return r;
}
function compareLeaves(a: PreparedRenderObject, b: PreparedRenderObject): number {
  const pa = rankOf(a.pipeline);
  const pb = rankOf(b.pipeline);
  if (pa !== pb) return pa - pb;
  const la = a.groups[0]?.layout;
  const lb = b.groups[0]?.layout;
  if (la !== undefined && lb !== undefined && la !== lb) return rankOf(la) - rankOf(lb);
  return 0;
}

class UnorderedWalker extends NodeWalker {
  constructor(private readonly children: readonly NodeWalker[]) { super(); }
  update(token: AdaptiveToken): void { for (const c of this.children) c.update(token); }
  emit(out: PreparedRenderObject[]): void {
    const start = out.length;
    for (const c of this.children) c.emit(out);
    // Sort just our slice in-place.
    const slice = out.splice(start);
    slice.sort(compareLeaves);
    for (const p of slice) out.push(p);
  }
  dispose(): void { for (const c of this.children) c.dispose(); }
}

class AdaptiveWalker extends NodeWalker {
  private current: NodeWalker | undefined;
  private currentTree: RenderTree | undefined;
  constructor(private readonly source: aval<RenderTree>, private readonly ctx: BuildContext) {
    super();
  }
  update(token: AdaptiveToken): void {
    const t = this.source.getValue(token);
    if (t !== this.currentTree) {
      this.current?.dispose();
      this.current = build(t, this.ctx);
      this.currentTree = t;
    }
    this.current!.update(token);
  }
  emit(out: PreparedRenderObject[]): void { this.current?.emit(out); }
  dispose(): void { this.current?.dispose(); this.current = undefined; }
}

class OrderedFromListWalker extends NodeWalker {
  private readonly reader: IIndexListReader<RenderTree>;
  // Sorted-by-Index map. Reassigned (immutable persistent) on every delta.
  private map: MapExt<Index, NodeWalker> = MapExt.empty(indexCompare);
  constructor(source: alist<RenderTree>, private readonly ctx: BuildContext) {
    super();
    this.reader = source.getReader();
  }
  update(token: AdaptiveToken): void {
    const deltas = this.reader.getChanges(token);
    for (const [idx, op] of deltas) {
      if (op.tag === "Set") {
        const old = this.map.tryFind(idx);
        if (old !== undefined) old.dispose();
        const w = build(op.value, this.ctx);
        // Update the new child immediately so its initial state
        // is consistent before the next emit.
        w.update(token);
        this.map = this.map.add(idx, w);
      } else {
        const old = this.map.tryFind(idx);
        if (old !== undefined) {
          old.dispose();
          this.map = this.map.remove(idx);
        }
      }
    }
    // Existing children may have dirty sub-readers — update them too.
    for (const [, w] of this.map) w.update(token);
  }
  emit(out: PreparedRenderObject[]): void {
    for (const [, w] of this.map) w.emit(out);
  }
  dispose(): void {
    for (const [, w] of this.map) w.dispose();
    this.map = MapExt.empty(indexCompare);
  }
}

class UnorderedFromSetWalker extends NodeWalker {
  private readonly reader: IHashSetReader<RenderTree>;
  private readonly map = new Map<RenderTree, NodeWalker>();
  constructor(source: aset<RenderTree>, private readonly ctx: BuildContext) {
    super();
    this.reader = source.getReader();
  }
  update(token: AdaptiveToken): void {
    const deltas = this.reader.getChanges(token);
    for (const op of deltas) {
      if (op.count > 0) {
        if (!this.map.has(op.value)) {
          const w = build(op.value, this.ctx);
          w.update(token);
          this.map.set(op.value, w);
        }
      } else if (op.count < 0) {
        const old = this.map.get(op.value);
        if (old !== undefined) {
          old.dispose();
          this.map.delete(op.value);
        }
      }
    }
    for (const w of this.map.values()) w.update(token);
  }
  emit(out: PreparedRenderObject[]): void {
    const start = out.length;
    for (const w of this.map.values()) w.emit(out);
    const slice = out.splice(start);
    slice.sort(compareLeaves);
    for (const p of slice) out.push(p);
  }
  dispose(): void {
    for (const w of this.map.values()) w.dispose();
    this.map.clear();
  }
}

function indexCompare(a: Index, b: Index): number { return a.compareTo(b); }

function build(tree: RenderTree, ctx: BuildContext): NodeWalker {
  switch (tree.kind) {
    case "Empty": return new EmptyWalker();
    case "Leaf": return new LeafWalker(ctx, tree.object);
    case "Ordered": return new OrderedWalker(tree.children.map(c => build(c, ctx)));
    case "Unordered": return new UnorderedWalker(tree.children.map(c => build(c, ctx)));
    case "Adaptive": return new AdaptiveWalker(tree.tree, ctx);
    case "OrderedFromList": return new OrderedFromListWalker(tree.children, ctx);
    case "UnorderedFromSet": return new UnorderedFromSetWalker(tree.children, ctx);
  }
}

// ---------------------------------------------------------------------------
// ScenePass — public entry
// ---------------------------------------------------------------------------

export class ScenePass {
  private readonly root: NodeWalker;
  /** Bumped on `prepareRenderObject`. Tests use this to confirm O(deltas) prep cost. */
  readonly stats: WalkerStats = { prepareCount: 0 };

  constructor(
    device: GPUDevice,
    signature: FramebufferSignature,
    tree: RenderTree,
    compileEffect: (e: Effect) => CompiledEffect,
  ) {
    const ctx: BuildContext = { device, signature, compileEffect, stats: this.stats };
    this.root = build(tree, ctx);
  }

  /**
   * Pull deltas from every dynamic subtree, splice the affected
   * walkers, and produce the current ordered list of leaves.
   * Idempotent on a clean adaptive graph (no readers fire).
   */
  resolve(token: AdaptiveToken): PreparedRenderObject[] {
    this.root.update(token);
    const out: PreparedRenderObject[] = [];
    this.root.emit(out);
    return out;
  }

  dispose(): void {
    this.root.dispose();
  }
}
