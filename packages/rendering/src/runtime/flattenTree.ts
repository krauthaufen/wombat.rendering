// flattenTree — reactive `RenderTree → aset<RenderObject>`.
//
// Inside a single Render command, draw order is not load-bearing
// (formal contract: an `aset` of objects, not a sequence). The two
// node-kinds `Ordered` and `Unordered` collapse to the same flat
// set — anything ordering-sensitive belongs at the Command level
// (separate Render passes, in the Command list's declared order).
//
// Reactive variants (`Adaptive`, `OrderedFromList`, `UnorderedFromSet`)
// stay reactive: marks propagate through the fused reader, and the
// resulting aset reflects current tree state without a manual rebuild.
//
// `OrderedFromList`'s `alist<RenderTree>` is converted via
// `AListBridges.toASet` — this drops index/order info, which we
// wouldn't honor anyway.
//
// FUSION: a reactive child-set used to lower through FIVE aset stages
// (`filter(static).collectSeq(...)` ∪ `filter(!static).collect(...)`),
// each with its own History + full per-element state — at collection
// scale that is several complete N-element states of pure plumbing.
// `FlattenChildSetReader` fuses them into ONE reader: static subtrees
// (the overwhelmingly common case) enumerate their RenderObjects
// directly; reactive subtrees get an inner reader exactly like
// `ASet.collect`'s. One History, one state, two caches.

import {
  ASet, AbstractDirtyReader, Cache, HashSetDelta, SetOperation,
  hashSetDeltaMonoid,
  type AbstractReader, type AdaptiveToken, type IHashSetReader, type aset,
} from "@aardworx/wombat.adaptive";
import { AListBridges } from "@aardworx/wombat.adaptive";
import type { RenderObject } from "../core/renderObject.js";
import type { RenderTree } from "../core/renderTree.js";

/** True when `tree` contains no reactive nodes — its RenderObjects can
 *  be enumerated as a plain iterable (no inner aset / reader needed). */
function isStaticTree(tree: RenderTree): boolean {
  switch (tree.kind) {
    case "Empty":
    case "Leaf":
      return true;
    case "Ordered":
    case "Unordered":
      return tree.children.every(isStaticTree);
    default:
      return false;
  }
}

/** Enumerate a static subtree's RenderObjects (pre: `isStaticTree`). */
function* staticObjects(tree: RenderTree): Iterable<RenderObject> {
  switch (tree.kind) {
    case "Empty":
      return;
    case "Leaf":
      yield tree.object;
      return;
    case "Ordered":
    case "Unordered":
      for (const c of tree.children) yield* staticObjects(c);
      return;
    default:
      throw new Error(`staticObjects: reactive node '${tree.kind}'`);
  }
}

const INNER_TAG = "FlattenInnerReader";
const _tagIsInnerTag = (tag: unknown): boolean => tag === INNER_TAG;

/**
 * Fused flattener for a reactive container of subtrees. Per child
 * delta: a STATIC subtree contributes its objects directly (cached as
 * an array so removal emits exactly what addition emitted); a
 * reactive subtree contributes through an inner reader with
 * `ASet.collect`'s dirty-tracking protocol.
 */
class FlattenChildSetReader extends AbstractDirtyReader<
  IHashSetReader<RenderObject>,
  HashSetDelta<RenderObject>
> {
  private readonly _reader: IHashSetReader<RenderTree>;
  private readonly _static: Cache<RenderTree, RenderObject[]>;
  private readonly _inner: Cache<RenderTree, IHashSetReader<RenderObject>>;

  constructor(children: aset<RenderTree>) {
    super(hashSetDeltaMonoid<RenderObject>(), _tagIsInnerTag);
    this._reader = children.getReader();
    this._static = new Cache<RenderTree, RenderObject[]>(
      (t) => [...staticObjects(t)],
    );
    this._inner = new Cache<RenderTree, IHashSetReader<RenderObject>>(
      (t) => {
        const r = flattenRenderTree(t).getReader();
        r.tag = INNER_TAG;
        return r;
      },
    );
  }

  override compute(
    tok: AdaptiveToken,
    dirty: Set<IHashSetReader<RenderObject>>,
  ): HashSetDelta<RenderObject> {
    let deltas = this._reader.getChanges(tok).collect((d) => {
      const t = d.value;
      if (d.count === 1) {
        if (isStaticTree(t)) {
          let out = HashSetDelta.empty<RenderObject>();
          for (const ro of this._static.invoke(t)) out = out.add(SetOperation.add(ro));
          return out;
        }
        const r = this._inner.invoke(t);
        dirty.delete(r);
        return r.getChanges(tok);
      }
      if (d.count === -1) {
        if (isStaticTree(t)) {
          let out = HashSetDelta.empty<RenderObject>();
          for (const ro of this._static.revokeUnsafe(t)) out = out.add(SetOperation.rem(ro));
          return out;
        }
        const r = this._inner.tryRevokeAndGetDeleted(t);
        if (r === undefined) return HashSetDelta.empty<RenderObject>();
        dirty.delete(r.value);
        if (r.deleted) {
          r.value.outputs.remove(this);
          return r.value.state.removeAll();
        }
        return r.value.getChanges(tok);
      }
      throw new Error("flattenTree: unexpected delta count");
    });
    for (const d of dirty) {
      deltas = deltas.combine(d.getChanges(tok));
    }
    return deltas;
  }
}

function flattenChildSet(children: aset<RenderTree>): aset<RenderObject> {
  return ASet.ofReader(
    () => new FlattenChildSetReader(children) as unknown as AbstractReader<HashSetDelta<RenderObject>>,
  );
}

export function flattenRenderTree(tree: RenderTree): aset<RenderObject> {
  switch (tree.kind) {
    case "Empty":
      return ASet.empty<RenderObject>();
    case "Leaf":
      return ASet.single(tree.object);
    case "Ordered":
    case "Unordered": {
      // Fully-static subtree → one constant aset over a plain array.
      if (isStaticTree(tree)) return ASet.ofArray([...staticObjects(tree)]);
      // N-ary union via `unionMany(aset<aset<T>>)`. The naïve
      // `children.reduce(ASet.union)` builds a 2N-deep nested-union
      // tree; the reader's traversal recurses through it and blows
      // the stack at ~2K leaves. `unionMany` is O(1)-deep regardless.
      const children = tree.children.map(flattenRenderTree);
      if (children.length === 0) return ASet.empty<RenderObject>();
      if (children.length === 1) return children[0]!;
      return ASet.unionMany(ASet.ofArray(children));
    }
    case "Adaptive":
      return ASet.bind(flattenRenderTree, tree.tree);
    case "OrderedFromList":
      return flattenChildSet(AListBridges.toASet(tree.children));
    case "UnorderedFromSet":
      return flattenChildSet(tree.children);
  }
}
