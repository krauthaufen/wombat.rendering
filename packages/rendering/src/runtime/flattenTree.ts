// flattenTree — reactive `RenderTree → aset<RenderObject>`.
//
// Inside a single Render command, draw order is not load-bearing
// (formal contract: an `aset` of objects, not a sequence). The two
// node-kinds `Ordered` and `Unordered` collapse to the same flat
// set — anything ordering-sensitive belongs at the Command level
// (separate Render passes, in the Command list's declared order).
//
// Reactive variants (`Adaptive`, `OrderedFromList`, `UnorderedFromSet`)
// stay reactive: marks propagate through `bind` / `collect`, and the
// resulting aset reflects current tree state without a manual rebuild.
//
// `OrderedFromList`'s `alist<RenderTree>` is converted via
// `AListBridges.toASet` — this drops index/order info, which we
// wouldn't honor anyway.

import { ASet, type aset } from "@aardworx/wombat.adaptive";
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

/** Flatten a reactive container of subtrees. The static children go
 *  through `collectSeq` (plain iterables — ONE reader for the whole
 *  set, instead of a ConstantAset + reader + subscription PER LEAF,
 *  which at heap scale was megabytes of ballast); reactive children
 *  keep the full `collect`. */
function flattenChildSet(children: aset<RenderTree>): aset<RenderObject> {
  const staticPart = children.filter(isStaticTree).collectSeq(staticObjects);
  const reactivePart = children.filter(t => !isStaticTree(t)).collect(flattenRenderTree);
  return ASet.union(staticPart, reactivePart);
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
