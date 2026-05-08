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

export function flattenRenderTree(tree: RenderTree): aset<RenderObject> {
  switch (tree.kind) {
    case "Empty":
      return ASet.empty<RenderObject>();
    case "Leaf":
      return ASet.single(tree.object);
    case "Ordered":
    case "Unordered": {
      // N-ary fold over binary union. Static children, so this is
      // a one-shot construction; the resulting aset is reactive at
      // each child's level (a child can itself be Adaptive).
      const children = tree.children.map(flattenRenderTree);
      if (children.length === 0) return ASet.empty<RenderObject>();
      return children.reduce((acc, c) => ASet.union(acc, c));
    }
    case "Adaptive":
      return ASet.bind(flattenRenderTree, tree.tree);
    case "OrderedFromList":
      return ASet.collect(flattenRenderTree, AListBridges.toASet(tree.children));
    case "UnorderedFromSet":
      return ASet.collect(flattenRenderTree, tree.children);
  }
}
