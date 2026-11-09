// RenderTree — intra-pass ordering structure for RenderObjects.
//
// `Ordered` children execute left-to-right; `Unordered` children
// can be reordered by the backend to minimise pipeline / bind-
// group / vertex-buffer state changes. The Adaptive variants
// fold reactive containers (alist, aset, aval) directly into the
// tree.

import type { aval, alist, aset } from "@aardworx/wombat.adaptive";
import type { RenderObject } from "./renderObject.js";

export type RenderTree =
  | { readonly kind: "Empty" }
  | { readonly kind: "Leaf"; readonly object: RenderObject }
  | { readonly kind: "Ordered"; readonly children: readonly RenderTree[] }
  | { readonly kind: "Unordered"; readonly children: readonly RenderTree[] }
  | { readonly kind: "Adaptive"; readonly tree: aval<RenderTree> }
  | { readonly kind: "OrderedFromList"; readonly children: alist<RenderTree> }
  | { readonly kind: "UnorderedFromSet"; readonly children: aset<RenderTree> }
  | { readonly kind: "Rows"; readonly set: import("./rowSet.js").RenderRowSet }
  | { readonly kind: "Guarded"; readonly active: aval<boolean>; readonly child: RenderTree };

export const RenderTree = {
  empty: { kind: "Empty" } as const satisfies RenderTree,
  leaf: (object: RenderObject): RenderTree => ({ kind: "Leaf", object }),
  ordered: (...children: RenderTree[]): RenderTree => ({ kind: "Ordered", children }),
  unordered: (...children: RenderTree[]): RenderTree => ({ kind: "Unordered", children }),
  adaptive: (tree: aval<RenderTree>): RenderTree => ({ kind: "Adaptive", tree }),
  orderedFromList: (children: alist<RenderTree>): RenderTree => ({ kind: "OrderedFromList", children }),
  unorderedFromSet: (children: aset<RenderTree>): RenderTree => ({ kind: "UnorderedFromSet", children }),
  rows: (set: import("./rowSet.js").RenderRowSet): RenderTree => ({ kind: "Rows", set }),
  /**
   * A subtree handed over before it is shown (`active` false), which may
   * hold values valid only once shown. While `active` is false it is read
   * under a guard: its structure and its draws' values are evaluated, and
   * if that raises it is left out (silently) and retried when `active`
   * changes; otherwise its draws are registered (gated by their own
   * `active`), so its first show is a flag flip. Once `active` is true, or
   * the subtree was taken in, it is an ordinary subtree: a raise is a real
   * error. Its draws should carry `active` themselves (the guard does not
   * hide them).
   */
  guarded: (active: aval<boolean>, child: RenderTree): RenderTree => ({ kind: "Guarded", active, child }),
} as const;
