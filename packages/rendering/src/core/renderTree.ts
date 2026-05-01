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
  | { readonly kind: "UnorderedFromSet"; readonly children: aset<RenderTree> };

export const RenderTree = {
  empty: { kind: "Empty" } as const satisfies RenderTree,
  leaf: (object: RenderObject): RenderTree => ({ kind: "Leaf", object }),
  ordered: (...children: RenderTree[]): RenderTree => ({ kind: "Ordered", children }),
  unordered: (...children: RenderTree[]): RenderTree => ({ kind: "Unordered", children }),
  adaptive: (tree: aval<RenderTree>): RenderTree => ({ kind: "Adaptive", tree }),
  orderedFromList: (children: alist<RenderTree>): RenderTree => ({ kind: "OrderedFromList", children }),
  unorderedFromSet: (children: aset<RenderTree>): RenderTree => ({ kind: "UnorderedFromSet", children }),
} as const;
