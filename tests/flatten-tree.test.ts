// flattenRenderTree: one reader for the whole tree, against an oracle that
// reads the tree as it is now (every reactive node forced), under random
// edits of the cells, lists and sets in it — shared between places, with
// objects reached twice.

import { describe, expect, it } from "vitest";
import {
  AVal, AdaptiveToken, cval, clist, cset, transact,
  type ChangeableValue,
} from "@aardworx/wombat.adaptive";
import { RenderTree, type RenderObject } from "@aardworx/wombat.rendering/core";
import { flattenRenderTree } from "../packages/rendering/src/runtime/flattenTree.js";

/** A deterministic random source. */
function rng(seed: number) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

/** Objects by identity only: the flattening never looks inside. */
const pool: RenderObject[] = Array.from({ length: 12 }, (_, i) => ({ id: i } as unknown as RenderObject));

type Sources = {
  vals: ChangeableValue<RenderTree>[];
  lists: ReturnType<typeof clist<RenderTree>>[];
  sets: ReturnType<typeof cset<RenderTree>>[];
};

function gen(r: () => number, src: Sources, depth: number, share = true): RenderTree {
  const k = depth <= 0 ? Math.floor(r() * 2) : Math.floor(r() * 9);
  const kids = () => Array.from({ length: Math.floor(r() * 3) }, () => gen(r, src, depth - 1, share));
  switch (k) {
    case 0: return RenderTree.empty;
    case 1: return RenderTree.leaf(pool[Math.floor(r() * pool.length)]!);
    case 2: return RenderTree.ordered(...kids());
    case 3: return RenderTree.unordered(...kids());
    case 4: {
      // a cell, sometimes one already in the tree (shared; not in edits,
      // which could make a cell hold itself)
      if (share && src.vals.length > 0 && r() < 0.3) return RenderTree.adaptive(src.vals[Math.floor(r() * src.vals.length)]!);
      const c = cval(gen(r, src, depth - 1, share)); src.vals.push(c); return RenderTree.adaptive(c);
    }
    case 5: { const l = clist<RenderTree>(kids()); src.lists.push(l); return RenderTree.orderedFromList(l); }
    case 6: { const s = cset<RenderTree>(kids()); src.sets.push(s); return RenderTree.unorderedFromSet(s); }
    case 7: return RenderTree.guarded(AVal.constant(false), gen(r, src, depth - 1, share));
    default: return RenderTree.adaptive(AVal.constant(gen(r, src, depth - 1, share)));
  }
}

/** What the tree holds now, counted. */
function oracle(t: RenderTree, out: Map<RenderObject, number>): void {
  const top = AdaptiveToken.top;
  switch (t.kind) {
    case "Empty": return;
    case "Leaf": out.set(t.object, (out.get(t.object) ?? 0) + 1); return;
    case "Ordered": case "Unordered": for (const c of t.children) oracle(c, out); return;
    case "Adaptive": oracle(t.tree.getValue(top), out); return;
    case "OrderedFromList": for (const c of t.children.content.getValue(top)) oracle(c, out); return;
    case "UnorderedFromSet": for (const c of t.children.content.getValue(top)) oracle(c, out); return;
    case "Guarded": oracle(t.child, out); return;
    case "Rows": return;
  }
}

function present(m: Map<RenderObject, number>): Set<RenderObject> {
  return new Set([...m].filter(([, c]) => c > 0).map(([o]) => o));
}

describe("flattenRenderTree", () => {
  it("= the tree read now, under random edits", () => {
    for (let seed = 1; seed <= 60; seed++) {
      const r = rng(seed);
      const src: Sources = { vals: [], lists: [], sets: [] };
      const tree = gen(r, src, 4);
      const reader = flattenRenderTree(tree).getReader();
      const seen = new Set<RenderObject>();
      const pull = () => {
        for (const op of reader.getChanges(AdaptiveToken.top)) {
          if (op.count > 0) seen.add(op.value); else seen.delete(op.value);
        }
      };
      const check = (step: number) => {
        pull();
        const want = new Map<RenderObject, number>();
        oracle(tree, want);
        expect([...seen].map((o) => (o as unknown as { id: number }).id).sort(), `seed ${seed} step ${step}`)
          .toEqual([...present(want)].map((o) => (o as unknown as { id: number }).id).sort());
      };
      check(0);
      for (let step = 1; step <= 25; step++) {
        const edits = 1 + Math.floor(r() * 3);
        transact(() => {
          for (let e = 0; e < edits; e++) {
            const which = Math.floor(r() * 3);
            if (which === 0 && src.vals.length > 0) {
              src.vals[Math.floor(r() * src.vals.length)]!.value = gen(r, src, 2, false);
            } else if (which === 1 && src.lists.length > 0) {
              const l = src.lists[Math.floor(r() * src.lists.length)]!;
              const n = l.currentCount;
              const a = r();
              if (n > 0 && a < 0.35) l.removeAt(Math.floor(r() * n));
              else if (n > 0 && a < 0.6) l.setAt(Math.floor(r() * n), gen(r, src, 2, false));
              else l.insertAt(Math.floor(r() * (n + 1)), gen(r, src, 2, false));
            } else if (src.sets.length > 0) {
              const s = src.sets[Math.floor(r() * src.sets.length)]!;
              const items = [...s.content.getValue(AdaptiveToken.top)];
              if (items.length > 0 && r() < 0.4) s.remove(items[Math.floor(r() * items.length)]!);
              else s.add(gen(r, src, 2, false));
            }
          }
        });
        check(step);
      }
    }
  });

  it("an update reads only what changed: one reader for the tree", () => {
    const leaves = pool.slice(0, 4);
    const cells = leaves.map((l) => cval<RenderTree>(RenderTree.leaf(l)));
    const list = clist<RenderTree>(cells.map((c) => RenderTree.adaptive(c)));
    const reader = flattenRenderTree(RenderTree.orderedFromList(list)).getReader();
    expect(reader.getChanges(AdaptiveToken.top).count).toBe(4);
    transact(() => { cells[2]!.value = RenderTree.leaf(pool[7]!); });
    const d = [...reader.getChanges(AdaptiveToken.top)].map((op) => [(op.value as unknown as { id: number }).id, op.count]);
    expect(d.sort()).toEqual([[2, -1], [7, 1]]);
    // a removed child's cell no longer marks the flattening
    transact(() => { list.removeAt(2); });
    reader.getChanges(AdaptiveToken.top);
    expect(reader.outOfDate).toBe(false);
    transact(() => { cells[2]!.value = RenderTree.leaf(pool[9]!); });
    expect(reader.outOfDate).toBe(false);
  });
});
