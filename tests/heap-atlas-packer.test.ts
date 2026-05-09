// Atlas packer + BVH tests.
//
// Two parts:
//   1. BvhTree2d direct unit tests — empty queries, add+lookup,
//      remove+lookup, stress add-then-remove returns to empty.
//   2. TexturePacking property tests via fast-check — random
//      insert/remove sequences keep the invariant: every used rect is
//      non-overlapping and inside the atlas; remove-all returns to
//      empty; tryAdd returns null iff no fit.

import { describe, expect, it } from "vitest";
import fc from "fast-check";
import { Box2d, V2d, V2i } from "@aardworx/wombat.base";
import { BvhTree2d } from "../packages/rendering/src/runtime/textureAtlas/bvhTree2d.js";
import { TexturePacking } from "../packages/rendering/src/runtime/textureAtlas/packer.js";

const box = (x0: number, y0: number, x1: number, y1: number): Box2d =>
  Box2d.fromMinMax(new V2d(x0, y0), new V2d(x1, y1));

describe("BvhTree2d", () => {
  it("empty tree returns nothing for any query", () => {
    const t = BvhTree2d.empty<number, string>();
    expect([...t.getIntersecting(box(0, 0, 100, 100))]).toEqual([]);
    expect(t.count).toBe(0);
    expect(t.isEmpty).toBe(true);
  });

  it("add then query each: result includes the added box", () => {
    let t = BvhTree2d.empty<number, string>();
    const boxes: Array<readonly [number, Box2d, string]> = [];
    let seed = 1;
    const rnd = (): number => {
      // simple deterministic LCG
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    for (let i = 0; i < 100; i++) {
      const x0 = Math.floor(rnd() * 1000);
      const y0 = Math.floor(rnd() * 1000);
      const x1 = x0 + 1 + Math.floor(rnd() * 50);
      const y1 = y0 + 1 + Math.floor(rnd() * 50);
      const b = box(x0, y0, x1, y1);
      const v = `v${i}`;
      boxes.push([i, b, v]);
      t = t.add(i, b, v);
    }
    expect(t.count).toBe(100);
    for (const [k, b] of boxes) {
      const hits = [...t.getIntersecting(b)];
      const found = hits.some(([kk]) => kk === k);
      expect(found).toBe(true);
    }
  });

  it("remove then query: no longer included", () => {
    let t = BvhTree2d.empty<number, string>();
    t = t.add(1, box(0, 0, 10, 10), "a");
    t = t.add(2, box(20, 20, 30, 30), "b");
    expect([...t.getIntersecting(box(0, 0, 5, 5))].some(([k]) => k === 1)).toBe(true);
    t = t.remove(1);
    expect([...t.getIntersecting(box(0, 0, 5, 5))].some(([k]) => k === 1)).toBe(false);
    expect(t.count).toBe(1);
  });

  it("stress: add then remove 1000 boxes returns to empty", () => {
    let t = BvhTree2d.empty<number, string>();
    let seed = 7;
    const rnd = (): number => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    const keys: number[] = [];
    for (let i = 0; i < 1000; i++) {
      const x0 = Math.floor(rnd() * 10000);
      const y0 = Math.floor(rnd() * 10000);
      const x1 = x0 + 1 + Math.floor(rnd() * 100);
      const y1 = y0 + 1 + Math.floor(rnd() * 100);
      t = t.add(i, box(x0, y0, x1, y1), `v${i}`);
      keys.push(i);
    }
    expect(t.count).toBe(1000);
    // Shuffle keys for removal order.
    for (let i = keys.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const tmp = keys[i]!;
      keys[i] = keys[j]!;
      keys[j] = tmp;
    }
    for (const k of keys) t = t.remove(k);
    expect(t.count).toBe(0);
    expect(t.isEmpty).toBe(true);
  });

  it("tryRemove yields the value", () => {
    let t = BvhTree2d.empty<string, number>();
    t = t.add("a", box(0, 0, 1, 1), 42);
    const r = t.tryRemove("a");
    expect(r).not.toBeUndefined();
    expect(r![0]).toBe(42);
    expect(r![1].count).toBe(0);
  });
});

// ─── Packer property tests ──────────────────────────────────────────

const ATLAS_SIZE = 128;

function rectsOverlap(a: { min: V2i; max: V2i }, b: { min: V2i; max: V2i }): boolean {
  return (
    a.min.x <= b.max.x && a.max.x >= b.min.x &&
    a.min.y <= b.max.y && a.max.y >= b.min.y
  );
}

function inside(b: { min: V2i; max: V2i }, sz: V2i): boolean {
  return b.min.x >= 0 && b.min.y >= 0 && b.max.x < sz.x && b.max.y < sz.y;
}

describe("TexturePacking properties", () => {
  it("after random insert+remove, used rects are non-overlapping and in range", () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.tuple(fc.integer({ min: 1, max: 32 }), fc.integer({ min: 1, max: 32 })),
          { minLength: 0, maxLength: 30 },
        ),
        fc.array(fc.integer({ min: 0, max: 29 }), { minLength: 0, maxLength: 30 }),
        (sizes, removalIndices) => {
          let p = TexturePacking.empty<number>(new V2i(ATLAS_SIZE, ATLAS_SIZE), true);
          for (let i = 0; i < sizes.length; i++) {
            const [w, h] = sizes[i]!;
            const next = p.tryAdd(i, new V2i(w, h));
            if (next !== null) p = next;
          }
          // Remove a few.
          for (const idx of removalIndices) {
            if (idx < sizes.length) p = p.remove(idx);
          }
          // All used rects in range and non-overlapping.
          const arr = [...p.used];
          const sz = new V2i(ATLAS_SIZE, ATLAS_SIZE);
          for (let i = 0; i < arr.length; i++) {
            const [, ri] = arr[i]!;
            expect(inside(ri, sz)).toBe(true);
            for (let j = i + 1; j < arr.length; j++) {
              const [, rj] = arr[j]!;
              expect(rectsOverlap(ri, rj)).toBe(false);
            }
          }
        },
      ),
      { numRuns: 50 },
    );
  });

  it("removing all keys returns to an empty packing", () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.tuple(fc.integer({ min: 1, max: 16 }), fc.integer({ min: 1, max: 16 })),
          { minLength: 1, maxLength: 20 },
        ),
        sizes => {
          let p = TexturePacking.empty<number>(new V2i(ATLAS_SIZE, ATLAS_SIZE), true);
          const inserted: number[] = [];
          for (let i = 0; i < sizes.length; i++) {
            const [w, h] = sizes[i]!;
            const next = p.tryAdd(i, new V2i(w, h));
            if (next !== null) {
              p = next;
              inserted.push(i);
            }
          }
          for (const id of inserted) p = p.remove(id);
          expect(p.count).toBe(0);
          expect(p.isEmpty).toBe(true);
        },
      ),
      { numRuns: 30 },
    );
  });

  it("tryAdd is null iff the rect can't fit", () => {
    // After a packing is full, adding a too-large rect must return null.
    let p = TexturePacking.empty<number>(new V2i(8, 8), true);
    const next = p.tryAdd(1, new V2i(8, 8));
    expect(next).not.toBeNull();
    p = next!;
    // 1×1 won't fit any free area.
    expect(p.tryAdd(2, new V2i(1, 1))).toBeNull();
  });

  it("square produces correct results (deterministic small case)", () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.tuple(fc.integer({ min: 1, max: 32 }), fc.integer({ min: 1, max: 32 })),
          { minLength: 1, maxLength: 12 },
        ),
        sizes => {
          // Direct pack with a comfortable atlas to assert algorithmic invariants.
          const rects = sizes.map(([w, h], i) => [i, new V2i(w, h)] as const);
          let p = TexturePacking.empty<number>(new V2i(256, 256), true);
          for (const [k, sz] of rects) {
            const next = p.tryAdd(k, sz);
            expect(next).not.toBeNull();
            p = next!;
          }
          const arr = [...p.used];
          const sz = new V2i(256, 256);
          for (let i = 0; i < arr.length; i++) {
            const [, ri] = arr[i]!;
            expect(inside(ri, sz)).toBe(true);
            for (let j = i + 1; j < arr.length; j++) {
              const [, rj] = arr[j]!;
              expect(rectsOverlap(ri, rj)).toBe(false);
            }
          }
          // Sizes match (allow rotation: w×h or h×w).
          for (const [id, sz] of rects) {
            const placed = p.used.get(id);
            expect(placed).not.toBeUndefined();
            const pw = placed!.max.x - placed!.min.x + 1;
            const ph = placed!.max.y - placed!.min.y + 1;
            const ok = (pw === sz.x && ph === sz.y) || (pw === sz.y && ph === sz.x);
            expect(ok).toBe(true);
          }
        },
      ),
      { numRuns: 30 },
    );
  });
});
