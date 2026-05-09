// TexturePacking — incremental immutable MaxRects-style 2D rectangle
// packer over `BvhTree2d`.
//
// Port of `Aardvark.Geometry.TexturePacking<'a>` from
// `aardvark.base/src/Aardvark.Geometry/TexturePacking.fs`. Algorithm:
//   - Atlas is described by a set of free MaxRects (contained in a
//     `BvhTree2d` keyed on the rect itself for O(log N) queries).
//   - tryAdd picks the free rect with the smallest waste that fits the
//     incoming size (with optional 90° rotation), splits the chosen
//     rect into the placed sub-rect plus residual MaxRects.
//   - Remove restores a sub-rect and merges with adjacent free rects
//     using the F# `merge` routine, which incrementally maintains
//     non-overlapping MaxRects covering the released region + neighbours.
//
// The wombat port keys the BVH on `Box2i` value-objects. We use a
// stable string key (`"minX,minY,maxX,maxY"`) so identical rectangles
// share a Map slot — Box2i instances themselves are reference types in
// JS and would not collide otherwise.

import { Box2d, Box2i, V2d, V2i } from "@aardworx/wombat.base";
import { BvhTree2d } from "./bvhTree2d.js";

// ─── Box2i helpers (mirroring the F# helpers in this module) ───────

const boxKey = (b: Box2i): string => {
  const mn = b.min, mx = b.max;
  return `${mn.x},${mn.y},${mx.x},${mx.y}`;
};

const sizeOf = (b: Box2i): V2i => {
  const mn = b.min, mx = b.max;
  return new V2i(mx.x - mn.x + 1, mx.y - mn.y + 1);
};

const isValid2i = (b: Box2i): boolean => {
  const mn = b.min, mx = b.max;
  return mn.x <= mx.x && mn.y <= mx.y;
};

const intersects2i = (a: Box2i, b: Box2i): boolean => {
  const amin = a.min, amax = a.max, bmin = b.min, bmax = b.max;
  return amin.x <= bmax.x && amax.x >= bmin.x && amin.y <= bmax.y && amax.y >= bmin.y;
};

const intersectsTouching2i = (a: Box2i, b: Box2i): boolean => {
  // Mirror F# `intersects` inside `maxRects`: a 1-unit halo so adjacent
  // rectangles count as overlapping (used to detect contact).
  const amin = a.min, amax = a.max, bmin = b.min, bmax = b.max;
  return amin.x <= bmax.x + 1 && amax.x >= bmin.x - 1 && amin.y <= bmax.y + 1 && amax.y >= bmin.y - 1;
};

const contains2i = (outer: Box2i, inner: Box2i): boolean => {
  const omin = outer.min, omax = outer.max, imin = inner.min, imax = inner.max;
  return omin.x <= imin.x && omin.y <= imin.y && omax.x >= imax.x && omax.y >= imax.y;
};

const equals2i = (a: Box2i, b: Box2i): boolean => {
  const amin = a.min, amax = a.max, bmin = b.min, bmax = b.max;
  return amin.x === bmin.x && amin.y === bmin.y && amax.x === bmax.x && amax.y === bmax.y;
};

const mkBox = (minX: number, minY: number, maxX: number, maxY: number): Box2i =>
  new Box2i(minX, minY, maxX, maxY);

// `createBox`: half-pixel padded Box2d for BVH key. Exact mirror of
// F#'s `createBox`.
const createBox = (b: Box2i): Box2d => {
  const mn = b.min, mx = b.max;
  return Box2d.fromMinMax(
    new V2d(mn.x - 0.5, mn.y - 0.5),
    new V2d(mx.x + 0.5, mx.y + 0.5),
  );
};

// Subtract `b` from `a` (the placed rect from a free rect) and produce
// up to four residual MaxRects strips. Mirrors F#'s `split`.
const splitRects = (a: Box2i, b: Box2i): Box2i[] => {
  const cands: Box2i[] = [
    mkBox(b.max.x + 1, a.min.y, a.max.x, a.max.y),
    mkBox(a.min.x, a.min.y, b.min.x - 1, a.max.y),
    mkBox(a.min.x, b.max.y + 1, a.max.x, a.max.y),
    mkBox(a.min.x, a.min.y, a.max.x, b.min.y - 1),
  ];
  return cands.filter(isValid2i);
};

// MaxRects union/cleanup. Direct port of F#'s `maxRects`.
const maxRectsCombine = (a: Box2i, b: Box2i): Box2i[] => {
  if (intersectsTouching2i(a, b)) {
    if (contains2i(a, b)) return [a];
    if (contains2i(b, a)) return [b];
    const ax0 = a.min.x, ax1 = a.max.x, ay0 = a.min.y, ay1 = a.max.y;
    const bx0 = b.min.x, bx1 = b.max.x, by0 = b.min.y, by1 = b.max.y;

    // RangeX of a contains RangeX of b: ax0<=bx0 && ax1>=bx1
    const abx = ax0 <= bx0 && ax1 >= bx1;
    const bax = bx0 <= ax0 && bx1 >= ax1;
    const aby = ay0 <= by0 && ay1 >= by1;
    const bay = by0 <= ay0 && by1 >= ay1;

    if (abx && bax) {
      // identical X — Y-union
      return [mkBox(ax0, Math.min(ay0, by0), ax1, Math.max(ay1, by1))];
    }
    if (aby && bay) {
      return [mkBox(Math.min(ax0, bx0), ay0, Math.max(ax1, bx1), ay1)];
    }
    if (abx) {
      return [mkBox(bx0, Math.min(ay0, by0), bx1, Math.max(ay1, by1)), a];
    }
    if (bax) {
      return [mkBox(ax0, Math.min(ay0, by0), ax1, Math.max(ay1, by1)), b];
    }
    if (aby) {
      return [mkBox(Math.min(ax0, bx0), by0, Math.max(ax1, bx1), by1), a];
    }
    if (bay) {
      return [mkBox(Math.min(ax0, bx0), ay0, Math.max(ax1, bx1), ay1), b];
    }
    // Cross case: produce the two orthogonal strips of each rect's
    // overlap with the other, plus both originals; filter invalid.
    const out: Box2i[] = [
      mkBox(Math.min(ax0, bx0), Math.max(ay0, by0), Math.max(ax1, bx1), Math.min(ay1, by1)),
      mkBox(Math.max(ax0, bx0), Math.min(ay0, by0), Math.min(ax1, bx1), Math.max(ay1, by1)),
      a,
      b,
    ];
    return out.filter(isValid2i);
  }
  return [a, b];
};

// Merge a freshly-released rect `n` with already-known adjacent free
// rects, producing the new MaxRects covering the union. Direct port
// of F#'s `merge`.
const merge = (n: Box2i, boxes: Iterable<Box2i>): Box2i[] => {
  const boxArr = [...boxes];
  if (boxArr.length === 0) return [n];

  type FreeT = BvhTree2d<string, Box2i>;
  let free: FreeT = BvhTree2d.empty<string, Box2i>();

  const addOne = (b: Box2i): void => {
    const q = mkBox(b.min.x - 1, b.min.y - 1, b.max.x + 1, b.max.y + 1);
    const touching: Box2i[] = [];
    for (const [, val] of free.getIntersecting(createBox(q))) {
      if (intersectsTouching2i(val, q)) touching.push(val);
    }
    if (touching.length === 0) {
      free = free.add(boxKey(b), createBox(b), b);
      return;
    }
    for (const o of touching) {
      free = free.remove(boxKey(o));
      for (const r of maxRectsCombine(o, b)) {
        // skip if any existing free rect already contains r
        let contained = false;
        for (const [, val] of free.getContaining(createBox(r))) {
          if (contains2i(val, r)) { contained = true; break; }
        }
        if (!contained) {
          // remove any free rect r contains
          const containedKeys: string[] = [];
          for (const [k, val] of free.getContained(createBox(r))) {
            if (contains2i(r, val)) containedKeys.push(k);
            void val;
          }
          for (const k of containedKeys) free = free.remove(k);
          free = free.add(boxKey(r), createBox(r), r);
        }
      }
    }
  };

  addOne(n);
  for (const b of boxArr) addOne(b);

  return [...free.toSeq()].map(([, , v]) => v);
};

// Add a free rect into the tree iff no existing rect already contains
// it. Mirror of F#'s `addFree`.
const addFree = (b: Box2i, free: BvhTree2d<string, Box2i>): BvhTree2d<string, Box2i> => {
  if (!isValid2i(b)) return free;
  let containing = false;
  for (const [, v] of free.getContaining(createBox(b))) {
    if (contains2i(v, b)) { containing = true; break; }
  }
  if (containing) return free;
  return free.add(boxKey(b), createBox(b), b);
};

/**
 * Immutable 2D atlas. Generic over `K` (entry key — anything that's a
 * valid `Map` key works; we recommend strings or numbers).
 */
export class TexturePacking<K> {
  private constructor(
    public readonly size: V2i,
    public readonly allowRotate: boolean,
    private readonly free: BvhTree2d<string, Box2i>,
    /** Used rects keyed on the user's K. */
    public readonly used: ReadonlyMap<K, Box2i>,
  ) {}

  static empty<K>(size: V2i, allowRotate: boolean = true): TexturePacking<K> {
    if (size.x <= 0 || size.y <= 0) {
      return new TexturePacking<K>(new V2i(0, 0), allowRotate, BvhTree2d.empty<string, Box2i>(), new Map());
    }
    const bb = mkBox(0, 0, size.x - 1, size.y - 1);
    const tree = BvhTree2d.empty<string, Box2i>().add(boxKey(bb), createBox(bb), bb);
    return new TexturePacking<K>(size, allowRotate, tree, new Map());
  }

  static create<K>(size: V2i, allowRotate: boolean = true): TexturePacking<K> {
    return TexturePacking.empty<K>(size, allowRotate);
  }

  get count(): number {
    return this.used.size;
  }

  get isEmpty(): boolean {
    return this.used.size === 0;
  }

  /** All free MaxRects. */
  get freeRects(): Iterable<Box2i> {
    return [...this.free.toSeq()].map(([, , v]) => v);
  }

  get occupancy(): number {
    let area = 0;
    for (const [, b] of this.used) {
      const s = sizeOf(b);
      area += s.x * s.y;
    }
    const total = this.size.x * this.size.y;
    return total > 0 ? area / total : 0;
  }

  /**
   * Try to insert one entry. Returns the new packing on success, or
   * `null` if no free rect (with optional rotation) can accommodate
   * `size`.
   *
   * If `id` is already present:
   *   - same size → returns `this` (no-op).
   *   - different size → removes then re-adds.
   */
  tryAdd(id: K, size: V2i): TexturePacking<K> | null;
  /** Atomic batch insert: returns null if any element would not fit. */
  tryAdd(many: Iterable<readonly [K, V2i]>): TexturePacking<K> | null;
  tryAdd(a: K | Iterable<readonly [K, V2i]>, b?: V2i): TexturePacking<K> | null {
    if (b === undefined) {
      let cur: TexturePacking<K> | null = this;
      for (const [id, sz] of a as Iterable<readonly [K, V2i]>) {
        if (cur === null) return null;
        cur = cur.tryAddOne(id, sz);
      }
      return cur;
    }
    return this.tryAddOne(a as K, b);
  }

  private tryAddOne(id: K, size: V2i): TexturePacking<K> | null {
    if (size.x <= 0 || size.y <= 0) return this;
    const ex = this.used.get(id);
    if (ex !== undefined) {
      const s = sizeOf(ex);
      if (s.x === size.x && s.y === size.y) return this;
      const removed = this.remove(id);
      return removed.tryAddOne(id, size);
    }

    // Score each free rect; pick the smallest waste that fits.
    let best: { rect: Box2i; upright: boolean; score: number } | undefined;
    for (const [, , rect] of this.free.toSeq()) {
      const sh = sizeOf(rect);
      const f0 = sh.x >= size.x && sh.y >= size.y;
      const f1 = this.allowRotate ? sh.x >= size.y && sh.y >= size.x : false;
      const w0 = (sh.x - size.x) * (sh.y - size.y);
      const w1 = (sh.x - size.y) * (sh.y - size.x);
      let cand: { score: number; upright: boolean } | undefined;
      if (f0 && f1) {
        if (w0 < w1) cand = { score: w0, upright: true };
        else cand = { score: w1, upright: false };
      } else if (f0) cand = { score: w0, upright: true };
      else if (f1) cand = { score: w1, upright: false };
      if (cand !== undefined && (best === undefined || cand.score < best.score)) {
        best = { rect, upright: cand.upright, score: cand.score };
      }
    }

    if (best === undefined) return null;

    const placedSize = best.upright ? size : new V2i(size.y, size.x);
    const fitting = best.rect;

    // Two slabs left of `fitting` after placement: right strip + top
    // strip. Mirror F# layout.
    const r0 = mkBox(fitting.min.x + placedSize.x, fitting.min.y, fitting.max.x, fitting.max.y);
    const r1 = mkBox(fitting.min.x, fitting.min.y + placedSize.y, fitting.max.x, fitting.max.y);
    const rect = mkBox(
      fitting.min.x,
      fitting.min.y,
      fitting.min.x + placedSize.x - 1,
      fitting.min.y + placedSize.y - 1,
    );

    let free = this.free.remove(boxKey(fitting));
    free = addFree(r0, free);
    free = addFree(r1, free);

    // For every other free rect intersecting `rect`, replace with
    // `splitRects(other, rect)`.
    const intersecting: Array<[string, Box2i]> = [];
    for (const [, , v] of free.toSeq()) {
      if (intersects2i(v, rect)) intersecting.push([boxKey(v), v]);
    }
    for (const [k, other] of intersecting) {
      free = free.remove(k);
      for (const p of splitRects(other, rect)) free = addFree(p, free);
    }

    const newUsed = new Map(this.used);
    newUsed.set(id, rect);
    return new TexturePacking<K>(this.size, this.allowRotate, free, newUsed);
  }

  /** Remove `id` (no-op if absent). */
  remove(id: K): TexturePacking<K> {
    const rect = this.used.get(id);
    if (rect === undefined) return this;
    const newUsed = new Map(this.used);
    newUsed.delete(id);

    const qi = mkBox(rect.min.x - 1, rect.min.y - 1, rect.max.x + 1, rect.max.y + 1);
    const adjacent: Box2i[] = [];
    for (const [, , v] of [...this.free.getIntersecting(createBox(qi))].map(
      ([k, vv]) => [k, vv, vv] as const,
    )) {
      if (
        v.min.x === qi.max.x ||
        v.max.x === qi.min.x ||
        v.min.y === qi.max.y ||
        v.max.y === qi.min.y
      ) {
        adjacent.push(v);
      }
    }
    let free = this.free;
    for (const a of adjacent) free = free.remove(boxKey(a));
    for (const m of merge(rect, adjacent)) free = addFree(m, free);
    return new TexturePacking<K>(this.size, this.allowRotate, free, newUsed);
  }
}

// ─── Module-level helpers (matching the F# `module TexturePacking`) ─

export const empty = <K>(size: V2i): TexturePacking<K> => TexturePacking.empty<K>(size);
export const isEmpty = <K>(p: TexturePacking<K>): boolean => p.isEmpty;
export const count = <K>(p: TexturePacking<K>): number => p.count;
export const occupancy = <K>(p: TexturePacking<K>): number => p.occupancy;
export const tryAdd = <K>(id: K, size: V2i, p: TexturePacking<K>): TexturePacking<K> | null =>
  p.tryAdd(id, size);

/** Sort-then-tryAdd helper. Sorting by descending max-extent matches the F# heuristic. */
export const tryAddMany = <K>(
  elements: Iterable<readonly [K, V2i]>,
  p: TexturePacking<K>,
): TexturePacking<K> | null => {
  const arr = [...elements];
  arr.sort((a, b) => Math.max(b[1].x, b[1].y) - Math.max(a[1].x, a[1].y));
  return p.tryAdd(arr);
};

export const tryOfArray = <K>(
  size: V2i,
  elements: ReadonlyArray<readonly [K, V2i]>,
): TexturePacking<K> | null => tryAddMany(elements, empty<K>(size));

/**
 * Build an optimal-square packing for the given elements, doubling
 * the side until everything fits then binary-searching down. Mirrors
 * F#'s `TexturePacking.square`.
 */
export const square = <K>(elements: ReadonlyArray<readonly [K, V2i]>): TexturePacking<K> => {
  const arr = [...elements];
  arr.sort((a, b) => Math.max(b[1].x, b[1].y) - Math.max(a[1].x, a[1].y));
  if (arr.length === 0) return empty<K>(new V2i(0, 0));
  if (arr.length === 1) {
    const [id, sz] = arr[0]!;
    const s = Math.max(sz.x, sz.y);
    return empty<K>(new V2i(s, s)).tryAdd(id, sz)!;
  }
  let area = 0;
  for (const [, sz] of arr) area += sz.x * sz.y;
  let lo = Math.floor(Math.sqrt(area));
  let hi = nextPow2(lo);
  let best = tryOfArray(new V2i(hi, hi), arr);
  while (best === null) {
    lo = hi;
    hi = hi * 2;
    best = tryOfArray(new V2i(hi, hi), arr);
  }
  let bestPacking = best;
  while (hi > lo + 1) {
    const m = (hi + lo) >> 1;
    const p = tryOfArray(new V2i(m, m), arr);
    if (p !== null) {
      bestPacking = p;
      hi = m;
    } else {
      lo = m;
    }
  }
  return bestPacking;
};

function nextPow2(n: number): number {
  if (n <= 1) return 1;
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

// silence unused-helper warning at export-time tooling
void equals2i;
