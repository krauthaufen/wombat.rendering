// BvhTree2d — generic immutable 2D bounding-volume hierarchy.
//
// Port of `Aardvark.Geometry.BvhTree2d<'K, 'V>` from
// `aardvark.base/src/Aardvark.Geometry/Bvh.fs` (the 2D variant). The
// algorithm and structure mirror the F# original:
//   - Internal nodes are a discriminated union: `Leaf` (overflow
//     count, bounds, key→(bounds,value) map) or `Node` (best cost,
//     count, bounds, left, right).
//   - Splitting picks the dimension+pivot with the lowest SAH-style
//     cost across both axes; rebuilds a leaf into a node when the
//     count exceeds the limit.
//   - Add / remove preserve persistent semantics — every operation
//     returns a fresh tree; existing nodes are reused where possible.
//
// One simplification vs the F# source: `HashMap<K, _>` is replaced by
// `ReadonlyMap<K, _>` (JS Map). The packer never needs structural
// HashMap operations like `union` of distinct subtrees as a free
// HashMap union; we replicate `HashMap.union` by copying values into
// a fresh Map. For the scale we expect (~thousands of free rects per
// page) this is fine.

import { Box2d, V2d } from "@aardworx/wombat.base";

export const SPLIT_LIMIT_DEFAULT = 24;

// Box helpers — wombat's Box2d uses min/max V2d. We define a few small
// utilities that mirror Aardvark's surface (`isValid`, `area`, etc.)
// in terms of the wombat API.
function boxIsValid(b: Box2d): boolean {
  // wombat's Box2d.isValid considers min<=max on both axes.
  return b.isValid();
}
function boxArea(b: Box2d): number {
  if (!boxIsValid(b)) return 0;
  const s = b.size();
  return s.x * s.y;
}
function boxIntersects(a: Box2d, b: Box2d, eps: number = 0): boolean {
  if (!boxIsValid(a) || !boxIsValid(b)) return false;
  const amin = a.min, amax = a.max, bmin = b.min, bmax = b.max;
  return (
    amin.x - eps <= bmax.x &&
    amax.x + eps >= bmin.x &&
    amin.y - eps <= bmax.y &&
    amax.y + eps >= bmin.y
  );
}
function boxContains(outer: Box2d, inner: Box2d): boolean {
  if (!boxIsValid(outer) || !boxIsValid(inner)) return false;
  const omin = outer.min, omax = outer.max, imin = inner.min, imax = inner.max;
  return omin.x <= imin.x && omin.y <= imin.y && omax.x >= imax.x && omax.y >= imax.y;
}
function boxUnion(a: Box2d, b: Box2d): Box2d {
  if (!boxIsValid(a)) return b;
  if (!boxIsValid(b)) return a;
  const amin = a.min, amax = a.max, bmin = b.min, bmax = b.max;
  return Box2d.fromMinMax(
    new V2d(Math.min(amin.x, bmin.x), Math.min(amin.y, bmin.y)),
    new V2d(Math.max(amax.x, bmax.x), Math.max(amax.y, bmax.y)),
  );
}
function boxIntersection(a: Box2d, b: Box2d): Box2d {
  if (!boxIsValid(a) || !boxIsValid(b)) return Box2d.empty;
  const amin = a.min, amax = a.max, bmin = b.min, bmax = b.max;
  const minX = Math.max(amin.x, bmin.x);
  const minY = Math.max(amin.y, bmin.y);
  const maxX = Math.min(amax.x, bmax.x);
  const maxY = Math.min(amax.y, bmax.y);
  if (minX > maxX || minY > maxY) return Box2d.empty;
  return Box2d.fromMinMax(new V2d(minX, minY), new V2d(maxX, maxY));
}

// ─── Internal node representation ──────────────────────────────────
// Leaf:   { kind:"leaf", overflow, bounds, values: Map<K,[Box2d,V]> }
// Node:   { kind:"node", bestCost, count, bounds, left, right }
//
// Tagged shape mirrors the F# DU; `kind` is the discriminant.

interface LeafNode<K, V> {
  readonly kind: "leaf";
  readonly overflow: number;
  readonly bounds: Box2d;
  readonly values: ReadonlyMap<K, readonly [Box2d, V]>;
}
interface InnerNode<K, V> {
  readonly kind: "node";
  readonly bestCost: number;
  readonly count: number;
  readonly bounds: Box2d;
  readonly left: BvhNode<K, V>;
  readonly right: BvhNode<K, V>;
}
type BvhNode<K, V> = LeafNode<K, V> | InnerNode<K, V>;

function nodeCount<K, V>(n: BvhNode<K, V>): number {
  return n.kind === "leaf" ? n.values.size : n.count;
}
function nodeBounds<K, V>(n: BvhNode<K, V>): Box2d {
  return n.bounds;
}

function cost(invArea: number, lBox: Box2d, lCnt: number, rBox: Box2d, rCnt: number): number {
  const inter = boxIntersection(lBox, rBox);
  const iVol = boxIsValid(inter) ? boxArea(inter) * invArea : 0;
  const lVol = boxArea(lBox) * invArea;
  const rVol = boxArea(rBox) * invArea;
  const cnt = lCnt + rCnt;
  const l = lCnt / cnt;
  const r = rCnt / cnt;
  return (1 / cnt + l * lVol + r * rVol + iVol) / 2;
}

function* iterMap<K, V>(m: ReadonlyMap<K, readonly [Box2d, V]>): Iterable<[K, Box2d, V]> {
  for (const [k, [b, v]] of m) yield [k, b, v];
}

function* iterAll<K, V>(n: BvhNode<K, V>): Iterable<[K, Box2d, V]> {
  if (n.kind === "leaf") {
    yield* iterMap(n.values);
  } else {
    yield* iterAll(n.left);
    yield* iterAll(n.right);
  }
}

// Try to find a binary split with cost < 1.0 across both dims.
function trySplit<K, V>(
  invArea: number,
  elements: ReadonlyMap<K, readonly [Box2d, V]>,
):
  | { cost: number; lBox: Box2d; lEls: Map<K, readonly [Box2d, V]>; rBox: Box2d; rEls: Map<K, readonly [Box2d, V]> }
  | undefined {
  if (elements.size <= 1) return undefined;
  const arr: Array<[K, Box2d, V]> = [];
  for (const [k, [b, v]] of elements) arr.push([k, b, v]);

  let bestCost = Number.POSITIVE_INFINITY;
  let bestPerm: number[] | null = null;
  let bestSplit = -1;
  let bestLBox: Box2d = Box2d.empty;
  let bestRBox: Box2d = Box2d.empty;

  const n = arr.length;
  for (let dim = 0; dim < 2; dim++) {
    const perm = arr.map((_, i) => i);
    perm.sort((a, b) => {
      const ca = arr[a]![1].center();
      const cb = arr[b]![1].center();
      return dim === 0 ? ca.x - cb.x : ca.y - cb.y;
    });

    const lBoxes: Box2d[] = new Array(n);
    const rBoxes: Box2d[] = new Array(n);

    let last = arr[perm[0]!]![1];
    lBoxes[0] = last;
    for (let i = 1; i < n; i++) {
      const b = boxUnion(arr[perm[i]!]![1], last);
      lBoxes[i] = b;
      last = b;
    }
    last = Box2d.empty;
    for (let i = n - 1; i >= 0; i--) {
      const b = boxUnion(arr[perm[i]!]![1], last);
      rBoxes[i] = b;
      last = b;
    }

    for (let lCnt = 1; lCnt < n; lCnt++) {
      const rCnt = n - lCnt;
      const lBox = lBoxes[lCnt - 1]!;
      const rBox = rBoxes[lCnt]!;
      const c = cost(invArea, lBox, lCnt, rBox, rCnt);
      if (c < bestCost) {
        bestCost = c;
        bestPerm = perm;
        bestSplit = lCnt;
        bestLBox = lBox;
        bestRBox = rBox;
      }
    }
  }

  if (bestCost < 1.0 && bestPerm !== null) {
    const lEls = new Map<K, readonly [Box2d, V]>();
    const rEls = new Map<K, readonly [Box2d, V]>();
    for (let i = 0; i < bestSplit; i++) {
      const [k, b, v] = arr[bestPerm[i]!]!;
      lEls.set(k, [b, v]);
    }
    for (let i = bestSplit; i < n; i++) {
      const [k, b, v] = arr[bestPerm[i]!]!;
      rEls.set(k, [b, v]);
    }
    return { cost: bestCost, lBox: bestLBox, lEls, rBox: bestRBox, rEls };
  }
  return undefined;
}

function build<K, V>(
  limit: number,
  bounds: Box2d,
  elements: ReadonlyMap<K, readonly [Box2d, V]>,
): BvhNode<K, V> {
  if (elements.size === 0) throw new Error("BvhTree2d.build: empty elements");
  if (elements.size <= limit) {
    return { kind: "leaf", overflow: 0, bounds, values: elements };
  }
  const inv = boxArea(bounds) > 0 ? 1.0 / boxArea(bounds) : 0;
  const s = trySplit(inv, elements);
  if (s !== undefined) {
    const l = build(limit, s.lBox, s.lEls);
    const r = build(limit, s.rBox, s.rEls);
    return { kind: "node", bestCost: s.cost, count: elements.size, bounds, left: l, right: r };
  }
  // bad split — fall back to oversized leaf with overflow = size
  return { kind: "leaf", overflow: elements.size, bounds, values: elements };
}

function add<K, V>(
  limit: number,
  key: K,
  bounds: Box2d,
  value: V,
  node: BvhNode<K, V>,
): BvhNode<K, V> {
  if (node.kind === "leaf") {
    const existing = node.values.get(key);
    if (existing !== undefined) {
      const [oldB] = existing;
      const newValues = new Map(node.values);
      newValues.set(key, [bounds, value]);
      let bb: Box2d;
      if (boxContains(bounds, oldB)) {
        bb = node.bounds;
      } else {
        bb = bounds;
        for (const [, [b]] of newValues) bb = boxUnion(bb, b);
      }
      return { kind: "leaf", overflow: node.overflow, bounds: bb, values: newValues };
    }
    const newValues = new Map(node.values);
    newValues.set(key, [bounds, value]);
    const bb = boxUnion(node.bounds, bounds);
    if (newValues.size >= 2 * node.overflow && newValues.size > limit) {
      return build(limit, bb, newValues);
    }
    return { kind: "leaf", overflow: node.overflow, bounds: bb, values: newValues };
  }

  // Inner: pick child by SAH.
  const nb = boxUnion(node.bounds, bounds);
  const lb = nodeBounds(node.left);
  const rb = nodeBounds(node.right);
  const lc = nodeCount(node.left);
  const rc = nodeCount(node.right);
  const invVol = boxArea(nb) > 0 ? 1.0 / boxArea(nb) : 0;
  const lCost = cost(invVol, boxUnion(lb, bounds), 1 + lc, rb, rc);
  const rCost = cost(invVol, lb, lc, boxUnion(rb, bounds), 1 + rc);

  if (lCost < rCost) {
    if (lCost > 2 * node.bestCost) {
      const all = new Map<K, readonly [Box2d, V]>();
      for (const [k, b, v] of iterAll(node)) all.set(k, [b, v]);
      all.set(key, [bounds, value]);
      return build(limit, nb, all);
    }
    const l = add(limit, key, bounds, value, node.left);
    const lb2 = nodeBounds(l);
    const lc2 = nodeCount(l);
    const cc = cost(invVol, lb2, lc2, rb, rc);
    const nb2 = boxUnion(lb2, rb);
    return {
      kind: "node",
      bestCost: Math.min(node.bestCost, cc),
      count: lc2 + rc,
      bounds: nb2,
      left: l,
      right: node.right,
    };
  } else {
    if (rCost > 2 * node.bestCost) {
      const all = new Map<K, readonly [Box2d, V]>();
      for (const [k, b, v] of iterAll(node)) all.set(k, [b, v]);
      all.set(key, [bounds, value]);
      return build(limit, nb, all);
    }
    const r = add(limit, key, bounds, value, node.right);
    const rb2 = nodeBounds(r);
    const rc2 = nodeCount(r);
    const cc = cost(invVol, lb, lc, rb2, rc2);
    const nb2 = boxUnion(lb, rb2);
    return {
      kind: "node",
      bestCost: Math.min(node.bestCost, cc),
      count: lc + rc2,
      bounds: nb2,
      left: node.left,
      right: r,
    };
  }
}

function tryRemove<K, V>(
  limit: number,
  key: K,
  bounds: Box2d,
  node: BvhNode<K, V>,
): { value: V; rest: BvhNode<K, V> | undefined } | undefined {
  if (node.kind === "leaf") {
    if (!boxIntersects(node.bounds, bounds)) return undefined;
    const found = node.values.get(key);
    if (found === undefined) return undefined;
    const [, v] = found;
    const newValues = new Map(node.values);
    newValues.delete(key);
    if (newValues.size === 0) return { value: v, rest: undefined };
    let bb: Box2d = Box2d.empty;
    for (const [, [b]] of newValues) bb = boxUnion(bb, b);
    return {
      value: v,
      rest: { kind: "leaf", overflow: Math.max(0, node.overflow - 1), bounds: bb, values: newValues },
    };
  }

  if (!boxIntersects(node.bounds, bounds)) return undefined;
  const tl = tryRemove(limit, key, bounds, node.left);
  if (tl !== undefined) {
    if (tl.rest === undefined) return { value: tl.value, rest: node.right };
    const lc = nodeCount(tl.rest);
    const rc = nodeCount(node.right);
    const lb = nodeBounds(tl.rest);
    const rb = nodeBounds(node.right);
    const o = boxUnion(lb, rb);
    const cnt = lc + rc;
    if (cnt <= limit) {
      const merged = new Map<K, readonly [Box2d, V]>();
      for (const [k, b, v] of iterAll(tl.rest)) merged.set(k, [b, v]);
      for (const [k, b, v] of iterAll(node.right)) merged.set(k, [b, v]);
      return {
        value: tl.value,
        rest: { kind: "leaf", overflow: 0, bounds: o, values: merged },
      };
    }
    const c = cost(boxArea(o) > 0 ? 1.0 / boxArea(o) : 0, lb, lc, rb, rc);
    return {
      value: tl.value,
      rest: {
        kind: "node",
        bestCost: Math.min(c, node.bestCost),
        count: cnt,
        bounds: o,
        left: tl.rest,
        right: node.right,
      },
    };
  }
  const tr = tryRemove(limit, key, bounds, node.right);
  if (tr !== undefined) {
    if (tr.rest === undefined) return { value: tr.value, rest: node.left };
    const lc = nodeCount(node.left);
    const rc = nodeCount(tr.rest);
    const lb = nodeBounds(node.left);
    const rb = nodeBounds(tr.rest);
    const o = boxUnion(lb, rb);
    const cnt = lc + rc;
    if (cnt <= limit) {
      const merged = new Map<K, readonly [Box2d, V]>();
      for (const [k, b, v] of iterAll(node.left)) merged.set(k, [b, v]);
      for (const [k, b, v] of iterAll(tr.rest)) merged.set(k, [b, v]);
      return {
        value: tr.value,
        rest: { kind: "leaf", overflow: 0, bounds: o, values: merged },
      };
    }
    const c = cost(boxArea(o) > 0 ? 1.0 / boxArea(o) : 0, lb, lc, rb, rc);
    return {
      value: tr.value,
      rest: {
        kind: "node",
        bestCost: Math.min(c, node.bestCost),
        count: cnt,
        bounds: o,
        left: node.left,
        right: tr.rest,
      },
    };
  }
  return undefined;
}

function getIntersecting<K, V>(query: Box2d, node: BvhNode<K, V>, out: Map<K, readonly [Box2d, V]>): void {
  if (!boxIntersects(node.bounds, query, 1e-20)) return;
  if (node.kind === "leaf") {
    for (const [k, [b, v]] of node.values) {
      if (boxIntersects(b, query)) out.set(k, [b, v]);
    }
  } else {
    getIntersecting(query, node.left, out);
    getIntersecting(query, node.right, out);
  }
}

function getContaining<K, V>(query: Box2d, node: BvhNode<K, V>, out: Map<K, readonly [Box2d, V]>): void {
  if (!boxContains(node.bounds, query)) return;
  if (node.kind === "leaf") {
    for (const [k, [b, v]] of node.values) {
      if (boxContains(b, query)) out.set(k, [b, v]);
    }
  } else {
    getContaining(query, node.left, out);
    getContaining(query, node.right, out);
  }
}

function getContained<K, V>(query: Box2d, node: BvhNode<K, V>, out: Map<K, readonly [Box2d, V]>): void {
  if (boxContains(query, node.bounds)) {
    for (const [k, b, v] of iterAll(node)) out.set(k, [b, v]);
    return;
  }
  if (!boxIntersects(query, node.bounds)) return;
  if (node.kind === "leaf") {
    for (const [k, [b, v]] of node.values) {
      if (boxContains(query, b)) out.set(k, [b, v]);
    }
  } else {
    getContained(query, node.left, out);
    getContained(query, node.right, out);
  }
}

/**
 * Immutable / persistent 2D bounding-volume hierarchy.
 *
 * Generic over K (entry key — must be usable as a Map key, i.e. either
 * a primitive or a stable reference) and V (per-entry payload). Every
 * mutation returns a new tree that shares unchanged subtrees with the
 * original where possible.
 */
export class BvhTree2d<K, V> {
  private constructor(
    private readonly limit: number,
    private readonly root: BvhNode<K, V> | undefined,
    private readonly keyBounds: ReadonlyMap<K, Box2d>,
  ) {}

  static empty<K, V>(splitLimit: number = SPLIT_LIMIT_DEFAULT): BvhTree2d<K, V> {
    return new BvhTree2d<K, V>(splitLimit, undefined, new Map());
  }

  /** Same as `empty` — kept as an alias for closer F# parity. */
  static Empty<K, V>(splitLimit: number = SPLIT_LIMIT_DEFAULT): BvhTree2d<K, V> {
    return BvhTree2d.empty(splitLimit);
  }

  get count(): number {
    return this.root === undefined ? 0 : nodeCount(this.root);
  }

  get isEmpty(): boolean {
    return this.root === undefined;
  }

  add(key: K, bounds: Box2d, value: V): BvhTree2d<K, V> {
    if (!boxIsValid(bounds)) return this;
    const newKB = new Map(this.keyBounds);
    newKB.set(key, bounds);
    let newRoot: BvhNode<K, V>;
    if (this.root === undefined) {
      const m = new Map<K, readonly [Box2d, V]>();
      m.set(key, [bounds, value]);
      newRoot = { kind: "leaf", overflow: 0, bounds, values: m };
    } else {
      newRoot = add(this.limit, key, bounds, value, this.root);
    }
    return new BvhTree2d(this.limit, newRoot, newKB);
  }

  remove(key: K): BvhTree2d<K, V> {
    const r = this.tryRemove(key);
    return r === undefined ? this : r[1];
  }

  tryRemove(key: K): [V, BvhTree2d<K, V>] | undefined {
    const b = this.keyBounds.get(key);
    if (b === undefined || this.root === undefined) return undefined;
    const newKB = new Map(this.keyBounds);
    newKB.delete(key);
    const r = tryRemove(this.limit, key, b, this.root);
    if (r === undefined) return undefined;
    return [r.value, new BvhTree2d(this.limit, r.rest, newKB)];
  }

  *getIntersecting(query: Box2d): Iterable<[K, V]> {
    if (this.root === undefined) return;
    const out = new Map<K, readonly [Box2d, V]>();
    getIntersecting(query, this.root, out);
    for (const [k, [, v]] of out) yield [k, v];
  }

  *getContaining(query: Box2d): Iterable<[K, V]> {
    if (this.root === undefined) return;
    const out = new Map<K, readonly [Box2d, V]>();
    getContaining(query, this.root, out);
    for (const [k, [, v]] of out) yield [k, v];
  }

  *getContained(query: Box2d): Iterable<[K, V]> {
    if (this.root === undefined) return;
    const out = new Map<K, readonly [Box2d, V]>();
    getContained(query, this.root, out);
    for (const [k, [, v]] of out) yield [k, v];
  }

  /** Iterate every entry as `[K, Box2d, V]`. */
  *toSeq(): Iterable<[K, Box2d, V]> {
    if (this.root === undefined) return;
    yield* iterAll(this.root);
  }

  *keys(): Iterable<K> {
    if (this.root === undefined) return;
    for (const [k] of iterAll(this.root)) yield k;
  }

  /** Lookup the bounds we stored for `key`, or `undefined`. */
  getBounds(key: K): Box2d | undefined {
    return this.keyBounds.get(key);
  }
}
