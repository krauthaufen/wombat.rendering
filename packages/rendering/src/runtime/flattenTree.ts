// flattenTree — reactive `RenderTree → aset<RenderObject>`.
//
// Inside a single Render command, draw order is not load-bearing
// (formal contract: an `aset` of objects, not a sequence). The two
// node-kinds `Ordered` and `Unordered` collapse to the same flat
// set — anything ordering-sensitive belongs at the Command level
// (separate Render passes, in the Command list's declared order).
//
// ONE READER PER TREE. The whole tree is flattened by a single
// `FlatTreeReader`: its reactive nodes (`Adaptive`, `OrderedFromList`,
// `UnorderedFromSet`, `Guarded`, `Rows`) are plain records inside it,
// each registered under the adaptive input it reads (the aval, or the
// collection's reader). A marked input names the nodes to update
// (`inputChanged`), so an evaluation touches only what changed; a node
// replaced or removed takes its objects back and drops its inputs.
// Before, every reactive node was an aset of its own (an `ASet.bind`,
// a `toASet` bridge, a union, a fused child-set reader, a guard), each
// with a History, a reader of it and a copy of its objects: at scale
// (thousands of similar subtrees) several adaptive objects and two
// copies of the content per node, for plumbing.
//
// The output is a counting delta: an object reached twice is added
// twice, and stays until both are gone.

import {
  ASet,
  AbstractReader,
  AdaptiveToken as Token,
  HashSetDelta,
  IndexList,
  SetOperation,
  type IAdaptiveObject,
  type IOpReader,
  type IndexListDelta,
  type aset,
  type aval,
} from "@aardworx/wombat.adaptive";
import type { RenderObject } from "../core/renderObject.js";
import type { RenderTree } from "../core/renderTree.js";
import { materializeRow, type RenderRow, type RenderRowSet } from "../core/rowSet.js";
import { heapValuesRaise } from "./heapAdapter.js";

/** Objects added (+) and removed (−) in one evaluation, by identity. */
type Acc = Map<RenderObject, number>;

function put(acc: Acc, ro: RenderObject, n: number): void {
  const c = (acc.get(ro) ?? 0) + n;
  if (c === 0) acc.delete(ro);
  else acc.set(ro, c);
}

type Node =
  /** No reactive node below: its objects. */
  | { k: "static"; readonly objs: readonly RenderObject[] }
  /** Children of which at least one is reactive. */
  | { k: "group"; readonly kids: Node[] }
  | { k: "adaptive"; readonly src: aval<RenderTree>; tree: RenderTree; child: Node }
  | { k: "list"; readonly reader: IOpReader<IndexListDelta<RenderTree>>; kids: IndexList<{ t: RenderTree; n: Node }> }
  | { k: "set"; readonly reader: IOpReader<HashSetDelta<RenderTree>>; readonly kids: Map<RenderTree, { c: number; n: Node }> }
  /** `inner` undefined: left out, waiting for `active` to change. */
  | { k: "guarded"; readonly active: aval<boolean>; readonly child: RenderTree; inner: Node | undefined }
  | { k: "rows"; readonly reader: IOpReader<HashSetDelta<RenderRow>>; readonly set: RenderRowSet; readonly objs: Map<RenderRow, RenderObject> };

const EMPTY: Node = { k: "static", objs: [] };

/** True when `tree` contains no reactive nodes. */
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

function staticObjects(tree: RenderTree, out: RenderObject[]): void {
  switch (tree.kind) {
    case "Leaf": out.push(tree.object); return;
    case "Ordered":
    case "Unordered": for (const c of tree.children) staticObjects(c, out); return;
    default: return;
  }
}

class FlatTreeReader extends AbstractReader<HashSetDelta<RenderObject>> {
  private _root: Node | undefined;
  /** The nodes reading each input (one, or several sharing it). */
  private readonly _inputs = new Map<IAdaptiveObject, Node | Node[]>();
  private _dirty = new Set<IAdaptiveObject>();
  /** While reading a hidden guarded subtree: the inputs it registered,
   *  innermost attempt last (dropped again if the attempt fails). */
  private readonly _attempts: Array<Array<[IAdaptiveObject, Node]>> = [];

  constructor(private readonly _tree: RenderTree) {
    super(HashSetDelta.empty<RenderObject>());
  }

  override inputChanged(_t: unknown, o: IAdaptiveObject): void {
    if (this._inputs.has(o)) this._dirty.add(o);
  }

  override compute(tok: Token): HashSetDelta<RenderObject> {
    const acc: Acc = new Map();
    if (this._root === undefined) {
      this._root = this.build(this._tree, tok, acc);
    } else {
      const dirty = this._dirty;
      this._dirty = new Set();
      for (const o of dirty) {
        const ns = this._inputs.get(o);
        if (ns === undefined) continue; // its node went with an update before
        if (Array.isArray(ns)) {
          for (const n of [...ns]) if (this.reads(o, n)) this.update(n, tok, acc);
        } else this.update(ns, tok, acc);
      }
    }
    if (acc.size === 0) return HashSetDelta.empty<RenderObject>();
    const ops: SetOperation<RenderObject>[] = [];
    for (const [ro, c] of acc) ops.push(new SetOperation(ro, c));
    return HashSetDelta.ofArray(ops);
  }

  // ─── inputs ───────────────────────────────────────────────────────

  /** Whether `n` still reads `o` (an earlier update may have dropped it). */
  private reads(o: IAdaptiveObject, n: Node): boolean {
    const ns = this._inputs.get(o);
    return ns === n || (Array.isArray(ns) && ns.includes(n));
  }

  private reg(o: IAdaptiveObject, n: Node): void {
    const ns = this._inputs.get(o);
    if (ns === undefined) this._inputs.set(o, n);
    else if (Array.isArray(ns)) ns.push(n);
    else this._inputs.set(o, [ns, n]);
    const log = this._attempts[this._attempts.length - 1];
    if (log !== undefined) log.push([o, n]);
  }

  private unreg(o: IAdaptiveObject, n: Node): void {
    const ns = this._inputs.get(o);
    if (ns === undefined) return;
    if (Array.isArray(ns)) {
      const i = ns.indexOf(n);
      if (i >= 0) ns.splice(i, 1);
      if (ns.length === 1) this._inputs.set(o, ns[0]!);
      else if (ns.length > 0) return;
      else { this._inputs.delete(o); o.outputs.remove(this); this._dirty.delete(o); }
      return;
    }
    if (ns !== n) return;
    this._inputs.delete(o);
    o.outputs.remove(this);
    this._dirty.delete(o);
  }

  // ─── building and taking back ─────────────────────────────────────

  private build(t: RenderTree, tok: Token, acc: Acc): Node {
    switch (t.kind) {
      case "Empty":
        return EMPTY;
      case "Leaf":
        put(acc, t.object, 1);
        return { k: "static", objs: [t.object] };
      case "Ordered":
      case "Unordered": {
        if (isStaticTree(t)) {
          const objs: RenderObject[] = [];
          staticObjects(t, objs);
          for (const ro of objs) put(acc, ro, 1);
          return objs.length === 0 ? EMPTY : { k: "static", objs };
        }
        return { k: "group", kids: t.children.map((c) => this.build(c, tok, acc)) };
      }
      case "Adaptive": {
        if (t.tree.isConstant) return this.build(t.tree.getValue(Token.top), tok, acc);
        const tree = t.tree.getValue(tok);
        const n: Node = { k: "adaptive", src: t.tree, tree, child: EMPTY };
        this.reg(t.tree, n);
        n.child = this.build(tree, tok, acc);
        return n;
      }
      case "OrderedFromList": {
        const n: Node = { k: "list", reader: t.children.getReader(), kids: IndexList.empty() };
        const changes = n.reader.getChanges(tok);
        this.reg(n.reader, n);
        this.applyList(n, changes, tok, acc);
        return n;
      }
      case "UnorderedFromSet": {
        const n: Node = { k: "set", reader: t.children.getReader(), kids: new Map() };
        const changes = n.reader.getChanges(tok);
        this.reg(n.reader, n);
        this.applySet(n, changes, tok, acc);
        return n;
      }
      case "Rows": {
        const n: Node = { k: "rows", reader: t.set.rows.getReader(), set: t.set, objs: new Map() };
        const changes = n.reader.getChanges(tok);
        this.reg(n.reader, n);
        this.applyRows(n, changes, acc);
        return n;
      }
      case "Guarded": {
        if (t.active.isConstant) return this.build(t.child, tok, acc);
        const n: Node = { k: "guarded", active: t.active, child: t.child, inner: undefined };
        this.enter(n, tok, acc);
        return n;
      }
    }
  }

  /** Takes `n` in: shown, as it is; hidden, read under the guard (its
   *  structure and its draws' values), left out if that raises and read
   *  again (afresh) when `active` changes. Taken in it is an ordinary
   *  subtree from then on: the guard costs nothing more, a raise
   *  surfaces. */
  private enter(n: Node & { k: "guarded" }, tok: Token, acc: Acc): void {
    // untracked: a subtree taken in keeps no edge on the gate
    if (n.active.getValue(Token.top)) {
      n.inner = this.build(n.child, tok, acc);
      return;
    }
    const log: Array<[IAdaptiveObject, Node]> = [];
    const sub: Acc = new Map();
    this._attempts.push(log);
    let inner: Node | undefined;
    try {
      inner = this.build(n.child, tok, sub);
      for (const [ro, c] of sub) {
        if (c > 0 && heapValuesRaise(ro)) { inner = undefined; break; }
      }
    } catch {
      inner = undefined;
    } finally {
      this._attempts.pop();
    }
    if (inner === undefined) {
      for (const [o, m] of log) this.unreg(o, m);
      n.active.getValue(tok);
      this.reg(n.active, n);
      return;
    }
    const outer = this._attempts[this._attempts.length - 1];
    if (outer !== undefined) outer.push(...log);
    for (const [ro, c] of sub) put(acc, ro, c);
    n.inner = inner;
  }

  /** Takes back what `n` added and drops its inputs. */
  private drop(n: Node, acc: Acc): void {
    switch (n.k) {
      case "static":
        for (const ro of n.objs) put(acc, ro, -1);
        return;
      case "group":
        for (const c of n.kids) this.drop(c, acc);
        return;
      case "adaptive":
        this.unreg(n.src, n);
        this.drop(n.child, acc);
        return;
      case "list":
        this.unreg(n.reader, n);
        for (const { n: c } of n.kids) this.drop(c, acc);
        return;
      case "set":
        this.unreg(n.reader, n);
        for (const { c, n: m } of n.kids.values()) for (let i = 0; i < c; i++) this.drop(m, acc);
        return;
      case "guarded":
        if (n.inner !== undefined) this.drop(n.inner, acc);
        else this.unreg(n.active, n);
        return;
      case "rows":
        this.unreg(n.reader, n);
        for (const ro of n.objs.values()) put(acc, ro, -1);
        return;
    }
  }

  // ─── updates ──────────────────────────────────────────────────────

  private update(n: Node, tok: Token, acc: Acc): void {
    switch (n.k) {
      case "adaptive": {
        const tree = n.src.getValue(tok);
        if (tree === n.tree) return;
        this.drop(n.child, acc);
        n.tree = tree;
        n.child = EMPTY;
        n.child = this.build(tree, tok, acc);
        return;
      }
      case "list": this.applyList(n, n.reader.getChanges(tok), tok, acc); return;
      case "set": this.applySet(n, n.reader.getChanges(tok), tok, acc); return;
      case "rows": this.applyRows(n, n.reader.getChanges(tok), acc); return;
      case "guarded":
        if (n.inner !== undefined) return;
        this.unreg(n.active, n);
        this.enter(n, tok, acc);
        return;
      default:
        return;
    }
  }

  private applyList(n: Node & { k: "list" }, changes: IndexListDelta<RenderTree>, tok: Token, acc: Acc): void {
    for (const [i, op] of changes) {
      const old = n.kids.tryGetByIndex(i);
      if (op.tag === "Remove") {
        if (old !== undefined) { this.drop(old.n, acc); n.kids = n.kids.removeByIndex(i); }
      } else {
        if (old !== undefined) {
          if (Object.is(old.t, op.value)) continue;
          this.drop(old.n, acc);
        }
        n.kids = n.kids.setByIndex(i, { t: op.value, n: this.build(op.value, tok, acc) });
      }
    }
  }

  private applySet(n: Node & { k: "set" }, changes: HashSetDelta<RenderTree>, tok: Token, acc: Acc): void {
    for (const op of changes) {
      const e = n.kids.get(op.value);
      if (op.count > 0) {
        if (e !== undefined) {
          for (let i = 0; i < op.count; i++) this.addAgain(e.n, acc);
          e.c += op.count;
        } else {
          const m = this.build(op.value, tok, acc);
          for (let i = 1; i < op.count; i++) this.addAgain(m, acc);
          n.kids.set(op.value, { c: op.count, n: m });
        }
      } else if (e !== undefined) {
        const k = Math.min(-op.count, e.c);
        e.c -= k;
        if (e.c === 0) { n.kids.delete(op.value); this.drop(e.n, acc); }
        else for (let i = 0; i < k; i++) this.recount(e.n, acc, -1);
      }
    }
  }

  /** A set element counted once more: its objects once more. */
  private addAgain(n: Node, acc: Acc): void { this.recount(n, acc, 1); }

  private recount(n: Node, acc: Acc, s: number): void {
    switch (n.k) {
      case "static": for (const ro of n.objs) put(acc, ro, s); return;
      case "group": for (const c of n.kids) this.recount(c, acc, s); return;
      case "adaptive": this.recount(n.child, acc, s); return;
      case "list": for (const { n: c } of n.kids) this.recount(c, acc, s); return;
      case "set": for (const { c, n: m } of n.kids.values()) for (let i = 0; i < c; i++) this.recount(m, acc, s); return;
      case "guarded": if (n.inner !== undefined) this.recount(n.inner, acc, s); return;
      case "rows": for (const ro of n.objs.values()) put(acc, ro, s); return;
    }
  }

  private applyRows(n: Node & { k: "rows" }, changes: HashSetDelta<RenderRow>, acc: Acc): void {
    for (const op of changes) {
      if (op.count > 0) {
        let ro = n.objs.get(op.value);
        if (ro === undefined) { ro = materializeRow(n.set, op.value); n.objs.set(op.value, ro); }
        put(acc, ro, 1);
      } else {
        const ro = n.objs.get(op.value);
        if (ro !== undefined) { n.objs.delete(op.value); put(acc, ro, -1); }
      }
    }
  }
}

export function flattenRenderTree(tree: RenderTree): aset<RenderObject> {
  switch (tree.kind) {
    case "Empty":
      return ASet.empty<RenderObject>();
    case "Leaf":
      return ASet.single(tree.object);
    default: {
      // Fully-static tree → one constant aset over a plain array.
      if (isStaticTree(tree)) {
        const objs: RenderObject[] = [];
        staticObjects(tree, objs);
        return ASet.ofArray(objs);
      }
      return ASet.ofReader(() => new FlatTreeReader(tree));
    }
  }
}
