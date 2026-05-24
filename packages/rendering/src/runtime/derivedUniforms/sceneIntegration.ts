// §7 v2 — heap-scene integration.
//
// One DerivedUniformsScene per heap scene: owns the rule registry, a per-chunk
// records buffer (§3), the constituent-slots heap (df32 trafo halves, scene-
// wide), the constituents GPU buffer, and the dispatcher.
//
// Multi-chunk (§3): each arena chunk has its own GPUBuffer for drawHeaders.
// Derived-uniform records reference HostHeap slots by absolute byte offset
// within ONE buffer — so a record must be bound to its bucket's chunk at
// dispatch. We partition records by chunkIdx: one RecordsBuffer per chunk,
// one dispatch per chunk binding that chunk's main heap.
//
// Per frame, inside the heap scene's `evaluateAlways(token, …)` scope:
//   const dirty = scene.pullDirty(token);   // drains changed trafo avals (re-subscribes)
//   scene.uploadDirty(dirty);               // O(changed) value uploads
//   scene.encode(enc);                      // one compute dispatch per chunk
// Clean frames: zero CPU work, records buffer static.
// See docs/derived-uniforms-extensible.md.

import type { aval, IAdaptiveObject, AdaptiveToken } from "@aardworx/wombat.adaptive";
import type { Trafo3d } from "@aardworx/wombat.base";
import {
  ConstituentSlots, DF32_MAT4_BYTES,
  type PairedSlots, type SlotIndex,
} from "./slots.js";
import { DerivedUniformRegistry } from "./registry.js";
import { RecordsBuffer, SlotTag, makeHandle } from "./records.js";
import { CHAIN_RULE_ID } from "./codegen.js";
import { DerivedUniformsDispatcher, uploadConstituentsRange } from "./dispatch.js";
import { flatten, type RuleInput } from "./flatten.js";
import { ruleFromIR, type DerivedRule } from "./rule.js";

export interface DerivedUniformsSceneOptions {
  readonly initialConstituentSlots?: number;
}

export class DerivedUniformsScene {
  readonly device: GPUDevice;
  readonly constituents: ConstituentSlots;
  readonly registry = new DerivedUniformRegistry();
  /** Per-chunk records buffers (§3). Constituents + registry are
   *  scene-wide; records partition by chunk so each dispatch binds
   *  the right chunk's main heap. */
  private readonly recordsByChunk = new Map<number, RecordsBuffer>();
  /** CHAIN records partitioned by trie LEVEL (GPU transform propagation). A
   *  trie node at level L computes `link · parentNode.world` — reading its
   *  parent (level L-1) output — so levels must dispatch in ascending order,
   *  BEFORE the §7 per-chunk passes. Scene-wide: they target the shared
   *  constituents buffer (per-node Model), not any chunk's main heap.
   *  Prefix-sharing: sibling chains share trie nodes (see TrafoTree). */
  private readonly chainByLevel = new Map<number, RecordsBuffer>();
  /** The suffix trie that dedups shared ancestor sub-chains across ROs. */
  readonly trafoTree: TrafoTree;
  readonly dispatcher: DerivedUniformsDispatcher;
  /** Constituents storage GPU buffer (`array<vec2<f32>>` — df32 mat4 halves). */
  constituentsBuf: GPUBuffer;
  /** Per-chunk main-heap GPUBuffer getters. Populated as chunks are
   *  registered via `setMainHeapForChunk` from the scene factory. */
  private readonly mainHeapByChunk = new Map<number, () => GPUBuffer>();
  /** Bumped each time a shared GPU buffer is replaced. */
  bufferEpoch = 0;

  constructor(device: GPUDevice, opts?: DerivedUniformsSceneOptions) {
    this.device = device;
    const initial = opts?.initialConstituentSlots ?? 64;
    this.constituents = new ConstituentSlots(() => {}, initial);
    this.constituentsBuf = device.createBuffer({
      label: "derivedUniforms.constituents",
      size: initial * DF32_MAT4_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.dispatcher = new DerivedUniformsDispatcher(device, {
      constituentsBuf: () => this.constituentsBuf,
    });
    this.trafoTree = new TrafoTree(this);
  }

  /** Get-or-create the CHAIN records buffer for trie `level`. */
  chainRecordsFor(level: number): RecordsBuffer {
    let r = this.chainByLevel.get(level);
    if (r === undefined) {
      r = new RecordsBuffer();
      this.chainByLevel.set(level, r);
    }
    return r;
  }

  /** Register (or replace) the GPUBuffer getter for a chunk's main
   *  heap. Called by the scene factory at construction and whenever
   *  a new chunk opens or an existing chunk's buffer reallocates. */
  setMainHeapForChunk(chunkIdx: number, getter: () => GPUBuffer): void {
    this.mainHeapByChunk.set(chunkIdx, getter);
  }

  /** Any registered main-heap getter (the chain pass needs a MainHeap binding
   *  even though its arm never touches it). */
  private firstHeapGetter(): (() => GPUBuffer) | undefined {
    for (const g of this.mainHeapByChunk.values()) return g;
    return undefined;
  }

  /** Get-or-create the records buffer for `chunkIdx`. */
  recordsFor(chunkIdx: number): RecordsBuffer {
    let r = this.recordsByChunk.get(chunkIdx);
    if (r === undefined) {
      r = new RecordsBuffer();
      this.recordsByChunk.set(chunkIdx, r);
    }
    return r;
  }

  /** Total record count summed across every chunk's records buffer.
   *  Used by the scene's stats / diagnostics. */
  get totalRecordCount(): number {
    let n = 0;
    for (const r of this.recordsByChunk.values()) n += r.recordCount;
    return n;
  }

  ensureConstituentsCapacity(requiredBytes: number): void {
    if (requiredBytes <= this.constituentsBuf.size) return;
    let cap = this.constituentsBuf.size;
    while (cap < requiredBytes) cap *= 2;
    this.constituentsBuf.destroy();
    this.constituentsBuf = this.device.createBuffer({
      label: "derivedUniforms.constituents",
      size: cap,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.bufferEpoch++;
    // Fresh buffer is zero-initialised; re-upload everything packed so far.
    const slotCount = this.constituents.slotCount;
    if (slotCount > 0) {
      const mirror = this.constituents.cpuMirror;
      this.device.queue.writeBuffer(
        this.constituentsBuf, 0, mirror.buffer, mirror.byteOffset, slotCount * DF32_MAT4_BYTES,
      );
    }
  }

  /** Routed from the heap scene's `inputChanged(t, o)`. True iff `o` is a tracked trafo aval. */
  routeInputChanged(o: IAdaptiveObject): boolean {
    if (!this.constituents.has(o)) return false;
    this.constituents.markDirty(o);
    return true;
  }

  /** Drain changed trafo avals (re-subscribes via `getValue`). Call inside the scene's evaluate scope. */
  pullDirty(token: AdaptiveToken): ReadonlySet<SlotIndex> {
    return this.constituents.pullDirty(token);
  }

  /** Upload the changed constituent value range. O(changed). */
  uploadDirty(dirty: ReadonlySet<SlotIndex>): void {
    if (dirty.size === 0) return;
    this.ensureConstituentsCapacity(this.constituents.slotCount * DF32_MAT4_BYTES);
    uploadConstituentsRange(this.device, this.constituentsBuf, this.constituents.cpuMirror, dirty);
  }

  /** Run the uber-kernel once per chunk that holds records. No-op
   *  when no chunk has records. Each dispatch binds that chunk's
   *  main heap buffer. */
  encode(enc: GPUCommandEncoder): boolean {
    let any = false;
    // Chain pass — writes per-RO/per-node Model constituents (fwd+inv) into the
    // shared constituents buffer. Dispatch trie LEVELS in ascending order: a
    // level-L node reads its parent's (level L-1) output. Must run BEFORE the
    // §7 passes that read the Models. MainHeap binding is unused here; bind any.
    const levels = [...this.chainByLevel.keys()].sort((a, b) => a - b);
    if (levels.length > 0) {
      const anyHeap = this.firstHeapGetter();
      if (anyHeap === undefined) {
        throw new Error("derivedUniforms.encode: chain records present but no chunk main-heap registered to bind.");
      }
      for (const lvl of levels) {
        const recs = this.chainByLevel.get(lvl)!;
        if (recs.recordCount === 0) continue;
        if (this.dispatcher.encodeChunk(enc, this.registry, recs, anyHeap)) any = true;
      }
    }
    // Phase 2: §7 per-chunk (reads constituents incl. the chain-written Model).
    for (const [chunkIdx, records] of this.recordsByChunk) {
      if (records.recordCount === 0) continue;
      const heapGetter = this.mainHeapByChunk.get(chunkIdx);
      if (heapGetter === undefined) {
        throw new Error(
          `derivedUniforms.encode: no main-heap binding registered for chunkIdx=${chunkIdx}. ` +
          `Call setMainHeapForChunk(idx, getter) when each chunk opens.`,
        );
      }
      if (this.dispatcher.encodeChunk(enc, this.registry, records, heapGetter)) any = true;
    }
    return any;
  }

  dispose(): void {
    this.dispatcher.dispose();
    this.constituentsBuf.destroy();
  }
}

// ─── Trafo suffix-trie (prefix sharing) ───────────────────────────────
//
// Dedups shared ancestor sub-chains across ROs. Each RO's chain is interned
// from the ROOT end, so sibling chains share trie nodes for their common
// ancestor suffix. A trie node owns a per-node Model constituent (fwd+inv) and
// two CHAIN records computing `node.world = link · parent.world` (and the
// reversed inverse) — at the node's depth LEVEL, so a shared ancestor is
// composed ONCE and every descendant reads it. Levels dispatch in ascending
// order (DerivedUniformsScene.encode). Refcounted: a node is freed when no RO
// path references it. See docs/gpu-transform-propagation.md (Phase 2).

export interface TrieNode {
  readonly id: number;
  readonly key: string;
  readonly modelPair: PairedSlots;
  readonly level: number;
  readonly parent: TrieNode | undefined;
  refcount: number;
}

export class TrafoTree {
  private readonly byKey = new Map<string, TrieNode>();
  private nextId = 0;
  constructor(private readonly scene: DerivedUniformsScene) {}

  /** Number of live trie nodes — diagnostics / tests (prefix-sharing check). */
  get nodeCount(): number { return this.byKey.size; }

  /** Intern a chain (array order `[leaf, …, root]`); returns the LEAF trie node
   *  whose Model is the full product. increfs every node on the path so shared
   *  ancestor nodes survive while any RO references them. */
  intern(linkPairs: readonly PairedSlots[]): TrieNode {
    let parent: TrieNode | undefined;
    // Root end (last) → leaf (first), so shared ancestor suffixes share nodes.
    for (let i = linkPairs.length - 1; i >= 0; i--) {
      parent = this.internOne(parent, linkPairs[i]!);
    }
    if (parent === undefined) throw new Error("TrafoTree.intern: empty chain");
    return parent;
  }

  /** Decref every node on the leaf→root path; free those that hit 0. */
  release(leaf: TrieNode): void {
    let n: TrieNode | undefined = leaf;
    while (n !== undefined) {
      const parent: TrieNode | undefined = n.parent;
      if (--n.refcount === 0) {
        this.byKey.delete(n.key);
        this.scene.chainRecordsFor(n.level).removeAllForOwner(n);
        this.scene.constituents.freeOutputPair(n.modelPair);
      }
      n = parent;
    }
  }

  private internOne(parent: TrieNode | undefined, link: PairedSlots): TrieNode {
    const key = `${parent?.id ?? -1}:${link.fwd}`;
    const existing = this.byKey.get(key);
    if (existing !== undefined) { existing.refcount++; return existing; }
    const level = parent === undefined ? 0 : parent.level + 1;
    const modelPair = this.scene.constituents.allocOutputPair();
    const node: TrieNode = { id: this.nextId++, key, modelPair, level, parent, refcount: 1 };
    this.byKey.set(key, node);
    const h = (s: number): number => makeHandle(SlotTag.Constituent, s);
    const recs = this.scene.chainRecordsFor(level);
    // node.world.fwd = link.fwd · parent.world.fwd  (just link at the root end).
    // node.world.inv = parent.world.inv · link.inv  (reversed; just link at root).
    if (parent === undefined) {
      recs.add(node, CHAIN_RULE_ID, h(modelPair.fwd), [1, h(link.fwd)]);
      recs.add(node, CHAIN_RULE_ID, h(modelPair.inv), [1, h(link.inv)]);
    } else {
      recs.add(node, CHAIN_RULE_ID, h(modelPair.fwd), [2, h(link.fwd), h(parent.modelPair.fwd)]);
      recs.add(node, CHAIN_RULE_ID, h(modelPair.inv), [2, h(parent.modelPair.inv), h(link.inv)]);
    }
    return node;
  }
}

// ─── Per-RO registration ──────────────────────────────────────────────

export interface RoDerivedRequest {
  /** Derived rules to install, keyed by the uniform name each produces. */
  readonly rules: ReadonlyMap<string, DerivedRule>;
  /**
   * GPU transform propagation: the SG ancestor trafo chain for THIS RO's
   * `Model` (root→leaf order, constant runs folded), as `aval<Trafo3d>` links.
   * Links are constituents shared by aval identity, so a root trafo shared by
   * N ROs is ONE slot referenced N times (no CPU fan-out). When present, the
   * chain pass computes a per-RO `Model` constituent (fwd from the links, inv
   * from the reversed inverse links) and the §7 rules' `Model` leaf resolves to
   * it — so ModelTrafo / ModelView / ModelViewInv / NormalMatrix / custom rules
   * all derive from the GPU-composed Model, unchanged. See
   * docs/gpu-transform-propagation.md.
   */
  readonly modelChain?: readonly aval<Trafo3d>[];
  /** Trafo avals available on this RO, keyed by the uniform name a rule leaf may reference (e.g. "ModelTrafo"). */
  readonly trafoAvals: ReadonlyMap<string, aval<Trafo3d>>;
  /** drawHeader byte offset (relative to `drawHeaderBaseByte`) of a host-supplied uniform on this RO; undefined if not present. */
  readonly hostUniformOffset: (name: string) => number | undefined;
  /** drawHeader byte offset (relative to `drawHeaderBaseByte`) where each derived output is written; undefined if absent. */
  readonly outputOffset: (name: string) => number | undefined;
  /** Absolute byte offset of this RO's drawHeader within the main heap. */
  readonly drawHeaderBaseByte: number;
  /** §3: which arena chunk this RO lives in. Records are partitioned
   *  by chunk so each dispatch binds the right main-heap buffer. */
  readonly chunkIdx: number;
}

export interface RoRegistration {
  /** The records-buffer owner key (the RenderObject). */
  readonly owner: object;
  readonly constituentAvals: readonly aval<Trafo3d>[];
  /** Flattened-rule hashes — for `registry.release` on teardown. */
  readonly ruleHashes: readonly string[];
  /** Chunk this registration was placed into — used by deregister. */
  readonly chunkIdx: number;
  /** This RO's leaf trie node (transform propagation); its path is decref'd on
   *  deregister. Present iff the request carried a `modelChain`. */
  readonly modelLeaf?: TrieNode;
}

export function registerRoDerivations(
  scene: DerivedUniformsScene,
  owner: object,
  req: RoDerivedRequest,
): RoRegistration {
  if (req.rules.size === 0 && req.modelChain === undefined) {
    return { owner, constituentAvals: [], ruleHashes: [], chunkIdx: req.chunkIdx };
  }
  const records = scene.recordsFor(req.chunkIdx);

  const acquiredAvals: aval<Trafo3d>[] = [];
  const ruleHashes: string[] = [];
  const pairByName = new Map<string, PairedSlots>();
  const acquireTrafo = (name: string): PairedSlots => {
    let p = pairByName.get(name);
    if (p !== undefined) return p;
    const av = req.trafoAvals.get(name);
    if (av === undefined) {
      throw new Error(`derivedUniforms: rule references trafo '${name}' not available on this RO`);
    }
    p = scene.constituents.acquire(av);
    pairByName.set(name, p);
    acquiredAvals.push(av);
    return p;
  };
  const resolve = (inp: RuleInput): number => {
    // A matrix-typed leaf bound as an aval is a `Trafo3d` ⇒ a df32 constituent slot
    // (its `Inverse` reads the stored backward half — free). `Model` may instead
    // be a GPU-composed chain output already in `pairByName` (modelChain case).
    if (inp.type.kind === "Matrix" && (req.trafoAvals.has(inp.name) || pairByName.has(inp.name))) {
      const p = acquireTrafo(inp.name);
      return makeHandle(SlotTag.Constituent, inp.inverse ? p.inv : p.fwd);
    }
    // Anything else is a plain host uniform — whatever value reached this RO (the sg
    // already collapsed any subtree overrides).
    if (inp.inverse) {
      throw new Error(`derivedUniforms: Inverse of a non-constituent input '${inp.name}' is not supported`);
    }
    const off = req.hostUniformOffset(inp.name);
    if (off !== undefined) return makeHandle(SlotTag.HostHeap, req.drawHeaderBaseByte + off);
    throw new Error(
      `derivedUniforms: rule input '${inp.name}' cannot be resolved on this RO ` +
        `(not a Trafo3d binding, and non-trafo host-uniform leaves aren't supported yet)`,
    );
  };

  // GPU transform propagation: compute this RO's `Model` as a per-RO constituent
  // from its ancestor chain, and point §7's `Model` leaf at it. The chain pass
  // (a separate, earlier dispatch) writes the fwd half from the links and the
  // inv half from the reversed inverse links.
  let modelLeaf: TrieNode | undefined;
  if (req.modelChain !== undefined && req.modelChain.length > 0) {
    const linkPairs = req.modelChain.map((av) => {
      acquiredAvals.push(av);
      return scene.constituents.acquire(av);
    });
    // Intern the chain into the suffix trie — siblings share ancestor nodes,
    // each composed once (link · parent.world) at its level. The RO's Model is
    // the leaf node's per-node Model constituent.
    modelLeaf = scene.trafoTree.intern(linkPairs);
    pairByName.set("Model", modelLeaf.modelPair);
  }

  for (const [outName, rule] of req.rules) {
    // `flatten` returns the same Expr object when nothing was substituted (the common case —
    // the standard recipes reference no derived names), so we can reuse `rule` (it already
    // carries a precomputed `.hash`) and skip a redundant IR hash walk per RO.
    const flatIr = flatten(outName, rule.ir, req.rules);
    const flat = flatIr === rule.ir ? rule : ruleFromIR(flatIr, rule.outputType);
    const id = scene.registry.register(flat);
    ruleHashes.push(flat.hash);
    const entry = scene.registry.get(id)!;
    const inSlots = entry.inputs.map(resolve);
    const outOff = req.outputOffset(outName);
    if (outOff === undefined) {
      throw new Error(`derivedUniforms: no drawHeader slot for derived uniform '${outName}' on this RO`);
    }
    records.add(owner, id, makeHandle(SlotTag.HostHeap, req.drawHeaderBaseByte + outOff), inSlots);
  }

  scene.ensureConstituentsCapacity(scene.constituents.slotCount * DF32_MAT4_BYTES);
  return {
    owner, constituentAvals: acquiredAvals, ruleHashes, chunkIdx: req.chunkIdx,
    ...(modelLeaf !== undefined ? { modelLeaf } : {}),
  };
}

export function deregisterRoDerivations(scene: DerivedUniformsScene, reg: RoRegistration): void {
  scene.recordsFor(reg.chunkIdx).removeAllForOwner(reg.owner);
  for (const h of reg.ruleHashes) scene.registry.release(h);
  for (const av of reg.constituentAvals) scene.constituents.release(av);
  if (reg.modelLeaf !== undefined) scene.trafoTree.release(reg.modelLeaf);
}
