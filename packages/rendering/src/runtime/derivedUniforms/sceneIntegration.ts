// §7 v2 — heap-scene integration.
//
// One DerivedUniformsScene per heap scene: owns the rule registry, the (single,
// scene-wide) records buffer, the constituent-slots heap (df32 trafo halves), the
// constituents GPU buffer, and the dispatcher. Per RO: flatten each rule against the
// RO's rule set, register it, resolve every input leaf to a tagged slot handle
// (constituent fwd/bwd / host drawHeader byte), and push one record.
//
// Per frame, inside the heap scene's `evaluateAlways(token, …)` scope:
//   const dirty = scene.pullDirty(token);   // drains changed trafo avals (re-subscribes)
//   scene.uploadDirty(dirty);               // O(changed) value uploads
//   scene.encode(enc);                      // one compute dispatch
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
  /** Single scene-wide records buffer — all buckets share constituents AND the main heap, so per-bucket records would just multiply dispatch overhead. */
  readonly records = new RecordsBuffer();
  readonly dispatcher: DerivedUniformsDispatcher;
  /** Constituents storage GPU buffer (`array<vec2<f32>>` — df32 mat4 halves). */
  constituentsBuf: GPUBuffer;
  private readonly mainHeapRef: { current: GPUBuffer };
  /** Bumped each time a shared GPU buffer is replaced. */
  bufferEpoch = 0;

  constructor(device: GPUDevice, mainHeapBuf: GPUBuffer, opts?: DerivedUniformsSceneOptions) {
    this.device = device;
    const initial = opts?.initialConstituentSlots ?? 64;
    this.constituents = new ConstituentSlots(() => {}, initial);
    this.constituentsBuf = device.createBuffer({
      label: "derivedUniforms.constituents",
      size: initial * DF32_MAT4_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.mainHeapRef = { current: mainHeapBuf };
    this.dispatcher = new DerivedUniformsDispatcher(device, {
      constituentsBuf: () => this.constituentsBuf,
      mainHeapBuf: () => this.mainHeapRef.current,
    });
  }

  rebindMainHeap(mainHeapBuf: GPUBuffer): void {
    this.mainHeapRef.current = mainHeapBuf;
  }

  ensureConstituentsCapacity(requiredBytes: number): void {
    if (requiredBytes <= this.constituentsBuf.size) return;
    let cap = this.constituentsBuf.size;
    while (cap < requiredBytes) cap *= 2;
    this.constituentsBuf.destroy();
    this.constituentsBuf = this.device.createBuffer({
      label: "derivedUniforms.constituents",
      size: cap,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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

  /** Run the uber-kernel. No-op if there are no records. */
  encode(enc: GPUCommandEncoder): boolean {
    return this.dispatcher.encode(enc, this.registry, this.records);
  }

  dispose(): void {
    this.dispatcher.dispose();
    this.constituentsBuf.destroy();
  }
}

// ─── Per-RO registration ──────────────────────────────────────────────

export interface RoDerivedRequest {
  /** Derived rules to install, keyed by the uniform name each produces. */
  readonly rules: ReadonlyMap<string, DerivedRule>;
  /** Trafo avals available on this RO, keyed by the uniform name a rule leaf may reference (e.g. "ModelTrafo"). */
  readonly trafoAvals: ReadonlyMap<string, aval<Trafo3d>>;
  /** drawHeader byte offset (relative to `drawHeaderBaseByte`) of a host-supplied uniform on this RO; undefined if not present. */
  readonly hostUniformOffset: (name: string) => number | undefined;
  /** drawHeader byte offset (relative to `drawHeaderBaseByte`) where each derived output is written; undefined if absent. */
  readonly outputOffset: (name: string) => number | undefined;
  /** Absolute byte offset of this RO's drawHeader within the main heap. */
  readonly drawHeaderBaseByte: number;
}

export interface RoRegistration {
  /** The records-buffer owner key (the RenderObject). */
  readonly owner: object;
  readonly constituentAvals: readonly aval<Trafo3d>[];
  /** Flattened-rule hashes — for `registry.release` on teardown. */
  readonly ruleHashes: readonly string[];
}

export function registerRoDerivations(
  scene: DerivedUniformsScene,
  owner: object,
  req: RoDerivedRequest,
): RoRegistration {
  if (req.rules.size === 0) return { owner, constituentAvals: [], ruleHashes: [] };

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
    if (req.trafoAvals.has(inp.name)) {
      const p = acquireTrafo(inp.name);
      return makeHandle(SlotTag.Constituent, inp.inverse ? p.inv : p.fwd);
    }
    // Anything that isn't a constituent trafo is a plain host uniform — whatever
    // value reached this RO (the sg already collapsed any subtree overrides).
    if (inp.inverse) {
      throw new Error(`derivedUniforms: Inverse of a non-constituent input '${inp.name}' is not supported in v0`);
    }
    const off = req.hostUniformOffset(inp.name);
    if (off !== undefined) return makeHandle(SlotTag.HostHeap, req.drawHeaderBaseByte + off);
    throw new Error(
      `derivedUniforms: rule input '${inp.name}' cannot be resolved on this RO (not a trafo nor a host uniform)`,
    );
  };

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
    scene.records.add(owner, id, makeHandle(SlotTag.HostHeap, req.drawHeaderBaseByte + outOff), inSlots);
  }
  scene.ensureConstituentsCapacity(scene.constituents.slotCount * DF32_MAT4_BYTES);
  return { owner, constituentAvals: acquiredAvals, ruleHashes };
}

export function deregisterRoDerivations(scene: DerivedUniformsScene, reg: RoRegistration): void {
  scene.records.removeAllForOwner(reg.owner);
  for (const h of reg.ruleHashes) scene.registry.release(h);
  for (const av of reg.constituentAvals) scene.constituents.release(av);
}
