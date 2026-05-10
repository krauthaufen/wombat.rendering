// §7 → heap scene integration helper.
//
// Subscription model. Standard adaptive idiom: calling `av.getValue(t)`
// inside an `evaluateAlways(t, ...)` scope (re-)adds the evaluating
// object to `av.outputs`. We don't manage subscription manually.
//
// Per frame, the heap scene's `update()` runs `evaluateAlways` once on
// the parent scene-object (as it already does). Inside that scope:
//   1. `scene.constituents.pullDirty(token)` drains the dirty queue.
//      For each dirty aval, it calls `getValue(token)` — which both
//      gives us the new value AND re-establishes our subscription on
//      that specific aval. O(changed). Avals that didn't change keep
//      us in their outputs from earlier evaluations.
//   2. `scene.uploadDirty(dirtySlots)` — O(changed) value uploads +
//      O(changed) bitmask flips.
//   3. Per bucket: `bucket.dispatcher.encode(enc)`. The kernel
//      early-returns on records whose inputs are all clean.
//
// On clean frames pullDirty/uploadDirty are no-ops. The records
// buffer is static. CPU steady-state cost: zero.

import type { aval, IAdaptiveObject } from "@aardworx/wombat.adaptive";
import type { Trafo3d } from "@aardworx/wombat.base";
import {
  ConstituentSlots,
  DF32_MAT4_BYTES,
  type DerivationRecord, type PairedSlots, type SlotIndex,
  type SubscribeFn,
} from "./slots.js";
import {
  RecipeId, recipeIdByName, recipeInputs, recipeInputCount,
  DERIVED_UNIFORM_NAMES,
  type ConstituentRef,
} from "./recipes.js";
import {
  DerivedUniformsDispatcher, DerivedUniformsPipeline,
  uploadConstituentsRange,
} from "./dispatch.js";

/** Scene-wide §7 state. One per heap scene. Owns the single records
 *  buffer + dispatcher — all buckets share constituents, dirty mask,
 *  AND main heap (arena.attrs.buffer), so per-bucket records would
 *  just multiply dispatch overhead for no reason. */
export class DerivedUniformsScene {
  readonly device: GPUDevice;
  readonly constituents: ConstituentSlots;
  readonly pipeline:     DerivedUniformsPipeline;
  readonly dispatcher:   DerivedUniformsDispatcher;
  /** Constituents storage GPU buffer (df32 mat4 array). */
  constituentsBuf: GPUBuffer;
  /** Caller-owned main heap (arena.attrs.buffer). Updated via
   *  `rebindMainHeap` on arena resize. */
  private mainHeapRef: { current: GPUBuffer };
  /** Bumped each time a shared GPU buffer is replaced. */
  bufferEpoch = 0;

  constructor(
    device:      GPUDevice,
    mainHeapBuf: GPUBuffer,
    opts?:       { initialConstituentSlots?: number; initialRecordCapacity?: number },
  ) {
    this.device = device;
    const initial = opts?.initialConstituentSlots ?? 64;
    const subscribe: SubscribeFn = () => {};
    this.constituents = new ConstituentSlots(subscribe, initial);
    this.pipeline     = new DerivedUniformsPipeline(device);
    this.constituentsBuf = device.createBuffer({
      label: "derivedUniforms.constituents",
      size:  initial * DF32_MAT4_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.mainHeapRef = { current: mainHeapBuf };
    this.dispatcher = new DerivedUniformsDispatcher(
      device,
      this.pipeline,
      {
        constituentsBuf: () => this.constituentsBuf,
        mainHeapBuf:     () => this.mainHeapRef.current,
      },
      opts?.initialRecordCapacity ?? 64,
    );
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
      size:  cap,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.bufferEpoch++;
    // The fresh buffer is zero-initialized; the previous contents are
    // gone. Re-upload everything we've ever packed (slot 0..slotCount).
    // Without this, slots that were last written before the grow read
    // zero on the GPU — every df_mul on them produces NaN.
    const slotCount = this.constituents.slotCount;
    if (slotCount > 0) {
      const mirror = this.constituents.cpuMirror;
      this.device.queue.writeBuffer(
        this.constituentsBuf, 0,
        mirror.buffer, mirror.byteOffset,
        slotCount * DF32_MAT4_BYTES,
      );
    }
  }

  /** Routed from heap scene's `inputChanged(t, o)`. Returns true iff
   *  `o` matched a known constituent aval. */
  routeInputChanged(o: IAdaptiveObject): boolean {
    if (!this.constituents.has(o)) return false;
    this.constituents.markDirty(o);
    return true;
  }

  /** Apply the dirty set: upload changed constituent values. O(changed). */
  uploadDirty(dirty: ReadonlySet<SlotIndex>): void {
    if (dirty.size === 0) return;
    this.ensureConstituentsCapacity(this.constituents.slotCount * DF32_MAT4_BYTES);
    uploadConstituentsRange(
      this.device, this.constituentsBuf, this.constituents.cpuMirror, dirty,
    );
  }

  dispose(): void {
    this.dispatcher.dispose();
    this.constituentsBuf.destroy();
  }
}

// ─── Per-RO registration ──────────────────────────────────────────────

export interface RoTrafoInputs {
  readonly modelTrafo?: aval<Trafo3d> | undefined;
  readonly viewTrafo?:  aval<Trafo3d> | undefined;
  readonly projTrafo?:  aval<Trafo3d> | undefined;
}

export interface RoDerivedRequest {
  readonly trafos:           RoTrafoInputs;
  readonly requiredNames:    readonly string[];
  readonly byteOffsetByName: ReadonlyMap<string, number>;
  readonly drawHeaderBaseByte: number;
}

export interface RoRegistration {
  /** Per-record handles (RecordsBuffer ids) — passed back to deregister. */
  readonly recordIds: readonly number[];
  readonly constituentAvals: readonly aval<Trafo3d>[];
}

/** Register all derived-uniform records for one RO. */
export function registerRoDerivations(
  scene:  DerivedUniformsScene,
  req:    RoDerivedRequest,
): RoRegistration {
  if (req.requiredNames.length === 0) {
    return { recordIds: [], constituentAvals: [] };
  }

  const acquiredAvals: aval<Trafo3d>[] = [];
  const pairFor = new Map<"Model" | "View" | "Proj", PairedSlots>();
  const TRAFO_FIELD: Record<"Model" | "View" | "Proj", keyof RoTrafoInputs> = {
    Model: "modelTrafo", View: "viewTrafo", Proj: "projTrafo",
  };
  const acquirePair = (key: "Model" | "View" | "Proj"): PairedSlots => {
    let p = pairFor.get(key);
    if (p !== undefined) return p;
    const av = req.trafos[TRAFO_FIELD[key]];
    if (av === undefined) {
      throw new Error(
        `derivedUniforms: RO requires '${key}' trafo but trafos.${TRAFO_FIELD[key]} is undefined`,
      );
    }
    p = scene.constituents.acquire(av);
    pairFor.set(key, p);
    acquiredAvals.push(av);
    return p;
  };

  const resolveConstituent = (ref: ConstituentRef): SlotIndex => {
    switch (ref) {
      case "Model.fwd": return acquirePair("Model").fwd;
      case "Model.bwd": return acquirePair("Model").inv;
      case "View.fwd":  return acquirePair("View" ).fwd;
      case "View.bwd":  return acquirePair("View" ).inv;
      case "Proj.fwd":  return acquirePair("Proj" ).fwd;
      case "Proj.bwd":  return acquirePair("Proj" ).inv;
    }
  };

  const recordIds: number[] = [];
  for (const name of req.requiredNames) {
    const id = recipeIdByName(name);
    if (id === undefined) {
      throw new Error(`derivedUniforms: unknown derived name '${name}'`);
    }
    const off = req.byteOffsetByName.get(name);
    if (off === undefined) {
      throw new Error(
        `derivedUniforms: '${name}' has no byte offset in drawHeader for this RO`,
      );
    }
    const refs = recipeInputs(id);
    const slots = refs.map(resolveConstituent);
    const inCount = recipeInputCount(id);
    const rec: DerivationRecord = {
      recipe:  id as number,
      in0:     slots[0]!,
      in1:     inCount >= 2 ? slots[1]! : (0 as SlotIndex),
      in2:     inCount >= 3 ? slots[2]! : (0 as SlotIndex),
      outByte: req.drawHeaderBaseByte + off,
    };
    recordIds.push(scene.dispatcher.records.add(rec));
  }

  return { recordIds, constituentAvals: acquiredAvals };
}

export function deregisterRoDerivations(
  scene: DerivedUniformsScene,
  reg:   RoRegistration,
): void {
  for (const id of reg.recordIds) scene.dispatcher.records.remove(id);
  for (const av of reg.constituentAvals) scene.constituents.release(av);
}

export function isDerivedUniformName(name: string): boolean {
  return DERIVED_UNIFORM_NAMES.has(name);
}

// Re-export RecipeId for callers (e.g. heapScene's drawHeader walk).
export { RecipeId };
