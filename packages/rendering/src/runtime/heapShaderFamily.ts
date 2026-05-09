// heapShaderFamily — analysis pass for §6 family-merge.
//
// Given a set of `Effect`s, build a `ShaderFamilySchema` that describes
// the family-wide layout the rest of §6 will plumb into the heap path:
//
//   - deterministic `id` over the sorted-by-effect.id member set
//   - `layoutIdOf` mapping each effect to a 0..N-1 layoutId
//   - per-effect varying slot maps (anonymous Varying0..N packing)
//   - unified `drawHeaderUnion` (union of every effect's BucketLayout)
//   - per-effect `HeapEffectSchema` for downstream use
//
// This file is a pure analysis pass. It does NOT touch heapScene, does
// NOT compile WGSL, does NOT rewrite IR. It only computes the
// load-bearing descriptor that subsequent slices of §6 will consume.

import type { Effect } from "@aardworx/wombat.shader";
import { combineHashes } from "@aardworx/wombat.shader/ir";
import {
  compileHeapEffect,
  buildBucketLayout,
  type HeapEffectSchema,
  type HeapVarying,
  type BucketLayout,
  type DrawHeaderField,
  type FragmentOutputLayout,
} from "./heapEffect.js";

/**
 * One varying's place in the family-wide anonymous-slot layout. Each
 * slot is a `vec4<f32>` interpolant; varyings smaller than vec4 share
 * a slot via `(offset, size)`. `offset + size <= 4`.
 */
export interface FamilySlot {
  /** Index into `Varying0..N` slots (0-based). */
  readonly slot: number;
  /** 0..3 — float offset within the slot's vec4. */
  readonly offset: number;
  /** 1..4 — float count consumed. */
  readonly size: number;
}

/**
 * User-supplied slot assigner. Receives the effect, the varying name,
 * and its WGSL type string (e.g. `"vec3<f32>"`); returns a `FamilySlot`
 * if the caller wants to override the default greedy packer for this
 * varying. Returning `undefined` falls back to the default.
 */
export type FamilySlotAssigner = (
  effect: Effect,
  name: string,
  wgslType: string,
) => FamilySlot | undefined;

export interface ShaderFamilySchema {
  /** Stable hash over the sorted-by-id effect set. Bucket-key axis. */
  readonly id: string;
  /** Member effects in deterministic order (sorted by `effect.id`). */
  readonly effects: readonly Effect[];
  /** layoutId assignment, 0..N-1, parallel to `effects`. */
  readonly layoutIdOf: ReadonlyMap<Effect, number>;
  /** Total vec4<f32> slots in family VsOut (max across effects). */
  readonly varyingSlots: number;
  /** Per-effect slot positions for each named, non-builtin varying. */
  readonly perEffectSlotMap: ReadonlyMap<Effect, ReadonlyMap<string, FamilySlot>>;
  /**
   * Unified BucketLayout: union of all effects' drawHeader fields,
   * plus a `layoutId: u32` field appended at the end. Same-name fields
   * collapse to one entry across effects (after type-equality check).
   */
  readonly drawHeaderUnion: BucketLayout;
  /** Per-effect schema as compiled (diagnostics + downstream consumers). */
  readonly perEffectSchema: ReadonlyMap<Effect, HeapEffectSchema>;
}

/**
 * Build a family schema from a (possibly unordered) set of effects.
 *
 * Throws on:
 *   - same-name varying with mismatched WGSL type across effects
 *   - same-name drawHeader field with mismatched WGSL type across effects
 *
 * The error message identifies both effects + the offending field so
 * the user can disambiguate via `effect.rename({varyings: ...})`.
 */
export function buildShaderFamily(
  effects: readonly Effect[],
  fragmentOutputLayout?: FragmentOutputLayout,
  slotAssigner?: FamilySlotAssigner,
): ShaderFamilySchema {
  // 1. Deterministic ordering. Sort by `effect.id` (lexicographic).
  //    `Array.prototype.sort` is mutating, so copy first.
  const sortedEffects = [...effects].sort((a, b) =>
    a.id < b.id ? -1 : a.id > b.id ? 1 : 0,
  );

  const layoutIdOf = new Map<Effect, number>();
  for (let i = 0; i < sortedEffects.length; i++) {
    layoutIdOf.set(sortedEffects[i]!, i);
  }

  // 2. Compile every effect's schema. Don't run `compileHeapEffectIR`
  //    here — that does heap-shape rewriting; we just want the schema.
  const perEffectSchema = new Map<Effect, HeapEffectSchema>();
  for (const e of sortedEffects) {
    const compiled = compileHeapEffect(e, fragmentOutputLayout);
    perEffectSchema.set(e, compiled.schema);
  }

  // 3. Family hash: combine the (sorted) member ids deterministically.
  const id = combineHashes(...sortedEffects.map(e => e.id));

  // 4. Cross-effect varying type-equality check. Same-name varying ⇒
  //    same WGSL type (and same builtin-or-not flavor).
  checkVaryingTypeAgreement(sortedEffects, perEffectSchema);

  // 5. Per-effect varying slot packing.
  const perEffectSlotMap = new Map<Effect, ReadonlyMap<string, FamilySlot>>();
  let varyingSlots = 0;
  for (const e of sortedEffects) {
    const schema = perEffectSchema.get(e)!;
    const slotMap = packVaryings(e, schema.varyings, slotAssigner);
    perEffectSlotMap.set(e, slotMap);
    let maxSlot = 0;
    for (const s of slotMap.values()) {
      if (s.slot + 1 > maxSlot) maxSlot = s.slot + 1;
    }
    if (maxSlot > varyingSlots) varyingSlots = maxSlot;
  }

  // 6. drawHeader union across all effects, then append layoutId.
  const drawHeaderUnion = unionDrawHeaders(sortedEffects, perEffectSchema);

  return {
    id,
    effects: sortedEffects,
    layoutIdOf,
    varyingSlots,
    perEffectSlotMap,
    drawHeaderUnion,
    perEffectSchema,
  };
}

// ─── Varying packing ────────────────────────────────────────────────

/** Float-count of a WGSL varying type (vec4=4, vec3=3, vec2=2, scalar=1). */
function wgslFloatCount(wgslType: string): number {
  // Cover the realistic varying-typed surface: f32, vec2/3/4<f32>,
  // and the scalar-int variants (u32, i32 — used for flat-interpolated
  // atlas threading and discrete payloads). Matrices are not legal as
  // an inter-stage type per WGSL, so we don't need to handle them.
  if (wgslType === "f32" || wgslType === "u32" || wgslType === "i32" || wgslType === "bool") return 1;
  const m = /^vec([234])<\s*[fui]32\s*>$/.exec(wgslType);
  if (m !== null) return Number(m[1]);
  throw new Error(
    `buildShaderFamily: cannot determine slot size for varying type '${wgslType}'`,
  );
}

/**
 * Greedy packing: walk the effect's non-builtin varyings in order;
 * place each in the first slot that has enough remaining float-room.
 * Open a fresh slot when the current one can't fit. Per-effect — each
 * effect's slot space starts at slot 0; the family's `varyingSlots` is
 * the max across effects.
 */
function packVaryings(
  effect: Effect,
  varyings: readonly HeapVarying[],
  slotAssigner: FamilySlotAssigner | undefined,
): ReadonlyMap<string, FamilySlot> {
  const result = new Map<string, FamilySlot>();
  const slotFill: number[] = []; // floats consumed in each slot so far

  for (const v of varyings) {
    if (v.builtin !== undefined) continue; // gl_Position etc. — not in family VsOut
    const size = wgslFloatCount(v.wgslType);

    // Caller override path.
    if (slotAssigner !== undefined) {
      const override = slotAssigner(effect, v.name, v.wgslType);
      if (override !== undefined) {
        if (override.size !== size) {
          throw new Error(
            `buildShaderFamily: slotAssigner returned size ${override.size} for varying ` +
            `'${v.name}: ${v.wgslType}' (expected ${size})`,
          );
        }
        if (override.offset < 0 || override.offset + override.size > 4) {
          throw new Error(
            `buildShaderFamily: slotAssigner returned out-of-range slot ` +
            `(offset=${override.offset}, size=${override.size}) for varying '${v.name}'`,
          );
        }
        result.set(v.name, override);
        // Track the slot fill for callers that mix override + default.
        while (slotFill.length <= override.slot) slotFill.push(0);
        const used = slotFill[override.slot]!;
        const top = override.offset + override.size;
        if (top > used) slotFill[override.slot] = top;
        continue;
      }
    }

    // Default greedy: first-fit by remaining room.
    let placed = false;
    for (let i = 0; i < slotFill.length; i++) {
      const used = slotFill[i]!;
      if (used + size <= 4) {
        result.set(v.name, { slot: i, offset: used, size });
        slotFill[i] = used + size;
        placed = true;
        break;
      }
    }
    if (!placed) {
      const slot = slotFill.length;
      slotFill.push(size);
      result.set(v.name, { slot, offset: 0, size });
    }
  }
  return result;
}

// ─── DrawHeader union ───────────────────────────────────────────────

/**
 * Union the per-effect BucketLayouts into a single drawHeader. Same-
 * name field across effects ⇒ same WGSL type (else throws). The
 * result has every unique field laid out in encounter order, then
 * `layoutId: u32` appended at a fresh aligned offset.
 *
 * `perInstanceUniforms` and `perInstanceAttributes` unions across all
 * effects (a uniform/attribute that's per-instance in any effect is
 * per-instance in the family). Texture/sampler bindings union by name
 * with the same type-agreement check; conflicting types throw.
 */
function unionDrawHeaders(
  sortedEffects: readonly Effect[],
  perEffectSchema: ReadonlyMap<Effect, HeapEffectSchema>,
): BucketLayout {
  // Build each effect's BucketLayout with default opts. The v1 PoC's
  // family-build call site (a future slice) will pass the real
  // perInstance / atlas opts; here we just want the schema-driven
  // union.
  const perEffectLayout = new Map<Effect, BucketLayout>();
  for (const e of sortedEffects) {
    const schema = perEffectSchema.get(e)!;
    perEffectLayout.set(e, buildBucketLayout(schema, false, {}));
  }

  // Union drawHeaderFields by name. First-seen layout wins for the
  // field's metadata; we just verify type-agreement on subsequent
  // appearances. We DON'T trust the per-effect byteOffsets — those are
  // recomputed for the unified layout below.
  const fieldByName = new Map<string, { field: DrawHeaderField; ownerId: string }>();
  for (const e of sortedEffects) {
    const layout = perEffectLayout.get(e)!;
    for (const f of layout.drawHeaderFields) {
      const existing = fieldByName.get(f.name);
      if (existing === undefined) {
        fieldByName.set(f.name, { field: f, ownerId: e.id });
      } else {
        // Compare the underlying logical type, not the ref-slot type
        // (which is always "u32" for uniform-ref / attribute-ref).
        const existingLogical = logicalFieldType(existing.field);
        const incomingLogical = logicalFieldType(f);
        if (existingLogical !== incomingLogical) {
          throw new Error(
            `buildShaderFamily: drawHeader field '${f.name}' has conflicting WGSL type ` +
            `across effects: '${existingLogical}' (effect ${existing.ownerId}) vs ` +
            `'${incomingLogical}' (effect ${e.id}). Disambiguate via effect.rename(...).`,
          );
        }
        if (existing.field.kind !== f.kind) {
          throw new Error(
            `buildShaderFamily: drawHeader field '${f.name}' has conflicting kind across ` +
            `effects: '${existing.field.kind}' (effect ${existing.ownerId}) vs ` +
            `'${f.kind}' (effect ${e.id}).`,
          );
        }
      }
    }
  }

  // Re-lay out the unioned fields with fresh byte offsets. Order is
  // deterministic: encounter order over the sorted effects + their
  // own field order.
  const ordered: DrawHeaderField[] = [];
  const seen = new Set<string>();
  for (const e of sortedEffects) {
    const layout = perEffectLayout.get(e)!;
    for (const f of layout.drawHeaderFields) {
      if (seen.has(f.name)) continue;
      seen.add(f.name);
      ordered.push(f);
    }
  }

  const fields: DrawHeaderField[] = [];
  let off = 0;
  for (const f of ordered) {
    off = roundUp(off, Math.min(f.byteSize, 16));
    const next: DrawHeaderField = {
      ...f,
      byteOffset: off,
    };
    fields.push(next);
    off += f.byteSize;
  }
  // Append the family-wide layoutId selector. u32 → 4-byte align.
  off = roundUp(off, 4);
  const layoutIdField: DrawHeaderField = {
    name: "__layoutId",
    wgslName: "layoutId",
    wgslType: "u32",
    byteOffset: off,
    byteSize: 4,
    kind: "uniform-ref",
  };
  fields.push(layoutIdField);
  off += 4;

  const drawHeaderBytes = roundUp(off, 16);
  const strideU32 = drawHeaderBytes / 4;

  // Union perInstance sets and texture/sampler bindings.
  const perInstanceUniforms = new Set<string>();
  const perInstanceAttributes = new Set<string>();
  for (const e of sortedEffects) {
    const layout = perEffectLayout.get(e)!;
    for (const n of layout.perInstanceUniforms) perInstanceUniforms.add(n);
    for (const n of layout.perInstanceAttributes) perInstanceAttributes.add(n);
  }

  type TexBinding = { name: string; wgslType: string; binding: number };
  const textureByName = new Map<string, TexBinding>();
  const samplerByName = new Map<string, TexBinding>();
  const atlasUnion = new Set<string>();
  for (const e of sortedEffects) {
    const layout = perEffectLayout.get(e)!;
    for (const t of layout.textureBindings) {
      const existing = textureByName.get(t.name);
      if (existing === undefined) textureByName.set(t.name, { ...t });
      else if (existing.wgslType !== t.wgslType) {
        throw new Error(
          `buildShaderFamily: texture binding '${t.name}' has conflicting WGSL type ` +
          `across effects: '${existing.wgslType}' vs '${t.wgslType}'.`,
        );
      }
    }
    for (const s of layout.samplerBindings) {
      const existing = samplerByName.get(s.name);
      if (existing === undefined) samplerByName.set(s.name, { ...s });
      else if (existing.wgslType !== s.wgslType) {
        throw new Error(
          `buildShaderFamily: sampler binding '${s.name}' has conflicting WGSL type ` +
          `across effects: '${existing.wgslType}' vs '${s.wgslType}'.`,
        );
      }
    }
    for (const n of layout.atlasTextureBindings) atlasUnion.add(n);
  }

  // Renumber bindings deterministically — first the unioned textures,
  // then samplers, in encounter order over the sorted effects.
  const orderedTextures: TexBinding[] = [];
  const seenTex = new Set<string>();
  for (const e of sortedEffects) {
    const layout = perEffectLayout.get(e)!;
    for (const t of layout.textureBindings) {
      if (seenTex.has(t.name)) continue;
      seenTex.add(t.name);
      orderedTextures.push(textureByName.get(t.name)!);
    }
  }
  const orderedSamplers: TexBinding[] = [];
  const seenSmp = new Set<string>();
  for (const e of sortedEffects) {
    const layout = perEffectLayout.get(e)!;
    for (const s of layout.samplerBindings) {
      if (seenSmp.has(s.name)) continue;
      seenSmp.add(s.name);
      orderedSamplers.push(samplerByName.get(s.name)!);
    }
  }
  // We don't reassign binding numbers here — the v1 PoC slice that
  // wires this into a real WGSL prelude will do that. For the
  // descriptor-only purpose of this slice, the per-effect-allocated
  // numbers are retained as-is on each entry.

  return {
    drawHeaderFields: fields,
    drawHeaderBytes,
    preludeWgsl: "", // Family-level prelude is a future slice; not produced here.
    strideU32,
    perInstanceUniforms,
    perInstanceAttributes,
    textureBindings: orderedTextures,
    samplerBindings: orderedSamplers,
    atlasTextureBindings: atlasUnion,
  };
}

function checkVaryingTypeAgreement(
  sortedEffects: readonly Effect[],
  perEffectSchema: ReadonlyMap<Effect, HeapEffectSchema>,
): void {
  const seen = new Map<string, { wgslType: string; isBuiltin: boolean; ownerId: string }>();
  for (const e of sortedEffects) {
    const schema = perEffectSchema.get(e)!;
    for (const v of schema.varyings) {
      const existing = seen.get(v.name);
      const isBuiltin = v.builtin !== undefined;
      if (existing === undefined) {
        seen.set(v.name, { wgslType: v.wgslType, isBuiltin, ownerId: e.id });
      } else {
        if (existing.wgslType !== v.wgslType || existing.isBuiltin !== isBuiltin) {
          throw new Error(
            `buildShaderFamily: varying '${v.name}' has conflicting type across effects: ` +
            `'${existing.wgslType}'${existing.isBuiltin ? " (builtin)" : ""} ` +
            `(effect ${existing.ownerId}) vs '${v.wgslType}'${isBuiltin ? " (builtin)" : ""} ` +
            `(effect ${e.id}). Disambiguate via effect.rename({varyings: ...}).`,
          );
        }
      }
    }
  }
}

function roundUp(value: number, mult: number): number {
  return Math.ceil(value / mult) * mult;
}

/**
 * Logical (semantic) WGSL type of a drawHeader field. For uniform-ref
 * and attribute-ref, the field's `wgslType` is always `"u32"` (the
 * pool ref); the real semantic type lives in
 * `uniformWgslType` / `attributeWgslType`. For texture-ref and any
 * other inline kinds, fall back to `wgslType`.
 */
function logicalFieldType(f: DrawHeaderField): string {
  if (f.kind === "uniform-ref"   && f.uniformWgslType   !== undefined) return f.uniformWgslType;
  if (f.kind === "attribute-ref" && f.attributeWgslType !== undefined) return f.attributeWgslType;
  return f.wgslType;
}
