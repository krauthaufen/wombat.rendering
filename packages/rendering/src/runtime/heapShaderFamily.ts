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

import type { Effect, CompileOptions } from "@aardworx/wombat.shader";
import { combineHashes } from "@aardworx/wombat.shader/ir";
import {
  compileHeapEffect,
  buildBucketLayout,
  megacallSearchPrelude,
  type HeapEffectSchema,
  type HeapVarying,
  type BucketLayout,
  type DrawHeaderField,
  type FragmentOutputLayout,
} from "./heapEffect.js";
import { compileHeapEffectIR } from "./heapEffectIR.js";

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
export interface BuildShaderFamilyOptions {
  /**
   * Route every texture binding declared by any member effect through
   * the atlas drawHeader path (pageRef / formatBits / origin / size
   * inline fields) instead of standalone texture bindings. Set by
   * heapScene's family-merge path so the union bucket has uniform
   * texture-binding shape regardless of which member effects are
   * textured. Defaults to false (analysis-only callers don't need it).
   */
  readonly atlasizeAllTextures?: boolean;
}

export function buildShaderFamily(
  effects: readonly Effect[],
  fragmentOutputLayout?: FragmentOutputLayout,
  slotAssigner?: FamilySlotAssigner,
  options: BuildShaderFamilyOptions = {},
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
  const drawHeaderUnion = unionDrawHeaders(
    sortedEffects, perEffectSchema, options.atlasizeAllTextures === true,
  );

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
  atlasizeAllTextures: boolean,
): BucketLayout {
  // Build each effect's BucketLayout with default opts. The v1 PoC's
  // family-build call site (a future slice) will pass the real
  // perInstance / atlas opts; here we just want the schema-driven
  // union.
  const perEffectLayout = new Map<Effect, BucketLayout>();
  for (const e of sortedEffects) {
    const schema = perEffectSchema.get(e)!;
    const atlasNames = atlasizeAllTextures
      ? new Set(schema.textures.map(t => t.name))
      : new Set<string>();
    perEffectLayout.set(e, buildBucketLayout(schema, false, {
      atlasTextureBindings: atlasNames,
    }));
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
    // Logical type for the heap-load IR builder. Without this,
    // `compileHeapEffectIR` would fail on the family layout because
    // `loadUniformByRef` is dispatched on `uniformWgslType`.
    uniformWgslType: "u32",
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

// ─── Family WGSL synthesis (slice 3b) ───────────────────────────────
//
// Take an analysis-only `ShaderFamilySchema` and produce a single VS+FS
// pair where the merged `@vertex fn family_vs_main` / `@fragment fn
// family_fs_main` switch on `layoutId` to dispatch to a per-effect
// helper. Per-effect helpers come from `compileHeapEffectIR(...,
// "family-member")` after we rename their entries / structs to make
// them collision-free at module scope.
//
// LIMITATIONS (v1):
//   - Per-effect heap-arena bindings (`@group(0) @binding(0..3)`) and
//     any user uniform bindings must agree across effects (same name ⇒
//     same `@group/@binding` and same WGSL type). Dedup checks this
//     and throws on conflict.
//   - Each varying gets its own dedicated `@location(N) vec4<f32>` slot
//     in the family `FamilyVsOut`, regardless of its real width — sub-
//     vec4 packing across effects is a v2 problem. The varying's
//     `FamilySlot.slot` from the schema is used directly as the
//     family location index.

export interface CompiledShaderFamily {
  readonly vs: string;
  readonly fs: string;
  readonly fragmentOutputLayout?: FragmentOutputLayout | undefined;
}

/**
 * Synthesise the merged VS+FS pair for a `ShaderFamilySchema`.
 *
 * The output is one self-contained WGSL module per stage. The VS
 * carries the megacall search prelude + all bindings (heap arena 0..3
 * plus drawTable / indexStorage / firstDrawInTile at 4..6 — same shape
 * as the standalone path); the FS carries no megacall bindings (it
 * receives the layoutId via `@interpolate(flat)`).
 *
 * Slice 3b is a pure analysis pass — it does NOT wire into heapScene.
 * Slice 3c will adopt this output behind the existing per-effect
 * pipeline.
 */
export function compileShaderFamily(
  family: ShaderFamilySchema,
  fragmentOutputLayout?: FragmentOutputLayout,
): CompiledShaderFamily {
  const opts: CompileOptions = fragmentOutputLayout !== undefined
    ? { target: "wgsl", fragmentOutputLayout }
    : { target: "wgsl" };

  // 1. Per-effect rewrite + compile in family-member mode.
  //    Each effect's entries get renamed to `family_vs_${K}` /
  //    `family_fs_${K}` so they don't collide at module scope when
  //    we concatenate them. The auto-generated VsOut / FsIn / FsOut
  //    structs derive from the entry names, so they're disambiguated
  //    automatically (e.g. `Family_vs_0Output`).
  const perEffectVs: { layoutId: number; vs: string; fsHelperName: string; vsHelperName: string; outStruct: string; fsInStruct: string; fsOutStruct: string }[] = [];
  const perEffectFs: string[] = [];
  for (const e of family.effects) {
    const k = family.layoutIdOf.get(e)!;
    // Composed effects (multiple same-stage Entries that compose into
    // one fused entry) need each source Entry renamed to a UNIQUE
    // target — `renameEntries` rejects collisions, and `composeStages`
    // joins source names into the fused entry's name. After IR compile
    // we read the actual emitted @vertex / @fragment name from the
    // WGSL and use that as the helper name.
    const vsEntries = findStageEntryNames(e, "vertex");
    const fsEntries = findStageEntryNames(e, "fragment");
    const vsPrefix = `family_vs_${k}`;
    const fsPrefix = `family_fs_${k}`;
    const entryMap = new Map<string, string>();
    if (vsEntries.length === 1) {
      entryMap.set(vsEntries[0]!, vsPrefix);
    } else {
      for (let i = 0; i < vsEntries.length; i++) {
        entryMap.set(vsEntries[i]!, `${vsPrefix}_p${i}`);
      }
    }
    if (fsEntries.length === 1) {
      entryMap.set(fsEntries[0]!, fsPrefix);
    } else {
      for (let i = 0; i < fsEntries.length; i++) {
        entryMap.set(fsEntries[i]!, `${fsPrefix}_p${i}`);
      }
    }
    const renamed = e.rename({ entries: entryMap });
    const ir = compileHeapEffectIR(renamed, family.drawHeaderUnion, opts, "family-member");
    const vsHelper = readEntryName(ir.vs, "vertex") ?? vsPrefix;
    const fsHelper = readEntryName(ir.fs, "fragment") ?? fsPrefix;
    // Auto-generated struct names = `${capitalise(entryName)}Output` /
    // `${capitalise(entryName)}Input` (see wombat.shader/wgsl/emit).
    const outStruct = capitalise(vsHelper) + "Output";
    const fsInStruct = capitalise(fsHelper) + "Input";
    const fsOutStruct = capitalise(fsHelper) + "Output";
    perEffectVs.push({ layoutId: k, vs: ir.vs, fsHelperName: fsHelper, vsHelperName: vsHelper, outStruct, fsInStruct, fsOutStruct });
    perEffectFs.push(ir.fs);
  }

  // 2. Strip `@vertex`/`@fragment` decorations from the per-effect
  //    helpers (they're regular fns now) and split each output into a
  //    set of top-level decls.
  const vsParts = perEffectVs.map(p => splitWgslTopLevel(stripStageDecoration(p.vs)));
  const fsParts = perEffectFs.map(s => splitWgslTopLevel(stripStageDecoration(s)));

  // 3. Dedup module-scope decls (storage/uniform bindings, atlas
  //    helpers, structs etc). The per-effect helper functions stay —
  //    they're already uniquely named.
  const vsDecls = dedupModuleDecls(vsParts.flat(), "VS");
  const fsDecls = dedupModuleDecls(fsParts.flat(), "FS");

  // 4. Compute drawHeader stride + layoutId offset for the wrapper.
  const familyStrideU32 = family.drawHeaderUnion.strideU32;
  const layoutIdField = family.drawHeaderUnion.drawHeaderFields.find(f => f.name === "__layoutId");
  if (layoutIdField === undefined) {
    throw new Error("compileShaderFamily: drawHeaderUnion missing '__layoutId' field");
  }
  const layoutIdOffsetU32 = layoutIdField.byteOffset / 4;

  // 5. Synthesize the wrapper VS and FS.
  const vs = synthesizeFamilyVs(family, perEffectVs, vsDecls, familyStrideU32, layoutIdOffsetU32);
  const fs = synthesizeFamilyFs(family, perEffectVs, fsDecls);

  return {
    vs,
    fs,
    fragmentOutputLayout,
  };
}

// ─── Helpers (slice 3b) ─────────────────────────────────────────────

function capitalise(s: string): string {
  return s.length === 0 ? s : s[0]!.toUpperCase() + s.slice(1);
}

/**
 * Find the name of the `@vertex` / `@fragment` fn in a WGSL source.
 * Returns undefined when no such decl is present.
 */
function readEntryName(src: string, stage: "vertex" | "fragment"): string | undefined {
  const re = new RegExp(`@${stage}\\s+fn\\s+(\\w+)\\b`);
  const m = re.exec(src);
  return m === null ? undefined : m[1];
}

/**
 * Find stage entry names for the given stage by scanning the effect's
 * stage templates for `Entry` ValueDefs. Returns a list (typically
 * length 1, but composed effects may have more).
 */
function findStageEntryNames(effect: Effect, stage: "vertex" | "fragment"): string[] {
  const names: string[] = [];
  for (const s of effect.stages) {
    for (const v of s.template.values) {
      if (v.kind !== "Entry") continue;
      if (v.entry.stage !== stage) continue;
      names.push(v.entry.name);
    }
  }
  return names;
}

/**
 * Strip the `@vertex` / `@fragment` decoration from each stage entry
 * function. The decoration is always immediately followed by `fn`
 * (possibly across whitespace/newlines).
 */
function stripStageDecoration(src: string): string {
  return src.replace(/@(?:vertex|fragment|compute)\s+(?=fn\s)/g, "");
}

/**
 * Split a WGSL source into top-level declarations. We walk the source
 * tracking brace depth to find statement boundaries:
 *   - `@…` decorators + `var …;` / `const …;` / `alias …;` (one
 *     statement up to the terminating `;` at depth 0).
 *   - `struct Name { … };` (block, brace-balanced, optional trailing
 *     `;`).
 *   - `fn Name(…) { … }` (block, brace-balanced).
 * Comments / blank lines between decls are dropped — we only retain
 * the declaration text.
 */
function splitWgslTopLevel(src: string): string[] {
  const decls: string[] = [];
  let i = 0;
  const n = src.length;
  while (i < n) {
    // Skip whitespace and line comments.
    while (i < n && /\s/.test(src[i]!)) i++;
    if (i < n && src[i] === "/" && src[i + 1] === "/") {
      while (i < n && src[i] !== "\n") i++;
      continue;
    }
    if (i >= n) break;
    const start = i;
    // Find the end of this declaration. Walk forward, tracking braces.
    // A decl ends either at `;` (depth 0) for non-block decls, or at
    // the matching `}` (with optional trailing `;`) for block decls.
    let depth = 0;
    let sawBrace = false;
    while (i < n) {
      const c = src[i]!;
      if (c === "{") { depth++; sawBrace = true; i++; continue; }
      if (c === "}") {
        depth--;
        i++;
        if (depth === 0 && sawBrace) {
          // Optional trailing semicolon.
          while (i < n && /\s/.test(src[i]!)) i++;
          if (i < n && src[i] === ";") i++;
          break;
        }
        continue;
      }
      if (c === ";" && depth === 0 && !sawBrace) {
        i++;
        break;
      }
      i++;
    }
    const decl = src.slice(start, i).trim();
    if (decl.length > 0) decls.push(decl);
  }
  return decls;
}

/**
 * Classify a top-level decl by kind + identifier (for dedup keying).
 * Returns `{ key, kind }`. Decls with the same `key` must have the
 * same text (else we throw — the family has structurally incompatible
 * per-effect outputs).
 */
function classifyDecl(decl: string): { key: string; kind: "binding" | "struct" | "fn" | "var" | "alias" | "other" } {
  // Binding: `@group(N) @binding(M) var<…> name : type;`
  let m = /^\s*@group\(\s*\d+\s*\)\s*@binding\(\s*\d+\s*\)\s*var(?:<[^>]*>)?\s+(\w+)\s*:/.exec(decl);
  if (m !== null) return { key: `binding:${m[1]}`, kind: "binding" };
  // Struct
  m = /^\s*struct\s+(\w+)\s*\{/.exec(decl);
  if (m !== null) return { key: `struct:${m[1]}`, kind: "struct" };
  // Fn
  m = /^\s*fn\s+(\w+)\s*\(/.exec(decl);
  if (m !== null) return { key: `fn:${m[1]}`, kind: "fn" };
  // var<private>/var
  m = /^\s*var(?:<[^>]*>)?\s+(\w+)\s*:/.exec(decl);
  if (m !== null) return { key: `var:${m[1]}`, kind: "var" };
  // alias
  m = /^\s*alias\s+(\w+)\s*=/.exec(decl);
  if (m !== null) return { key: `alias:${m[1]}`, kind: "alias" };
  return { key: `other:${decl.slice(0, 32)}`, kind: "other" };
}

/**
 * Dedup a flat list of top-level decls. Same-key decls must have the
 * same canonical text (whitespace-normalised); throws on conflict.
 * Per-effect helper fns (named `family_vs_${K}` / `family_fs_${K}`)
 * are unique by construction so they pass through.
 */
function dedupModuleDecls(decls: readonly string[], stageLabel: string): string[] {
  const seen = new Map<string, string>();
  const ordered: string[] = [];
  for (const d of decls) {
    const { key } = classifyDecl(d);
    const norm = d.replace(/\s+/g, " ").trim();
    const existing = seen.get(key);
    if (existing === undefined) {
      seen.set(key, norm);
      ordered.push(d);
    } else if (existing !== norm) {
      throw new Error(
        `compileShaderFamily: ${stageLabel} declaration '${key}' conflicts across effects:\n` +
        `  A: ${existing}\n` +
        `  B: ${norm}\n` +
        `Disambiguate via Effect.rename(...) or align bindings before merging.`,
      );
    }
  }
  return ordered;
}

/**
 * Build the family `FamilyVsOut` struct text. One `vec4<f32>` per
 * varying slot plus a flat-interpolated `layoutIdOut: u32`.
 */
function familyVsOutStruct(slots: number, layoutIdLoc: number): string {
  const lines: string[] = [];
  lines.push("struct FamilyVsOut {");
  lines.push("  @builtin(position) gl_Position: vec4<f32>,");
  for (let i = 0; i < slots; i++) {
    lines.push(`  @location(${i}) Varying${i}: vec4<f32>,`);
  }
  lines.push(`  @interpolate(flat) @location(${layoutIdLoc}) layoutIdOut: u32,`);
  lines.push("};");
  return lines.join("\n");
}

function familyFsInStruct(slots: number, layoutIdLoc: number): string {
  const lines: string[] = [];
  lines.push("struct FamilyFsIn {");
  for (let i = 0; i < slots; i++) {
    lines.push(`  @location(${i}) Varying${i}: vec4<f32>,`);
  }
  lines.push(`  @interpolate(flat) @location(${layoutIdLoc}) layoutIdIn: u32,`);
  lines.push("};");
  return lines.join("\n");
}

/**
 * Component-suffix for an offset+size into a vec4 slot. `(0,3)` →
 * `.xyz`, `(0,4)` → `` (whole vec4), `(1,2)` → `.yz`, etc. v1: each
 * varying takes the prefix `(0, size)` of a dedicated slot, but the
 * helper accepts arbitrary offsets so future packing works.
 */
function vec4Swizzle(offset: number, size: number): string {
  if (offset === 0 && size === 4) return "";
  const comps = ["x", "y", "z", "w"];
  return "." + comps.slice(offset, offset + size).join("");
}

/**
 * Per-varying read/write expression. For vec3/vec2/scalar, the source
 * value is a vec3/vec2/f32; we need to read/write the appropriate
 * `.xyz` / `.xy` / `.x` slice of the slot's `vec4<f32>`.
 */
function slotComponentExpr(slotName: string, offset: number, size: number): string {
  return slotName + vec4Swizzle(offset, size);
}

/**
 * Build a dispatch case for the VS wrapper. Calls `family_vs_${K}` and
 * packs its named-varying outputs into the family's anonymous slots.
 */
function buildVsCase(
  k: number,
  vsHelperName: string,
  outStruct: string,
  effect: Effect,
  family: ShaderFamilySchema,
): string {
  const slotMap = family.perEffectSlotMap.get(effect)!;
  const lines: string[] = [];
  lines.push(`    case ${k}u: {`);
  lines.push(`      let r: ${outStruct} = ${vsHelperName}(heap_drawIdx, instId, vid);`);
  lines.push(`      out.gl_Position = r.gl_Position;`);
  // Walk the effect's varyings (from schema) to know which fields exist
  // on the per-effect struct, then place them into slots by slotMap.
  const schema = family.perEffectSchema.get(effect)!;
  for (const v of schema.varyings) {
    if (v.builtin !== undefined) continue;
    const slot = slotMap.get(v.name);
    if (slot === undefined) continue;
    const slotName = `out.Varying${slot.slot}`;
    const lhs = slotComponentExpr(slotName, slot.offset, slot.size);
    lines.push(`      ${lhs} = r.${v.name};`);
  }
  lines.push(`    }`);
  return lines.join("\n");
}

function buildFsCase(
  k: number,
  fsHelperName: string,
  fsInStruct: string,
  fsOutStruct: string,
  effect: Effect,
  family: ShaderFamilySchema,
): string {
  const slotMap = family.perEffectSlotMap.get(effect)!;
  const schema = family.perEffectSchema.get(effect)!;
  const lines: string[] = [];
  lines.push(`    case ${k}u: {`);
  lines.push(`      var fin: ${fsInStruct};`);
  for (const v of schema.varyings) {
    if (v.builtin !== undefined) continue;
    const slot = slotMap.get(v.name);
    if (slot === undefined) continue;
    const slotName = `in.Varying${slot.slot}`;
    const rhs = slotComponentExpr(slotName, slot.offset, slot.size);
    lines.push(`      fin.${v.name} = ${rhs};`);
  }
  lines.push(`      let r: ${fsOutStruct} = ${fsHelperName}(fin);`);
  // Copy r's fields into the family FsOut. v1 assumes per-effect FS
  // output structs all carry the same fields (driven by
  // `fragmentOutputLayout`); we just map by name. Fragment outputs
  // appear on the schema as `fragmentOutputs`.
  for (const fo of schema.fragmentOutputs) {
    lines.push(`      fout.${fo.name} = r.${fo.name};`);
  }
  lines.push(`    }`);
  return lines.join("\n");
}

function familyFsOutStruct(family: ShaderFamilySchema): string {
  // Union all per-effect fragmentOutputs by name. Mismatched types ⇒
  // throw. v1: simple, no MRT-fan-out.
  type FO = { name: string; location: number; wgslType: string };
  const byName = new Map<string, FO>();
  for (const e of family.effects) {
    const sc = family.perEffectSchema.get(e)!;
    for (const fo of sc.fragmentOutputs) {
      const existing = byName.get(fo.name);
      if (existing === undefined) {
        byName.set(fo.name, { name: fo.name, location: fo.location, wgslType: fo.wgslType });
      } else if (existing.wgslType !== fo.wgslType || existing.location !== fo.location) {
        throw new Error(
          `compileShaderFamily: fragmentOutput '${fo.name}' conflicts across effects: ` +
          `'${existing.wgslType}@${existing.location}' vs '${fo.wgslType}@${fo.location}'`,
        );
      }
    }
  }
  const ordered = [...byName.values()].sort((a, b) => a.location - b.location);
  const lines: string[] = [];
  lines.push("struct FamilyFsOut {");
  for (const fo of ordered) {
    lines.push(`  @location(${fo.location}) ${fo.name}: ${fo.wgslType},`);
  }
  lines.push("};");
  return lines.join("\n");
}

function synthesizeFamilyVs(
  family: ShaderFamilySchema,
  perEffect: readonly { layoutId: number; vsHelperName: string; outStruct: string }[],
  dedupedDecls: readonly string[],
  familyStrideU32: number,
  layoutIdOffsetU32: number,
): string {
  const slots = family.varyingSlots;
  const layoutIdLoc = slots; // last @location after the N vec4 slots

  const lines: string[] = [];
  lines.push("// Family-merged vertex shader (slice 3b synthesis).");
  // Megacall storage-buffer bindings (same shape as standalone path).
  lines.push("@group(0) @binding(4) var<storage, read> drawTable:       array<u32>;");
  lines.push("@group(0) @binding(5) var<storage, read> indexStorage:    array<u32>;");
  lines.push("@group(0) @binding(6) var<storage, read> firstDrawInTile: array<u32>;");
  lines.push("");
  // Deduped module-scope decls (heap arena, user uniforms, per-effect
  // structs, per-effect helper fns).
  for (const d of dedupedDecls) {
    lines.push(d);
    lines.push("");
  }
  // Family VsOut struct.
  lines.push(familyVsOutStruct(slots, layoutIdLoc));
  lines.push("");
  // Wrapper @vertex.
  lines.push("@vertex fn family_vs_main(@builtin(vertex_index) emitIdx: u32) -> FamilyVsOut {");
  // Megacall search prelude — defines heap_drawIdx, instId, vid as
  // locals in this function's scope.
  lines.push(megacallSearchPrelude());
  // layoutId from drawHeader.
  lines.push(`  let layoutId: u32 = headersU32[(heap_drawIdx * ${familyStrideU32}u) + ${layoutIdOffsetU32}u];`);
  lines.push("  var out: FamilyVsOut;");
  // Initialise all slots to zero so the WGSL out-of-init-store rule
  // is satisfied across switch arms that don't write every slot.
  lines.push("  out.gl_Position = vec4<f32>(0.0);");
  for (let i = 0; i < slots; i++) {
    lines.push(`  out.Varying${i} = vec4<f32>(0.0);`);
  }
  lines.push("  out.layoutIdOut = layoutId;");
  lines.push("  switch (layoutId) {");
  for (const p of perEffect) {
    const e = family.effects[p.layoutId]!;
    lines.push(buildVsCase(p.layoutId, p.vsHelperName, p.outStruct, e, family));
  }
  lines.push("    default: { }");
  lines.push("  }");
  lines.push("  return out;");
  lines.push("}");
  return lines.join("\n");
}

function synthesizeFamilyFs(
  family: ShaderFamilySchema,
  perEffect: readonly { layoutId: number; fsHelperName: string; fsInStruct: string; fsOutStruct: string }[],
  dedupedDecls: readonly string[],
): string {
  const slots = family.varyingSlots;
  const layoutIdLoc = slots;
  const lines: string[] = [];
  // Suppress Tint's conservative derivative-uniformity analysis: layoutIdIn is
  // @interpolate(flat) and uniform per primitive in practice, but Tint treats
  // fragment-in params as non-uniform, rejecting dpdx/dpdy reachable from the
  // switch. v1 concession; v2 may hoist derivatives via parameter-threading.
  lines.push("diagnostic(off, derivative_uniformity);");
  lines.push("");
  lines.push("// Family-merged fragment shader (slice 3b synthesis).");
  for (const d of dedupedDecls) {
    lines.push(d);
    lines.push("");
  }
  lines.push(familyFsInStruct(slots, layoutIdLoc));
  lines.push("");
  lines.push(familyFsOutStruct(family));
  lines.push("");
  lines.push("@fragment fn family_fs_main(in: FamilyFsIn) -> FamilyFsOut {");
  lines.push("  var fout: FamilyFsOut;");
  lines.push("  switch (in.layoutIdIn) {");
  for (const p of perEffect) {
    const e = family.effects[p.layoutId]!;
    lines.push(buildFsCase(p.layoutId, p.fsHelperName, p.fsInStruct, p.fsOutStruct, e, family));
  }
  lines.push("    default: { }");
  lines.push("  }");
  lines.push("  return fout;");
  lines.push("}");
  return lines.join("\n");
}
