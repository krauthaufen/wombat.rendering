// Canonical per-axis enum tables + value mapping for derived modes.
//
// Used by the legacy CPU-eval path (derivedModes/cpuEval). heapScene's
// GPU-routing path keeps a parallel inline copy of this table + helpers
// (heapScene.ts ~AXIS_ENUM_TABLE / resolveAxisValue / resolveDeclaredU32);
// they must stay in sync — fold both onto this module when convenient.
// The index↔value mapping is the WebGPU enum order; a rule body returns a
// u32 index and the renderer maps it to the concrete mode value here.

import type { aval, AdaptiveToken } from "@aardworx/wombat.adaptive";
import type { DerivedModeRule, ModeAxis } from "./rule.js";

/** u32 index → mode value, per axis. `blend` has no canonical table —
 *  blend rules return object literals directly (see resolveAxisValue). */
export const AXIS_ENUM_TABLE: { readonly [A in ModeAxis]: ReadonlyArray<unknown> } = {
  cull:            ["none", "front", "back"],
  frontFace:       ["ccw", "cw"],
  topology:        ["point-list", "line-list", "line-strip", "triangle-list", "triangle-strip"],
  depthCompare:    ["never", "less", "equal", "less-equal", "greater", "not-equal", "greater-equal", "always"],
  depthWrite:      [false, true],
  alphaToCoverage: [false, true],
  blend:           [],
};

/**
 * Map a rule's resolved output to its axis-shape value:
 *  - object → already the full axis value (e.g. a blend AttachmentBlend).
 *  - number → enum-table lookup for enum axes; pass-through otherwise.
 */
export function resolveAxisValue<A extends ModeAxis>(rule: DerivedModeRule<A>, value: unknown): unknown {
  if (typeof value === "object" && value !== null) return value;
  if (typeof value === "number") {
    const table = AXIS_ENUM_TABLE[rule.axis];
    if (table.length > 0) {
      const v = table[value];
      if (v !== undefined) return v;
    }
    return value;
  }
  return value;
}

/**
 * Resolve a rule's `declared` SG-context value to its u32 index:
 *  - omitted → 0 (axis canonical zero).
 *  - number  → passed through.
 *  - aval / string / boolean → looked up in the canonical enum table.
 */
export function declaredToU32<A extends ModeAxis>(rule: DerivedModeRule<A>, tok: AdaptiveToken): number {
  const d = rule.declared;
  if (d === undefined) return 0;
  const v = (typeof d === "object" && d !== null && "getValue" in (d as object))
    ? (d as aval<unknown>).getValue(tok)
    : d;
  if (typeof v === "number") return v >>> 0;
  const table = AXIS_ENUM_TABLE[rule.axis];
  if (table.length === 0) {
    throw new Error(`derivedModes: axis '${rule.axis}' has no canonical enum table; pass \`declared\` as a u32 index or omit it`);
  }
  const i = table.indexOf(v);
  if (i < 0) {
    throw new Error(`derivedModes: declared value '${String(v)}' for axis '${rule.axis}' not in canonical enum table ${JSON.stringify(table)}`);
  }
  return i;
}
