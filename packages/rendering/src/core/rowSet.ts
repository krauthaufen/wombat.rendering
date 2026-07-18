// RenderRowSet — the row store's render-side contract (instance-tables
// M2 proper). A collection of same-shaped draws is ONE shared template
// `RenderObject` plus an `aset` of thin rows; per row only the fields
// that actually differ are retained (~a hundred bytes), instead of a
// full RenderObject + provider graph + spec per element.
//
// `materializeRow` produces a REAL RenderObject via prototype
// inheritance — every consumer that predates the row store (legacy
// ScenePass, flatten fallback, eligibility) keeps working unchanged on
// materialized rows, so correctness never depends on the fast path.

import type { aval, aset } from "@aardworx/wombat.adaptive";
import type { Trafo3d } from "@aardworx/wombat.base";
import type { DrawCall } from "./drawCall.js";
import type { IAttributeProvider, IUniformProvider } from "./provider.js";
import type { RenderObject } from "./renderObject.js";

/** The per-row delta over a `RenderRowSet.template`. */
export interface RenderRow {
  readonly uniforms: IUniformProvider;
  readonly drawCall: aval<DrawCall>;
  readonly instanceAttributes?: IAttributeProvider;
  readonly pickId?: number;
  readonly modelChain?: readonly aval<Trafo3d>[];
  readonly active?: aval<boolean>;
}

export interface RenderRowSet {
  /**
   * Shared fields — a RenderObject whose per-row fields are treated as
   * DEFAULTS (rows override via prototype inheritance). The producer
   * seeds it from the first lowered row; `template.heapAsserted` is
   * expected for the direct heap path (unasserted sets fall back to
   * materialized per-row consumption).
   *
   * Mutable ON PURPOSE: the tree node must exist before the first
   * child arrives through the reactive children set.
   */
  template: RenderObject | undefined;
  readonly rows: aset<RenderRow>;
}

const materialized = new WeakMap<RenderRow, RenderObject>();

/** Row → RenderObject with the set's template as prototype. Cached per
 *  row so aset deltas / eligibility caches see a stable identity. */
export function materializeRow(set: RenderRowSet, row: RenderRow): RenderObject {
  let ro = materialized.get(row);
  if (ro === undefined) {
    if (set.template === undefined) {
      throw new Error("materializeRow: RenderRowSet.template not seeded yet");
    }
    const o = Object.create(set.template) as Record<string, unknown>;
    o["uniforms"] = row.uniforms;
    o["drawCall"] = row.drawCall;
    if (row.instanceAttributes !== undefined) o["instanceAttributes"] = row.instanceAttributes;
    if (row.pickId !== undefined) o["pickId"] = row.pickId;
    if (row.modelChain !== undefined) o["modelChain"] = row.modelChain;
    if (row.active !== undefined) o["active"] = row.active;
    ro = o as unknown as RenderObject;
    materialized.set(row, ro);
  }
  return ro;
}
