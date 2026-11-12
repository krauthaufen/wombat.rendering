// stripToList — strip topologies on the heap path.
//
// A heap bucket draws ALL its records with one non-indexed draw: the
// vertex shader maps the running vertex index to (record, local index)
// through the drawTable. For list topologies that is exact — every
// primitive's vertices lie inside one record. For STRIP topologies it
// is wrong: the GPU assembles one strip over the whole draw, so the
// last vertices of one record form primitives with the first vertices
// of the next (many quads drawn as strips came out joined by smeared
// triangles). There is no index buffer bound in that draw, so there is
// no primitive restart to separate them either.
//
// So a strip never reaches a bucket. When a draw is added, its strip
// is rewritten ONCE into the list of the same primitives:
//
//   triangle-strip  v0 v1 v2 v3 …  →  (v0 v1 v2) (v1 v3 v2) (v2 v3 v4) …
//   line-strip      v0 v1 v2 …     →  (v0 v1) (v1 v2) …
//
// (odd triangles are (i, i+2, i+1): same winding as the strip's, and
// the first vertex stays the provoking one). An index equal to
// 0xFFFFFFFF is a primitive restart, as in WebGPU: it ends the strip
// and the next index starts a new one (with even parity).
//
// The rewritten spec has a list topology and an index buffer holding
// the list. The index data is memoized per source (index aval or
// array, or vertex count for non-indexed strips), so N draws of one
// strip mesh still share ONE index allocation, and the specs of one
// PipelineState still share one bucket. List and point draws are not
// touched at all.

import { AVal, type aval } from "@aardworx/wombat.adaptive";
import type { PipelineState, Topology } from "../../core/pipelineState.js";

export type StripKind = "triangle-strip" | "line-strip";

/** Primitive restart value of the heap's (u32) index data. */
export const STRIP_RESTART = 0xFFFFFFFF;

export const stripKindOf = (t: Topology): StripKind | undefined =>
  t === "triangle-strip" || t === "line-strip" ? t : undefined;

const listOf = (t: Topology): Topology =>
  t === "triangle-strip" ? "triangle-list" : t === "line-strip" ? "line-list" : t;

/**
 * The list of the primitives of the strip `src[first .. first+count)`
 * (`src` undefined: the non-indexed strip `first, first+1, …`).
 */
export function expandStrip(
  kind: StripKind,
  src: Uint32Array | undefined,
  first: number,
  count: number,
): Uint32Array {
  const at = src !== undefined
    ? (i: number): number => src[first + i]!
    : (i: number): number => first + i;
  const per = kind === "triangle-strip" ? 3 : 2;
  const need = per - 1;                       // vertices before the first primitive
  // Upper bound: one primitive per vertex after the first `need`.
  const out = new Uint32Array(Math.max(0, count - need) * per);
  let o = 0;
  let run = 0;                                // vertices in the current strip
  for (let i = 0; i < count; i++) {
    const v = at(i);
    if (src !== undefined && v === STRIP_RESTART) { run = 0; continue; }
    run++;
    if (run <= need) continue;
    if (kind === "line-strip") {
      out[o++] = at(i - 1); out[o++] = v;
    } else {
      const a = at(i - 2), b = at(i - 1);
      // (run - 3) is the triangle's number within its strip.
      if (((run - 3) & 1) === 0) { out[o++] = a; out[o++] = b; out[o++] = v; }
      else                       { out[o++] = a; out[o++] = v; out[o++] = b; }
    }
  }
  return o === out.length ? out : out.slice(0, o);
}

// ─── Memoized list data ─────────────────────────────────────────────

const sliceKey = (kind: StripKind, first: number, count: number | undefined): string =>
  `${kind}:${first}:${count ?? "*"}`;

/** Non-indexed strips: one list per (kind, vertex count), shared by every draw. */
const iotaLists = new Map<string, Uint32Array>();
function listOfVertexCount(kind: StripKind, n: number): Uint32Array {
  const key = `${kind}:${n}`;
  let l = iotaLists.get(key);
  if (l === undefined) { l = expandStrip(kind, undefined, 0, n); iotaLists.set(key, l); }
  return l;
}

/** Indexed strips: per index source (aval or array) and slice. */
const indexLists = new WeakMap<object, Map<string, aval<Uint32Array> | Uint32Array>>();
function listOfIndices(
  kind: StripKind,
  src: aval<Uint32Array> | Uint32Array,
  first: number,
  count: number | undefined,
): aval<Uint32Array> | Uint32Array {
  let bySlice = indexLists.get(src);
  if (bySlice === undefined) { bySlice = new Map(); indexLists.set(src, bySlice); }
  const key = sliceKey(kind, first, count);
  let l = bySlice.get(key);
  if (l === undefined) {
    const run = (arr: Uint32Array): Uint32Array =>
      expandStrip(kind, arr, first, count ?? (arr.length - first));
    l = src instanceof Uint32Array ? run(src) : (src as aval<Uint32Array>).map(run);
    bySlice.set(key, l);
  }
  return l;
}

// ─── Memoized list pipeline states ──────────────────────────────────

const LIST_TOPOLOGY: { readonly [K in StripKind]: aval<Topology> } = {
  "triangle-strip": AVal.constant<Topology>("triangle-list"),
  "line-strip":     AVal.constant<Topology>("line-list"),
};
const mappedTopology = new WeakMap<aval<Topology>, aval<Topology>>();
const listStates = new WeakMap<PipelineState, PipelineState>();

/**
 * `ps` with its strip topology replaced by the list one. A constant
 * topology becomes one of two shared constants (bucket keys hash the
 * inner avals by identity); a changing one is mapped, once per aval.
 */
function listStateOf(ps: PipelineState, kind: StripKind): PipelineState {
  let l = listStates.get(ps);
  if (l === undefined) {
    const t = ps.rasterizer.topology;
    let lt: aval<Topology>;
    if (t.isConstant) lt = LIST_TOPOLOGY[kind];
    else {
      let m = mappedTopology.get(t);
      if (m === undefined) { m = t.map(listOf); mappedTopology.set(t, m); }
      lt = m;
    }
    l = { ...ps, rasterizer: { ...ps.rasterizer, topology: lt } };
    listStates.set(ps, l);
  }
  return l;
}

/** The fields of a draw spec this rewrite reads and replaces. */
export interface StripSpecFields {
  readonly pipelineState?: PipelineState;
  readonly modeRules?: { readonly topology?: unknown };
  readonly indices?: aval<Uint32Array> | Uint32Array;
  readonly vertexCount?: number;
  readonly firstIndex?: number;
  readonly indexCount?: number;
}

/**
 * The strip kind `spec` is drawn with right now, or undefined for a
 * list/point draw. A spec whose topology is decided by a mode rule is
 * not rewritten (the rule picks the topology per draw from uniforms;
 * see `HeapDrawSpec.modeRules`).
 */
export function stripKindOfSpec(
  spec: Pick<StripSpecFields, "pipelineState" | "modeRules">,
): StripKind | undefined {
  const ps = spec.pipelineState;
  if (ps === undefined) return undefined;
  if (spec.modeRules?.topology !== undefined) return undefined;
  // Why force here: the heap bakes the topology into the bucket when
  // the draw is added (as `resolvePipelineState` does); a later change
  // is picked up by the mode-key drain, which re-adds the draw.
  return stripKindOf(ps.rasterizer.topology.force(/* allow-force */));
}

/**
 * `spec` as the list draw of the same primitives (see the file
 * header). The result chains onto `spec`, so every other field —
 * and whatever prototype sharing `spec` already has — is kept.
 */
export function stripSpecToList<S extends StripSpecFields>(spec: S, kind: StripKind): S {
  const indices = spec.indices !== undefined
    ? listOfIndices(kind, spec.indices, spec.firstIndex ?? 0, spec.indexCount)
    : listOfVertexCount(kind, spec.vertexCount ?? 0);
  const own = (value: unknown): PropertyDescriptor => ({ value, enumerable: true });
  return Object.create(spec, {
    pipelineState: own(listStateOf(spec.pipelineState!, kind)),
    indices:       own(indices),
    vertexCount:   own(undefined),
    firstIndex:    own(undefined),
    indexCount:    own(undefined),
  }) as S;
}
