// Test helper: build a real wombat.shader `Effect` from TS source.
//
// `parseShader` produces one IR Module containing every requested
// entry; `stage(module)` wraps it as a single-stage Effect whose
// `.compile({target})` runs the real optimiser + WGSL emitter
// pipeline. No mocks — anything we put on `RenderObject.effect`
// goes through exactly the same path real users hit when their
// `vertex(...) / fragment(...)` markers are processed by the
// Vite plugin.
//
// `extraValues` lets the source reference identifiers that the
// frontend wouldn't otherwise resolve — `Uniform` blocks,
// `Sampler` / `StorageBuffer` declarations. The `externalTypes`
// table is derived automatically from those so type dispatch
// works (e.g. `texture(samp, uv)` resolves to the right intrinsic).

import { parseShader, type EntryRequest } from "@aardworx/wombat.shader-frontend";
import { stage, type Effect } from "@aardworx/wombat.shader-runtime";
import type { Module, Type, ValueDef } from "@aardworx/wombat.shader-ir";

export function makeEffect(
  source: string,
  entries: readonly EntryRequest[],
  opts: { file?: string; extraValues?: readonly ValueDef[] } = {},
): Effect {
  const externalTypes = new Map<string, Type>();
  for (const v of opts.extraValues ?? []) {
    if (v.kind === "Uniform") for (const u of v.uniforms) externalTypes.set(u.name, u.type);
    else if (v.kind === "Sampler" || v.kind === "StorageBuffer")
      externalTypes.set(v.name, v.kind === "StorageBuffer" ? v.layout : v.type);
    else if (v.kind === "Constant") externalTypes.set(v.name, v.varType);
  }
  const parseInput: Parameters<typeof parseShader>[0] = {
    source, entries, externalTypes,
    ...(opts.file !== undefined ? { file: opts.file } : {}),
  };
  const parsed = parseShader(parseInput);
  const merged: Module = opts.extraValues && opts.extraValues.length > 0
    ? { ...parsed, values: [...opts.extraValues, ...parsed.values] }
    : parsed;
  return stage(merged);
}
