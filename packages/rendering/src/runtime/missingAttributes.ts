// The attributes a draw's shader reads that the draw does not have.
//
// The rule: what the LINKED shader reads must be there. An attribute a
// vertex shader only passes on to a varying no fragment output uses is
// dropped by linking and need not be given (FShade-style composition
// declares such pass-throughs all the time). One that is read and missing
// has no data on any path: the heap throws when the draw is added, the
// legacy path when it is prepared, and some GPUs (iOS Safari) draw
// nothing for it. `compileHybridScene` checks each draw once with this,
// before it reaches either path, and leaves a draw that fails out: the
// rest of the scene draws, and the draw is reported once per device.

import type { RenderObject } from "../core/renderObject.js";
import { asAttributeProvider } from "../core/provider.js";
import { compileHeapEffect, type FragmentOutputLayout } from "./heapEffect.js";
import { reportDeviceError } from "./deviceErrors.js";

/** The names of the attributes `ro` has (vertex and instance). */
export function attributeNamesOf(ro: RenderObject): string[] {
  const names = new Set<string>(asAttributeProvider(ro.vertexAttributes).names());
  if (ro.instanceAttributes !== undefined) {
    for (const n of asAttributeProvider(ro.instanceAttributes).names()) names.add(n);
  }
  return [...names];
}

/**
 * The attributes the linked shader of `ro` reads (for `layout`'s
 * fragment outputs) that `ro` lacks; empty when it has them all.
 */
export function missingAttributes(ro: RenderObject, layout?: FragmentOutputLayout): string[] {
  const read = compileHeapEffect(ro.effect, layout).schema.attributes;
  if (read.length === 0) return [];
  const has = new Set(attributeNamesOf(ro));
  const missing: string[] = [];
  for (const a of read) if (!has.has(a.name)) missing.push(a.name);
  return missing;
}

/** The message for a draw missing `missing`. */
export function missingAttributesMessage(ro: RenderObject, missing: readonly string[]): string {
  const has = attributeNamesOf(ro);
  return `${ro.debugName ?? `a draw (effect ${ro.effect.id})`}: its shader reads ` +
    `${missing.length === 1 ? "attribute" : "attributes"} ${missing.map((n) => `"${n}"`).join(", ")}, ` +
    `which it does not have (it has ${has.length === 0 ? "none" : has.join(", ")}); not drawn`;
}

const reported = new WeakMap<GPUDevice, Set<string>>();

/**
 * Whether `ro` can be drawn as far as its attributes go; when not, the
 * reason is reported once per device (console and `onDeviceError`, as a
 * recovered "scene" error: the rest of the scene draws).
 */
export function checkAttributes(device: GPUDevice, ro: RenderObject, layout?: FragmentOutputLayout): boolean {
  let missing: string[];
  try { missing = missingAttributes(ro, layout); }
  catch { return true; } // an effect that does not compile is reported by the path that draws it
  if (missing.length === 0) return true;
  const message = missingAttributesMessage(ro, missing);
  let seen = reported.get(device);
  if (seen === undefined) { seen = new Set(); reported.set(device, seen); }
  if (!seen.has(message)) {
    seen.add(message);
    reportDeviceError(device, { kind: "scene", message, recovered: true });
  }
  return false;
}
