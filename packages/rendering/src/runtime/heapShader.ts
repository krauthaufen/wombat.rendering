// heapShader — bridge from wombat.shader effects to heap-scene
// fragment WGSL.
//
// `buildHeapScene` requires a fragment-stage WGSL string per group
// key (see runtime/heapScene.ts). Writing those strings by hand
// works but is the wrong long-term shape — users want to write the
// FS in the wombat.shader DSL and get the same effect-composition,
// type-checking, and DCE the per-RO path enjoys.
//
// `compileHeapFragment(effect)` runs wombat.shader's emit on a
// fragment Effect and rewrites the resulting WGSL to plug into the
// heap-scene's `VsOut` shape:
//
//   ┌─ heap-scene's VsOut          ┌─ DSL fragment input must use
//   │ worldPos: vec3<f32>          │ matching field names:
//   │ normal:   vec3<f32>          │   worldPos, normal, color,
//   │ color:    vec4<f32>          │   lightLoc
//   │ lightLoc: vec3<f32>          │
//   └─                             └─
//
// The DSL effect's fragment OUTPUT must be a single field; its name
// is irrelevant (the adapter rewraps to bare `@location(0) vec4<f32>`).
// `uniform.X` references inside the fragment body are redirected to
// `in.X` — the contract is that any uniform the FS reads must have
// been thread through the VS into the corresponding VsOut field.
//
// What this DOES NOT do: rewrite the vertex stage. The heap-scene's
// VS is fixed (storage-buffer uniform reads, vertex pulling from
// position/normal slabs); user effects only contribute the fragment.

import type { Effect } from "@aardworx/wombat.shader";

/**
 * Compile a wombat.shader fragment `Effect` into a WGSL string
 * usable as a `buildHeapScene` `fragmentShaders` value.
 *
 * The effect's fragment input field names must match the
 * heap-scene `VsOut` fields: `worldPos`, `normal`, `color`,
 * `lightLoc`. The fragment output must be a single `vec4<f32>`
 * field (any name).
 */
export function compileHeapFragment(effect: Effect): string {
  const compiled = effect.compile({ target: "wgsl" });
  const fs = compiled.stages.find(s => s.stage === "fragment");
  if (fs === undefined) {
    throw new Error("compileHeapFragment: effect has no fragment stage");
  }
  return rewriteFragment(fs.source);
}

function rewriteFragment(wgsl: string): string {
  let s = wgsl;

  // 1. Drop the auto-emitted UBO struct + binding. Heap-scene FS
  //    doesn't bind a UBO at all (uniforms ride on VsOut from VS).
  s = s.replace(/struct\s+_UB_uniform\s*\{[\s\S]*?\};?\s*/g, "");
  s = s.replace(/@group\(\d+\)\s*@binding\(\d+\)\s*var<uniform>\s+_w_uniform\s*:\s*_UB_uniform\s*;?\s*/g, "");

  // 2. Redirect uniform reads to inter-stage. _w_uniform.LightLocation
  //    → in.lightLoc.
  s = s.replace(/_w_uniform\.(\w+)/g, (_m, name: string) => "in." + uniformToVsOut(name));

  // 3. Locate the @fragment function signature; capture the input
  //    struct name + output struct name from the signature itself.
  //    Two emit shapes seen in practice:
  //      - inline-marker: `wombat_fragment_<hash>` + `Wombat_fragment_<hash>{Input,Output}`
  //      - direct stage(): `<fnName>` + `<FnName>{Input,Output}`
  //    Detecting from the signature works for both.
  const fnSigMatch = s.match(
    /@fragment\s+fn\s+(\w+)\s*\(\s*(\w+)\s*:\s*(\w+)\s*\)\s*->\s*(\w+)\s*\{/,
  );
  if (fnSigMatch === null) {
    throw new Error("compileHeapFragment: emitted WGSL has no @fragment function header");
  }
  const fnHeader        = fnSigMatch[0];
  const paramName       = fnSigMatch[2]!;
  const inputStructName = fnSigMatch[3]!;
  const outputStructName = fnSigMatch[4]!;

  // 4. Drop the input struct.
  const inputStructRe = new RegExp(`struct\\s+${escapeRegExp(inputStructName)}\\s*\\{[\\s\\S]*?\\};?`);
  s = s.replace(inputStructRe, "");

  // 5. Capture + drop the output struct, recording its (single) field name.
  const outputStructRe = new RegExp(`struct\\s+${escapeRegExp(outputStructName)}\\s*\\{([\\s\\S]*?)\\};?`);
  const outputMatch = s.match(outputStructRe);
  if (outputMatch === null) {
    throw new Error(`compileHeapFragment: cannot locate output struct '${outputStructName}'`);
  }
  const outputBody = outputMatch[1]!;
  const fieldRe = /@location\(0\)\s+(\w+)\s*:\s*vec4<f32>\s*,?/g;
  const fieldMatches = [...outputBody.matchAll(fieldRe)];
  if (fieldMatches.length !== 1) {
    throw new Error(
      `compileHeapFragment: fragment output must be a single @location(0) vec4<f32> field; got ${fieldMatches.length} field(s)`,
    );
  }
  const outputFieldName = fieldMatches[0]![1]!;
  s = s.replace(outputMatch[0], "");

  // 6. Rewrite the function header to the heap-scene shape. Any parameter
  //    name (in, input, …) is renormalised to `in` so the user shader's
  //    field references work uniformly downstream.
  s = s.replace(fnHeader, "@fragment\nfn fs(in: VsOut) -> @location(0) vec4<f32> {");
  if (paramName !== "in") {
    // Rewrite `paramName.X` references to `in.X`.
    const paramRe = new RegExp(`\\b${escapeRegExp(paramName)}\\.`, "g");
    s = s.replace(paramRe, "in.");
  }

  // 7. Rewrite the body's `out: <Output>; ... out.<field> = expr; return out;` to
  //    return expr directly. wombat.shader's emit always uses this pattern; we
  //    transform it via two passes (capture the assignment, drop the var/return).
  const fieldRef = `out\\.${escapeRegExp(outputFieldName)}`;
  const assignRe = new RegExp(
    `var\\s+out\\s*:\\s*${escapeRegExp(outputStructName)}\\s*;([\\s\\S]*?)${fieldRef}\\s*=\\s*([\\s\\S]*?);([\\s\\S]*?)return\\s+out\\s*;`,
  );
  const assignMatch = s.match(assignRe);
  if (assignMatch === null) {
    // Fall back: if user wrote a more exotic body we surface a clear error.
    throw new Error(
      "compileHeapFragment: emitted body doesn't follow the var-out / out." + outputFieldName +
      " = ... / return out pattern (got unsupported emit shape)",
    );
  }
  const beforeAssign = assignMatch[1]!;     // statements between var-out decl and the field assignment
  const expr         = assignMatch[2]!;
  const afterAssign  = assignMatch[3]!;     // statements between the assignment and the return
  s = s.replace(assignMatch[0], beforeAssign + afterAssign + `return ${expr};`);

  // 8. Map references `in.<DslField>` to the corresponding heap VsOut
  //    field name (a 1:1 mapping when the user's input struct uses
  //    the canonical field names; identity otherwise).
  s = s.replace(/in\.(\w+)/g, (_m, name: string) => "in." + dslFieldToVsOut(name));

  // 9. Tidy: collapse 3+ blank lines to one.
  s = s.replace(/\n{3,}/g, "\n\n").trimStart();

  return s;
}

// Map a DSL uniform name to the heap-scene VsOut field that carries
// it. Today only `LightLocation` rides on VsOut; everything else
// would need to be added to the VS prelude in heapScene.ts.
function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function uniformToVsOut(name: string): string {
  switch (name) {
    case "LightLocation": return "lightLoc";
    default:
      throw new Error(
        `compileHeapFragment: uniform '${name}' is not carried on VsOut. Either thread it through the VS or use a fragment that doesn't read it.`,
      );
  }
}

// Map an inter-stage name from the DSL convention (PascalCase, e.g.
// 'Normals', 'WorldPositions') or our preferred direct names
// ('normal', 'worldPos') to the canonical VsOut field name.
function dslFieldToVsOut(name: string): string {
  switch (name) {
    // canonical heap-scene names — pass through.
    case "worldPos":
    case "normal":
    case "color":
    case "lightLoc":
      return name;
    // legacy PascalCase from the wombat.dom convention — auto-translate
    // so users can copy DSL effects from the per-RO path with minimal edits.
    case "WorldPositions": return "worldPos";
    case "WorldPosition":  return "worldPos";
    case "Normals":        return "normal";
    case "Normal":         return "normal";
    case "Colors":         return "color";
    case "Color":          return "color";
    case "LightLocation":  return "lightLoc";
    default:
      throw new Error(
        `compileHeapFragment: input field '${name}' has no mapping to VsOut. ` +
        `Use one of: worldPos, normal, color, lightLoc.`,
      );
  }
}
