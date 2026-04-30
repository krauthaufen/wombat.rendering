// Test helper: wrap a hand-built `CompiledEffect` as an `Effect`
// whose `.compile()` returns it.
//
// Used only when the test needs an interface shape that wombat.shader's
// non-Vite-plugin path can't produce easily — most notably texture +
// sampler bindings (real users get those via the
// `Sampler2D`-capture transform applied by `@aardworx/wombat.shader-vite`,
// which we don't run in vitest).
//
// Effect.id is set to a deterministic hash of the descriptor so the
// pipeline cache still keys correctly across calls.

import type { CompiledEffect, Effect } from "@aardworx/wombat.rendering-core";

let counter = 0;

export function fakeEffectFromCompiled(c: CompiledEffect, id?: string): Effect {
  const finalId = id ?? `fake-effect-${++counter}`;
  return {
    stages: [],
    id: finalId,
    compile: () => c,
    dumpIR: () => "",
  };
}
