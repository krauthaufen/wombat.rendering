// Test helper: wrap a hand-built `CompiledEffect` as an `Effect`
// whose `.compile()` returns the wrapped value. Used to feed the
// rendering layer with synthetic interfaces without going through
// the wombat.shader frontend (which needs TS source + IR + passes
// + emitter for every test).

import type { CompiledEffect, Effect } from "@aardworx/wombat.rendering-core";

export function fakeEffectFromCompiled(c: CompiledEffect): Effect {
  return {
    stages: [],
    id: "test",
    compile: () => c,
    dumpIR: () => "",
  };
}
