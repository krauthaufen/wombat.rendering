// §7 v2 — derived-uniform rule registry.
//
// Content-hash–keyed table of FLATTENED rules. `register` dedups on the
// flattened IR's hash, assigns a monotonic id (never reused), and bumps
// `version` whenever a new unique rule appears so the dispatcher knows to
// recompile its uber-kernel. v0 never sweeps dead entries (a stale rule's
// switch arm just goes unused — bytes per rule). See
// docs/derived-uniforms-extensible.md § "Registry" / "Lifecycle".

import type { Expr, Type } from "@aardworx/wombat.shader/ir";
import type { DerivedRule } from "./rule.js";
import { inputsOf, type RuleInput } from "./flatten.js";

export interface RuleEntry {
  /** Monotonic id — the value the uber-kernel `switch`es on. Never reused. */
  readonly id: number;
  /** Flattened IR (reads only non-derived inputs). */
  readonly ir: Expr;
  readonly outputType: Type;
  /** Distinct inputs of `ir`, sorted by name — defines the switch-arm argument order. */
  readonly inputs: readonly RuleInput[];
  refcount: number;
}

export class DerivedUniformRegistry {
  private readonly byHash = new Map<string, RuleEntry>();
  private readonly byId = new Map<number, RuleEntry>();
  private nextId = 0;
  /** Bumps on every `register` that finds a NEW hash. */
  version = 0;

  /**
   * Register a flattened rule. Returns its stable id. Idempotent per hash
   * (just bumps the refcount); a new hash allocates an id and bumps `version`.
   */
  register(rule: DerivedRule): number {
    const cached = this.byHash.get(rule.hash);
    if (cached !== undefined) {
      cached.refcount++;
      return cached.id;
    }
    const entry: RuleEntry = {
      id: this.nextId++,
      ir: rule.ir,
      outputType: rule.outputType,
      inputs: inputsOf(rule.ir),
      refcount: 1,
    };
    this.byHash.set(rule.hash, entry);
    this.byId.set(entry.id, entry);
    this.version++;
    return entry.id;
  }

  /** Drop one reference. v0: never sweeps — kept for forward-compat with v1 GC. */
  release(hash: string): void {
    const entry = this.byHash.get(hash);
    if (entry !== undefined && entry.refcount > 0) entry.refcount--;
  }

  get(id: number): RuleEntry | undefined {
    return this.byId.get(id);
  }

  getByHash(hash: string): RuleEntry | undefined {
    return this.byHash.get(hash);
  }

  /** All entries in id order — the codegen's switch-arm list. */
  entries(): RuleEntry[] {
    return [...this.byId.values()].sort((a, b) => a.id - b.id);
  }

  get size(): number {
    return this.byId.size;
  }
}
