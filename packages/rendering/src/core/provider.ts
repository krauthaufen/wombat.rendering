// IUniformProvider / IAttributeProvider — lazy, by-name lookup of a
// RenderObject's uniform values and vertex/instance attribute views.
//
// Modelled on Aardvark.Rendering's `IUniformProvider` /
// `IAttributeProvider`: the binding layer always pulls *shader-driven*
// — for each name the compiled shader actually declares it calls
// `tryGet(name)`. A name no shader reads is therefore never
// materialised by a lazy provider. This is what lets the scene-graph
// auto-inject all ~15 derived trafo uniforms (`ModelViewProjTrafo`,
// `NormalMatrix`, the inverses, …) per leaf for free: their `compose`
// / `inverse` aval chains are only built when some effect's interface
// references the name.
//
// `names()` is the (cheap) enumeration surface — a lazy provider lists
// the names it *could* produce without materialising their values. The
// only consumers that enumerate (rather than pull by shader name) are
// the texture-split in the Sg compile layer (operates on the user map,
// not a provider) and the heap-eligibility stride check (iterates the
// attribute provider's names — always map-backed in practice).

import type { HashMap, aval } from "@aardworx/wombat.adaptive";
import type { BufferView } from "./bufferView.js";

export interface IUniformProvider {
  /** Uniform value for `name`, or `undefined` if not provided. Pulled
   *  once per shader-declared uniform; lazy providers materialise the
   *  value here, on first request. */
  tryGet(name: string): aval<unknown> | undefined;
  /** Names this provider knows about (for enumeration / diagnostics).
   *  A lazy provider lists derivable names without building them. */
  names(): Iterable<string>;
}

export interface IAttributeProvider {
  /** Vertex/instance attribute view for `name`, or `undefined`. */
  tryGet(name: string): BufferView | undefined;
  names(): Iterable<string>;
}

// ─── shared impl helpers ───────────────────────────────────────────────

function mapKeys<K, V>(m: HashMap<K, V>): K[] {
  const out: K[] = [];
  for (const [k] of m) out.push(k);
  return out;
}

function hasOwn(o: object, k: string): boolean {
  return Object.prototype.hasOwnProperty.call(o, k);
}

const EMPTY_NAMES: readonly string[] = [];

// Class-based providers: methods live on the prototype, so a provider
// instance is ONE small object instead of an object + 2 closures + a
// shared closure context. At heap scale (a provider per RenderObject)
// that difference is measurable JS-heap ballast.

class MapProvider<T> {
  constructor(private readonly m: HashMap<string, T>) {}
  tryGet(n: string): T | undefined { return this.m.tryFind(n); }
  names(): Iterable<string> { return mapKeys(this.m); }
}

class ObjectProvider<T> {
  constructor(private readonly o: Readonly<Record<string, T>>) {}
  tryGet(n: string): T | undefined { return hasOwn(this.o, n) ? this.o[n] : undefined; }
  names(): Iterable<string> { return Object.keys(this.o); }
}

class UnionProvider<T> {
  constructor(private readonly ps: readonly { tryGet(name: string): T | undefined; names(): Iterable<string> }[]) {}
  tryGet(n: string): T | undefined {
    for (let i = 0; i < this.ps.length; i++) {
      const v = this.ps[i]!.tryGet(n);
      if (v !== undefined) return v;
    }
    return undefined;
  }
  names(): Iterable<string> {
    const s = new Set<string>();
    for (const p of this.ps) for (const n of p.names()) s.add(n);
    return s;
  }
}

class LazyProvider<T> {
  private readonly cache = new Map<string, T | undefined>();
  constructor(
    private readonly known: readonly string[],
    private readonly compute: (name: string) => T | undefined,
  ) {}
  tryGet(n: string): T | undefined {
    const hit = this.cache.get(n);
    if (hit !== undefined || this.cache.has(n)) return hit;
    const v = this.compute(n);
    this.cache.set(n, v);
    return v;
  }
  names(): Iterable<string> { return this.known; }
}

const EMPTY_PROVIDER = { tryGet: (): undefined => undefined, names: (): Iterable<string> => EMPTY_NAMES };

function makeProvider<T>(): {
  empty: { tryGet(name: string): T | undefined; names(): Iterable<string> };
  ofMap(m: HashMap<string, T>): { tryGet(name: string): T | undefined; names(): Iterable<string> };
  ofObject(o: Readonly<Record<string, T>>): { tryGet(name: string): T | undefined; names(): Iterable<string> };
  union(...ps: { tryGet(name: string): T | undefined; names(): Iterable<string> }[]): { tryGet(name: string): T | undefined; names(): Iterable<string> };
  lazy(known: readonly string[], compute: (name: string) => T | undefined): { tryGet(name: string): T | undefined; names(): Iterable<string> };
} {
  type P = { tryGet(name: string): T | undefined; names(): Iterable<string> };
  const empty = EMPTY_PROVIDER as P;
  return {
    empty,
    ofMap(m: HashMap<string, T>): P {
      if (m.containsKey === undefined) return empty; // defensive
      return new MapProvider(m);
    },
    ofObject(o: Readonly<Record<string, T>>): P {
      return new ObjectProvider(o);
    },
    /** First non-`undefined` wins. Pass providers in priority order —
     *  e.g. `[leafAttrs, scopeAttrs]` so a leaf shadows the scope, or
     *  `[userUniforms, autoInjected]` so the user overrides defaults. */
    union(...ps: P[]): P {
      if (ps.length === 0) return empty;
      if (ps.length === 1) return ps[0]!;
      return new UnionProvider(ps);
    },
    /** `compute(name)` runs the first time `name` is requested; the
     *  result (incl. `undefined`) is memoised. `known` is the cheap
     *  list of names `compute` can produce. */
    lazy(known: readonly string[], compute: (name: string) => T | undefined): P {
      return new LazyProvider(known, compute);
    },
  };
}

export const UniformProvider = makeProvider<aval<unknown>>() as {
  empty: IUniformProvider;
  ofMap(m: HashMap<string, aval<unknown>>): IUniformProvider;
  ofObject(o: Readonly<Record<string, aval<unknown>>>): IUniformProvider;
  union(...ps: IUniformProvider[]): IUniformProvider;
  lazy(known: readonly string[], compute: (name: string) => aval<unknown> | undefined): IUniformProvider;
};

export const AttributeProvider = makeProvider<BufferView>() as {
  empty: IAttributeProvider;
  ofMap(m: HashMap<string, BufferView>): IAttributeProvider;
  ofObject(o: Readonly<Record<string, BufferView>>): IAttributeProvider;
  union(...ps: IAttributeProvider[]): IAttributeProvider;
  lazy(known: readonly string[], compute: (name: string) => BufferView | undefined): IAttributeProvider;
};

function isProvider(x: unknown): x is { tryGet(n: string): unknown; names(): Iterable<string> } {
  return x !== null && typeof x === "object" && typeof (x as { tryGet?: unknown }).tryGet === "function";
}

/** Normalise a `RenderObject.uniforms` value: a real provider passes
 *  through; a plain `HashMap<string, aval>` (legacy shape, still used
 *  by some hand-built test fixtures) is wrapped as a map-backed
 *  provider. Consumers call this at the boundary so both shapes work. */
export function asUniformProvider(x: IUniformProvider | HashMap<string, aval<unknown>>): IUniformProvider {
  return isProvider(x) ? x : UniformProvider.ofMap(x);
}

/** Likewise for `vertexAttributes` / `instanceAttributes`. */
export function asAttributeProvider(x: IAttributeProvider | HashMap<string, BufferView>): IAttributeProvider {
  return isProvider(x) ? x : AttributeProvider.ofMap(x);
}
