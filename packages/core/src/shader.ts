// Re-exports + placeholder types for the wombat.shader contract.
//
// The rendering layer consumes wombat.shader as an opaque black
// box: an `Effect` is whatever the user gave to wombat.shader's
// frontend; a `CompiledEffect` is the result of compileEffect on
// some target (always WGSL here in v0.1).
//
// We don't depend on wombat.shader's runtime package at this
// layer (to keep `core` types-only). Concrete `Effect`/`CompiledEffect`
// types are imported from `@aardworx/wombat.shader-runtime` in the
// `resources` package once it lands.

/**
 * A wombat.shader Effect. Opaque to the rendering layer; consumed
 * by `compileEffect` in the resources package which knows how to
 * interpret it.
 */
export type Effect = { readonly __wombatEffect: unique symbol } & object;

/**
 * Compiled WGSL + program interface produced by wombat.shader.
 * The structure matches `@aardworx/wombat.shader-runtime`'s
 * `CompiledEffect`; we keep a structural minimum here so layer-3
 * code can read what it needs without a hard dep.
 */
export interface CompiledEffect {
  readonly target: "wgsl" | "glsl";
  readonly stages: readonly { readonly stage: string; readonly source: string }[];
  readonly interface: ProgramInterface;
}

/**
 * Subset of wombat.shader's `ProgramInterface` that the rendering
 * layer reads when lowering name-keyed inputs to WebGPU bindings.
 * The full interface lives in wombat.shader-runtime.
 */
export interface ProgramInterface {
  readonly vertexAttributes: readonly { readonly name: string; readonly location: number; readonly format: GPUVertexFormat }[];
  readonly fragmentOutputs: readonly { readonly name: string; readonly location: number }[];
  readonly uniformBuffers: readonly UniformBufferLayout[];
  readonly samplers: readonly { readonly name: string; readonly group: number; readonly binding: number }[];
  readonly textures: readonly { readonly name: string; readonly group: number; readonly binding: number; readonly sampleType: GPUTextureSampleType }[];
  readonly storageBuffers: readonly { readonly name: string; readonly group: number; readonly binding: number; readonly access: "read" | "read_write" }[];
}

export interface UniformBufferLayout {
  readonly name: string;
  readonly group: number;
  readonly binding: number;
  readonly sizeBytes: number;
  readonly fields: readonly { readonly name: string; readonly offset: number; readonly sizeBytes: number }[];
}
