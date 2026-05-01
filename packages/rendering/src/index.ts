// Public API of `@aardworx/wombat.rendering`.
//
// Default entry re-exports everything: types, resource preparers,
// command-stream functions, runtime, and the window/canvas glue.
// Subpath exports give granular access for advanced consumers.
//
//   import { ... } from "@aardworx/wombat.rendering";        // everything
//   import { ... } from "@aardworx/wombat.rendering/core";   // types only
//   import { ... } from "@aardworx/wombat.rendering/resources";
//   import { ... } from "@aardworx/wombat.rendering/commands";
//   import { ... } from "@aardworx/wombat.rendering/runtime";
//   import { ... } from "@aardworx/wombat.rendering/window";

export * from "./core/index.js";
export * from "./resources/index.js";
export * from "./commands/index.js";
export * from "./runtime/index.js";
export * from "./window/index.js";
