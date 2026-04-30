// Public API of @aardworx/wombat.rendering-commands.
// Layer 3 — command-stream functions on `GPUCommandEncoder`.

export { clear } from "./clear.js";

export {
  render,
  renderMany,
  type Recordable,
} from "./render.js";
