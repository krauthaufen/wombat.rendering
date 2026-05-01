// Public API of @aardworx/wombat.rendering-window.

export {
  attachCanvas,
  type AttachCanvasOptions,
  type CanvasAttachment,
} from "./canvas.js";

export {
  runFrame,
  type FrameCallback,
  type FrameInfo,
  type RunFrameOptions,
} from "./loop.js";
