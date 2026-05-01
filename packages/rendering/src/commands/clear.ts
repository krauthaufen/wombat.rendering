// clear(cmd, output, values) — open a render pass on `output`
// with `loadOp: "clear"` for the named attachments listed in
// `values`, and `loadOp: "load"` for the rest. End the pass
// immediately. This is the encoding for a `Clear` Command.
//
// Note that color clears are typed as `V4f | V4i | V4ui`; we
// translate to WebGPU's `GPUColorDict` (always `r/g/b/a` numbers)
// directly. WebGPU figures out the integer-vs-float interpretation
// from the texture's format.

import type { ClearValues, IFramebuffer } from "../core/index.js";

export function clear(
  cmd: GPUCommandEncoder,
  output: IFramebuffer,
  values: ClearValues,
): void {
  const colorAttachments: GPURenderPassColorAttachment[] = [];
  for (const [name] of output.signature.colors) {
    const view = output.colors.tryFind(name);
    if (view === undefined) {
      throw new Error(`clear: framebuffer is missing color attachment "${name}"`);
    }
    const cv = values.colors?.tryFind(name);
    if (cv !== undefined) {
      const d = cv as unknown as { _data: ArrayLike<number> };
      colorAttachments.push({
        view,
        loadOp: "clear",
        storeOp: "store",
        clearValue: { r: d._data[0]!, g: d._data[1]!, b: d._data[2]!, a: d._data[3]! },
      });
    } else {
      colorAttachments.push({ view, loadOp: "load", storeOp: "store" });
    }
  }
  let depthStencilAttachment: GPURenderPassDepthStencilAttachment | undefined;
  if (output.signature.depthStencil !== undefined) {
    if (output.depthStencil === undefined) {
      throw new Error("clear: signature has depthStencil but framebuffer.depthStencil is undefined");
    }
    const sig = output.signature.depthStencil;
    const att: GPURenderPassDepthStencilAttachment = { view: output.depthStencil };
    if (sig.hasDepth) {
      if (values.depth !== undefined) {
        att.depthLoadOp = "clear"; att.depthStoreOp = "store"; att.depthClearValue = values.depth;
      } else {
        att.depthLoadOp = "load"; att.depthStoreOp = "store";
      }
    }
    if (sig.hasStencil) {
      if (values.stencil !== undefined) {
        att.stencilLoadOp = "clear"; att.stencilStoreOp = "store"; att.stencilClearValue = values.stencil;
      } else {
        att.stencilLoadOp = "load"; att.stencilStoreOp = "store";
      }
    }
    depthStencilAttachment = att;
  }
  const desc: GPURenderPassDescriptor = {
    colorAttachments,
    ...(depthStencilAttachment !== undefined ? { depthStencilAttachment } : {}),
  };
  const pass = cmd.beginRenderPass(desc);
  pass.end();
}
