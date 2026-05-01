// render(cmd, prepared, output, token) — open a render pass on
// `output` (loadOp = "load"; clears go through the `Clear` command),
// record the prepared object, end the pass.
//
// Multiple prepared objects can be batched into one pass via
// `renderMany(...)` — same pass, multiple `record` calls — which
// is the path the RenderTask walker uses for an `Ordered` /
// `Unordered` subtree of leaves all targeting the same FBO.

import type { ClearValues, IFramebuffer } from "../core/index.js";
import type { AdaptiveToken } from "@aardworx/wombat.adaptive";

// Forward type — we don't import from -resources to avoid a
// commands→resources dependency cycle. Anyone constructing a
// PreparedRenderObject hands it to us; we only call `record`.
export interface Recordable {
  record(pass: GPURenderPassEncoder, token: AdaptiveToken): void;
}

/**
 * Build a render-pass descriptor for `output`. If `clear` is set,
 * named attachments use `loadOp: "clear"` with the requested
 * values; the rest fall back to `loadOp: "load"`. Used by
 * `render` / `renderMany` and by the runtime walker's Clear+Render
 * coalescing path.
 */
export function beginPassDescriptor(output: IFramebuffer, clear?: ClearValues): GPURenderPassDescriptor {
  const colorAttachments: GPURenderPassColorAttachment[] = [];
  const msaa = output.signature.sampleCount > 1;
  for (const name of output.signature.colorNames) {
    const view = output.colors.tryFind(name);
    if (view === undefined) {
      throw new Error(`render: framebuffer is missing color attachment "${name}"`);
    }
    const resolveTarget = msaa ? output.resolveColors?.tryFind(name) : undefined;
    if (msaa && resolveTarget === undefined) {
      throw new Error(`render: MSAA framebuffer is missing resolve target for "${name}"`);
    }
    const cv = clear?.colors?.tryFind(name);
    if (cv !== undefined) {
      const d = cv as unknown as { _data: ArrayLike<number> };
      colorAttachments.push({
        view, loadOp: "clear", storeOp: "store",
        ...(resolveTarget !== undefined ? { resolveTarget } : {}),
        clearValue: { r: d._data[0]!, g: d._data[1]!, b: d._data[2]!, a: d._data[3]! },
      });
    } else {
      colorAttachments.push({
        view, loadOp: "load", storeOp: "store",
        ...(resolveTarget !== undefined ? { resolveTarget } : {}),
      });
    }
  }
  let depthStencilAttachment: GPURenderPassDepthStencilAttachment | undefined;
  if (output.signature.depthStencil !== undefined) {
    if (output.depthStencil === undefined) {
      throw new Error("render: signature has depthStencil but framebuffer.depthStencil is undefined");
    }
    const sig = output.signature.depthStencil;
    const att: GPURenderPassDepthStencilAttachment = { view: output.depthStencil };
    if (sig.hasDepth) {
      if (clear?.depth !== undefined) {
        att.depthLoadOp = "clear"; att.depthStoreOp = "store"; att.depthClearValue = clear.depth;
      } else {
        att.depthLoadOp = "load"; att.depthStoreOp = "store";
      }
    }
    if (sig.hasStencil) {
      if (clear?.stencil !== undefined) {
        att.stencilLoadOp = "clear"; att.stencilStoreOp = "store"; att.stencilClearValue = clear.stencil;
      } else {
        att.stencilLoadOp = "load"; att.stencilStoreOp = "store";
      }
    }
    depthStencilAttachment = att;
  }
  return {
    colorAttachments,
    ...(depthStencilAttachment !== undefined ? { depthStencilAttachment } : {}),
  };
}

export function render(
  cmd: GPUCommandEncoder,
  prepared: Recordable,
  output: IFramebuffer,
  token: AdaptiveToken,
  clear?: ClearValues,
): void {
  const pass = cmd.beginRenderPass(beginPassDescriptor(output, clear));
  prepared.record(pass, token);
  pass.end();
}

export function renderMany(
  cmd: GPUCommandEncoder,
  prepared: readonly Recordable[],
  output: IFramebuffer,
  token: AdaptiveToken,
  clear?: ClearValues,
): void {
  if (prepared.length === 0) return;
  const pass = cmd.beginRenderPass(beginPassDescriptor(output, clear));
  for (const p of prepared) p.record(pass, token);
  pass.end();
}
