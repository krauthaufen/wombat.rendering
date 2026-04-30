// render(cmd, prepared, output, token) — open a render pass on
// `output` (loadOp = "load"; clears go through the `Clear` command),
// record the prepared object, end the pass.
//
// Multiple prepared objects can be batched into one pass via
// `renderMany(...)` — same pass, multiple `record` calls — which
// is the path the RenderTask walker uses for an `Ordered` /
// `Unordered` subtree of leaves all targeting the same FBO.

import type { IFramebuffer } from "@aardworx/wombat.rendering-core";
import type { AdaptiveToken } from "@aardworx/wombat.adaptive";

// Forward type — we don't import from -resources to avoid a
// commands→resources dependency cycle. Anyone constructing a
// PreparedRenderObject hands it to us; we only call `record`.
export interface Recordable {
  record(pass: GPURenderPassEncoder, token: AdaptiveToken): void;
}

function beginPassDescriptor(output: IFramebuffer): GPURenderPassDescriptor {
  const colorAttachments: GPURenderPassColorAttachment[] = [];
  for (const [name] of output.signature.colors) {
    const view = output.colors.tryFind(name);
    if (view === undefined) {
      throw new Error(`render: framebuffer is missing color attachment "${name}"`);
    }
    colorAttachments.push({ view, loadOp: "load", storeOp: "store" });
  }
  let depthStencilAttachment: GPURenderPassDepthStencilAttachment | undefined;
  if (output.signature.depthStencil !== undefined) {
    if (output.depthStencil === undefined) {
      throw new Error("render: signature has depthStencil but framebuffer.depthStencil is undefined");
    }
    const sig = output.signature.depthStencil;
    const att: GPURenderPassDepthStencilAttachment = { view: output.depthStencil };
    if (sig.hasDepth) { att.depthLoadOp = "load"; att.depthStoreOp = "store"; }
    if (sig.hasStencil) { att.stencilLoadOp = "load"; att.stencilStoreOp = "store"; }
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
): void {
  const pass = cmd.beginRenderPass(beginPassDescriptor(output));
  prepared.record(pass, token);
  pass.end();
}

export function renderMany(
  cmd: GPUCommandEncoder,
  prepared: readonly Recordable[],
  output: IFramebuffer,
  token: AdaptiveToken,
): void {
  if (prepared.length === 0) return;
  const pass = cmd.beginRenderPass(beginPassDescriptor(output));
  for (const p of prepared) p.record(pass, token);
  pass.end();
}
