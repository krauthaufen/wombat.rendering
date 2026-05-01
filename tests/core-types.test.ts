// Phase 0 sanity checks: the public types of @aardworx/wombat.rendering-core
// can be wired together to construct a Command, RenderTree, and a
// trivial AdaptiveResource. No GPU work — these run under Node.

import { describe, expect, it } from "vitest";
import { AdaptiveToken, HashMap, cval, transact } from "@aardworx/wombat.adaptive";
import { V2fArray, V4f } from "@aardworx/wombat.base";
import {
  AdaptiveResource,
  IBuffer,
  ITexture,
  RenderContext,
  RenderTree,
  type Command,
  type ClearValues,
  type FramebufferSignature,
  type IFramebuffer,
  type RenderObject,
  PipelineState,
} from "@aardworx/wombat.rendering/core";

const sig: FramebufferSignature = {
  colors: HashMap.empty<string, GPUTextureFormat>().add("albedo", "rgba8unorm"),
  depthStencil: { format: "depth24plus", hasDepth: true, hasStencil: false },
  sampleCount: 1,
};

const fb: IFramebuffer = {
  signature: sig,
  colors: HashMap.empty<string, GPUTextureView>(),
  width: 512,
  height: 512,
};

describe("core types", () => {
  it("RenderTree.empty / leaf / ordered / unordered compose", () => {
    const t = RenderTree.ordered(
      RenderTree.empty,
      RenderTree.unordered(RenderTree.empty),
    );
    expect(t.kind).toBe("Ordered");
    if (t.kind === "Ordered") {
      expect(t.children).toHaveLength(2);
      expect(t.children[1]!.kind).toBe("Unordered");
    }
  });

  it("Clear command with named V4f color clear", () => {
    const values: ClearValues = {
      colors: HashMap.empty<string, V4f>().add("albedo", new V4f(0, 0, 0, 1)),
      depth: 1,
    };
    const cmd: Command = { kind: "Clear", output: fb, values };
    expect(cmd.kind).toBe("Clear");
    if (cmd.kind === "Clear") {
      expect(cmd.values.depth).toBe(1);
      expect(cmd.values.colors!.tryFind("albedo")).toBeDefined();
    }
  });

  it("Render command wraps a RenderTree", () => {
    const cmd: Command = { kind: "Render", output: fb, tree: RenderTree.empty };
    expect(cmd.kind).toBe("Render");
  });

  it("Custom command exposes raw encoder", () => {
    let called = false;
    const cmd: Command = {
      kind: "Custom",
      encode: (_enc) => { called = true; },
    };
    if (cmd.kind === "Custom") cmd.encode({} as GPUCommandEncoder);
    expect(called).toBe(true);
  });

  it("RenderObject can be constructed with HashMap-typed input bags", () => {
    const ro: RenderObject = {
      effect: {} as never,
      pipelineState: PipelineState.constant({
        rasterizer: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
      }),
      vertexAttributes: HashMap.empty(),
      uniforms: HashMap.empty(),
      textures: HashMap.empty(),
      samplers: HashMap.empty(),
      drawCall: cval({ kind: "non-indexed" as const, vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    };
    expect(ro.vertexAttributes.count).toBe(0);
  });
});

describe("IBuffer / ITexture", () => {
  it("IBuffer.fromHost wraps a wombat.base packed array", () => {
    const positions = new V2fArray(64);  // 64 V2f = 512 bytes
    const buf = IBuffer.fromHost(positions.buffer);
    expect(buf.kind).toBe("host");
    if (buf.kind === "host") {
      expect(buf.sizeBytes).toBe(64 * 2 * 4);
      expect(buf.data).toBe(positions.buffer);
    }
  });

  it("IBuffer.fromHost wraps a TypedArray (ArrayBufferView)", () => {
    const arr = new Float32Array(16);
    const buf = IBuffer.fromHost(arr);
    expect(buf.kind).toBe("host");
    if (buf.kind === "host") expect(buf.sizeBytes).toBe(64);
  });

  it("IBuffer.fromGPU keeps the handle as-is", () => {
    const fakeGpu = { size: 1024 } as GPUBuffer;
    const buf = IBuffer.fromGPU(fakeGpu);
    expect(buf.kind).toBe("gpu");
    if (buf.kind === "gpu") {
      expect(buf.buffer).toBe(fakeGpu);
      expect(buf.sizeBytes).toBe(1024);
    }
  });

  it("ITexture.fromRaw and ITexture.fromGPU discriminate", () => {
    const raw = ITexture.fromRaw({ data: new Uint8Array(4), width: 1, height: 1, format: "rgba8unorm" });
    expect(raw.kind).toBe("host");
    const gpu = ITexture.fromGPU({} as GPUTexture);
    expect(gpu.kind).toBe("gpu");
  });
});

describe("AdaptiveResource as aval<T>", () => {
  // Counter-style adaptive resource: emits an integer that
  // depends on an underlying cval and tracks acquire/release.
  class FakeRes extends AdaptiveResource<number> {
    public created = 0;
    public destroyed = 0;
    public computes = 0;
    constructor(private readonly source: ReturnType<typeof cval<number>>) { super(); }
    protected create(): void { this.created++; }
    protected destroy(): void { this.destroyed++; }
    override compute(token: AdaptiveToken): number {
      this.computes++;
      return this.source.getValue(token) * 2;
    }
  }

  it("acquire/create on 0→1, release/destroy on 1→0", () => {
    const r = new FakeRes(cval(3));
    expect(r.refCount).toBe(0);
    r.acquire(); r.acquire();
    expect(r.created).toBe(1);
    expect(r.refCount).toBe(2);
    r.release();
    expect(r.destroyed).toBe(0);
    r.release();
    expect(r.destroyed).toBe(1);
  });

  it("getValue follows the underlying aval and caches", () => {
    const src = cval(5);
    const r = new FakeRes(src);
    r.acquire();
    const tok = AdaptiveToken.top;
    expect(r.getValue(tok)).toBe(10);
    expect(r.getValue(tok)).toBe(10);
    expect(r.computes).toBe(1);    // cached
    transact(() => { src.value = 7; });
    expect(r.getValue(tok)).toBe(14);
    expect(r.computes).toBe(2);
    r.release();
  });

  it("RenderContext.withEncoder threads the encoder through compute()", () => {
    const seen: GPUCommandEncoder[] = [];
    class EncRes extends AdaptiveResource<number> {
      protected create(): void {}
      protected destroy(): void {}
      override compute(_t: AdaptiveToken): number {
        seen.push(RenderContext.requireEncoder());
        return 1;
      }
    }
    const r = new EncRes();
    r.acquire();
    const enc = { id: "enc" } as unknown as GPUCommandEncoder;
    RenderContext.withEncoder(enc, () => {
      r.getValue(AdaptiveToken.top);
    });
    expect(seen).toEqual([enc]);
    expect(RenderContext.encoder).toBeNull();
    r.release();
  });

  it("requireEncoder throws outside withEncoder", () => {
    expect(() => RenderContext.requireEncoder()).toThrow(/no active encoder/);
  });

  it("release without acquire throws", () => {
    const r = new FakeRes(cval(0));
    expect(() => r.release()).toThrow(/release without matching acquire/);
  });
});
