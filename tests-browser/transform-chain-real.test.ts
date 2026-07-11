// GPU transform propagation — real-GPU verification of the modelChain path:
// the chain pass composes a per-RO Model constituent (fwd+inv) in df32, then
// §7 reads it. Verifies the Model constituent itself, the chain→constituent→§7
// ModelView end-to-end, and the fan-out property (shared root → O(1) dirty).

import { describe, expect, it } from "vitest";
import { AVal, cval, transact, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, M44d } from "@aardworx/wombat.base";
import {
  DerivedUniformsScene, registerRoDerivations, type RoDerivedRequest,
} from "../packages/rendering/src/runtime/derivedUniforms/sceneIntegration.js";
import { STANDARD_DERIVED_RULES } from "../packages/rendering/src/runtime/derivedUniforms/recipes.js";
import { requestRealDevice } from "./_realGpu.js";

async function readFloats(device: GPUDevice, buf: GPUBuffer, byteOffset: number, count: number): Promise<Float32Array> {
  const bytes = count * 4;
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, byteOffset, staging, 0, bytes);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap(); staging.destroy();
  return out;
}

// Collapse a df32 mat4 constituent slot (16 hi/lo pairs, row-major) to f32[16].
async function readConstituent(device: GPUDevice, scene: DerivedUniformsScene, slot: number): Promise<Float32Array> {
  const raw = await readFloats(device, scene.constituentsBuf, slot * 128, 32);
  const out = new Float32Array(16);
  for (let i = 0; i < 16; i++) out[i] = raw[i * 2]! + raw[i * 2 + 1]!;
  return out;
}

function expectMatClose(got: Float32Array, want: M44d): void {
  const w = [want.M00, want.M01, want.M02, want.M03, want.M10, want.M11, want.M12, want.M13,
             want.M20, want.M21, want.M22, want.M23, want.M30, want.M31, want.M32, want.M33];
  for (let i = 0; i < 16; i++) expect(Math.abs(got[i]! - w[i]!)).toBeLessThanOrEqual(1e-5 * (1 + Math.abs(w[i]!)));
}

function runFrame(scene: DerivedUniformsScene, device: GPUDevice): void {
  scene.uploadDirty(scene.pullDirty(AdaptiveToken.top));
  const enc = device.createCommandEncoder();
  scene.encode(enc);
  device.queue.submit([enc.finish()]);
}

const baseReq = (over: Partial<RoDerivedRequest>): RoDerivedRequest => ({
  rules: new Map(), trafoAvals: new Map(), hostUniformOffset: () => undefined,
  outputOffset: () => undefined, drawHeaderBaseByte: 0, chunkIdx: 0, ...over,
});

describe("GPU transform propagation — modelChain (real GPU)", () => {
  it("chain pass writes the per-RO Model constituent (fwd = product, inv = inverse)", async () => {
    const device = await requestRealDevice();
    const scene = new DerivedUniformsScene(device);
    const heap = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    scene.setMainHeapForChunk(0, () => heap);

    const a = Trafo3d.translation(new V3d(1, 2, 3));
    const b = Trafo3d.rotation(new V3d(0, 0, 1), 0.6);
    const c = Trafo3d.translation(new V3d(-4, 0.5, 2));
    const reg = registerRoDerivations(scene, {}, baseReq({
      modelChain: [AVal.constant(a), AVal.constant(b), AVal.constant(c)],
    }));
    runFrame(scene, device);

    const want = a.forward.mul(b.forward).mul(c.forward);
    expectMatClose(await readConstituent(device, scene, reg.modelLeaf!.modelPair.fwd), want);
    expectMatClose(await readConstituent(device, scene, reg.modelLeaf!.modelPair.inv), want.inverse());
    scene.dispose(); heap.destroy();
  });

  it("§7 reads the chain-written Model: ModelView == View · (chain)", async () => {
    const device = await requestRealDevice();
    const scene = new DerivedUniformsScene(device);
    const heap = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    scene.setMainHeapForChunk(0, () => heap);

    const m0 = Trafo3d.translation(new V3d(2, 0, 1));
    const m1 = Trafo3d.rotation(new V3d(1, 0, 0), 0.4);
    const view = Trafo3d.translation(new V3d(-3, 1, 2)).mul(Trafo3d.rotation(new V3d(0, 1, 0), 0.2));
    registerRoDerivations(scene, {}, baseReq({
      rules: new Map([["ModelViewTrafo", STANDARD_DERIVED_RULES.get("ModelViewTrafo")!]]),
      modelChain: [AVal.constant(m0), AVal.constant(m1)],
      trafoAvals: new Map([["View", AVal.constant(view)]]),
      outputOffset: (n) => (n === "ModelViewTrafo" ? 0 : undefined),
    }));
    runFrame(scene, device);

    const want = view.forward.mul(m0.forward.mul(m1.forward)); // View · Model
    expectMatClose(await readFloats(device, heap, 0, 16), want);
    scene.dispose(); heap.destroy();
  });

  it("df32 survives planet-scale magnitudes (ECEF): ModelView abs error < 1mm", async () => {
    const device = await requestRealDevice();
    const scene = new DerivedUniformsScene(device);
    const heap = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    scene.setMainHeapForChunk(0, () => heap);

    // Innsbruck-scale ECEF: model carries a 4.7e6 m translation; the camera
    // sits ~110 m away, so the composed ModelView is eye-relative (entries
    // ≤ a few hundred). A correct df32 chain keeps absolute error ≪ 1 mm;
    // a chain that degrades to f32 shows ~0.3–0.5 m (the observed wobble).
    const tileT = new V3d(4250557, 857436, 4662583);
    // mul is "a then b": rotate locally FIRST, then translate to the tile.
    const model = Trafo3d.rotation(new V3d(0, 0, 1), 0.3).mul(Trafo3d.translation(tileT));
    const eyeT = Trafo3d.translation(new V3d(-(tileT.x + 80), -(tileT.y - 50), -(tileT.z + 60)));
    const view = eyeT.mul(Trafo3d.rotation(new V3d(0, 1, 0), 0.7));
    registerRoDerivations(scene, {}, baseReq({
      rules: new Map([["ModelViewTrafo", STANDARD_DERIVED_RULES.get("ModelViewTrafo")!]]),
      modelChain: [AVal.constant(model)],
      trafoAvals: new Map([["View", AVal.constant(view)]]),
      outputOffset: (n) => (n === "ModelViewTrafo" ? 0 : undefined),
    }));
    runFrame(scene, device);

    const got = await readFloats(device, heap, 0, 16);
    const want = view.forward.mul(model.forward);
    const w = [want.M00, want.M01, want.M02, want.M03, want.M10, want.M11, want.M12, want.M13,
               want.M20, want.M21, want.M22, want.M23, want.M30, want.M31, want.M32, want.M33];
    for (let i = 0; i < 16; i++) {
      expect(Math.abs(got[i]! - w[i]!)).toBeLessThanOrEqual(1e-3);
    }
    scene.dispose(); heap.destroy();
  });

  it("shared root → root change marks O(1) slots; all Models update", async () => {
    const device = await requestRealDevice();
    const scene = new DerivedUniformsScene(device);
    const heap = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    scene.setMainHeapForChunk(0, () => heap);

    const root = cval(Trafo3d.translation(new V3d(0, 0, 0)));
    const N = 150;
    const regs = [];
    for (let i = 0; i < N; i++) {
      // chain [leaf, root] (SG order) → the root suffix is SHARED in the trie.
      regs.push(registerRoDerivations(scene, {}, baseReq({ modelChain: [AVal.constant(Trafo3d.translation(new V3d(i, 0, 0))), root] })));
    }
    // Prefix-sharing: one shared root node + N distinct leaf nodes (not 2·N).
    expect(scene.trafoTree.nodeCount).toBe(N + 1);
    runFrame(scene, device);

    transact(() => { root.value = Trafo3d.translation(new V3d(1000, 0, 0)); });
    scene.routeInputChanged(root);
    const dirty = scene.pullDirty(AdaptiveToken.top);
    expect(dirty.size).toBe(2); // root fwd+inv only — not 2·N
    scene.uploadDirty(dirty);
    const enc = device.createCommandEncoder();
    scene.encode(enc);
    device.queue.submit([enc.finish()]);

    // Spot-check a few ROs' Model fwd: x translation == 1000 + i.
    for (const i of [0, 1, 73, N - 1]) {
      const m = await readConstituent(device, scene, regs[i]!.modelLeaf!.modelPair.fwd);
      expect(m[3]!).toBeCloseTo(1000 + i, 2); // M03
    }
    scene.dispose(); heap.destroy();
  });
});
