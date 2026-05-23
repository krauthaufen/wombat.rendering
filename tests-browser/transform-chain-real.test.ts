// GPU transform propagation — real-GPU verification of the df32 CHAIN arm.
// Composes an ancestor chain on the GPU and checks it equals the CPU f64
// product, plus the fan-out property: changing a shared root marks O(1)
// constituent slots regardless of how many ROs reference it.

import { describe, expect, it } from "vitest";
import { AVal, cval, transact, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, M44d } from "@aardworx/wombat.base";
import {
  DerivedUniformsScene, registerRoDerivations,
} from "../packages/rendering/src/runtime/derivedUniforms/sceneIntegration.js";
import { requestRealDevice } from "./_realGpu.js";

async function readFloats(device: GPUDevice, buf: GPUBuffer, count: number): Promise<Float32Array> {
  const bytes = count * 4;
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, bytes);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

function runChain(scene: DerivedUniformsScene, device: GPUDevice, heap: GPUBuffer): void {
  const dirty = scene.pullDirty(AdaptiveToken.top);
  scene.uploadDirty(dirty);
  const enc = device.createCommandEncoder();
  scene.encode(enc);
  device.queue.submit([enc.finish()]);
}

// Compare GPU row-major output (MainHeap[r*4+c]) to a CPU M44d (M.Mrc).
// The GPU accumulates in df32 (≈f64) but collapses the result to f32 on write,
// so use a RELATIVE tolerance (~f32 output precision) — the df32 accumulation
// is what keeps a naive-f32 chain from losing precision far from the origin.
function expectMatrixClose(gpu: Float32Array, want: M44d): void {
  const w = [
    want.M00, want.M01, want.M02, want.M03,
    want.M10, want.M11, want.M12, want.M13,
    want.M20, want.M21, want.M22, want.M23,
    want.M30, want.M31, want.M32, want.M33,
  ];
  for (let i = 0; i < 16; i++) {
    const tol = 1e-5 * (1 + Math.abs(w[i]!));
    expect(Math.abs(gpu[i]! - w[i]!)).toBeLessThanOrEqual(tol);
  }
}

describe("GPU transform-propagation chain (real GPU)", () => {
  it("composes link0·link1·link2 in df32 == CPU f64 product", async () => {
    const device = await requestRealDevice();
    const scene = new DerivedUniformsScene(device);
    const heap = device.createBuffer({
      size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    scene.setMainHeapForChunk(0, () => heap);

    const a = Trafo3d.translation(new V3d(1, 2, 3));
    const b = Trafo3d.rotation(new V3d(0, 0, 1), 0.6);
    const c = Trafo3d.translation(new V3d(-4, 0.5, 2));
    registerRoDerivations(scene, {}, {
      rules: new Map(),
      chains: new Map([["ModelTrafo", [AVal.constant(a), AVal.constant(b), AVal.constant(c)]]]),
      trafoAvals: new Map(),
      hostUniformOffset: () => undefined,
      outputOffset: () => 0,
      drawHeaderBaseByte: 0,
      chunkIdx: 0,
    });

    runChain(scene, device, heap);
    const out = await readFloats(device, heap, 16);
    expectMatrixClose(out, a.forward.mul(b.forward).mul(c.forward));
    scene.dispose(); heap.destroy();
  });

  it("a deep chain stays df32-accurate (far-from-origin)", async () => {
    const device = await requestRealDevice();
    const scene = new DerivedUniformsScene(device);
    const heap = device.createBuffer({
      size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    scene.setMainHeapForChunk(0, () => heap);

    const links: Trafo3d[] = [];
    let want = Trafo3d.identity.forward; // M44d identity; compose in M44d order to match the GPU
    for (let i = 0; i < 8; i++) {
      const t = Trafo3d.translation(new V3d(100000 + i * 13, i * 7, -i * 5)).mul(Trafo3d.rotation(new V3d(0, 1, 0), 0.1 * i));
      links.push(t);
      want = want.mul(t.forward);
    }
    registerRoDerivations(scene, {}, {
      rules: new Map(),
      chains: new Map([["ModelTrafo", links.map((t) => AVal.constant(t))]]),
      trafoAvals: new Map(),
      hostUniformOffset: () => undefined,
      outputOffset: () => 0,
      drawHeaderBaseByte: 0,
      chunkIdx: 0,
    });
    runChain(scene, device, heap);
    const out = await readFloats(device, heap, 16);
    // df32 holds ~double precision; even at 1e5 translations the f32 readback matches.
    expectMatrixClose(out, want);
    scene.dispose(); heap.destroy();
  });

  it("changing a shared root marks O(1) constituent slots (no fan-out)", async () => {
    const device = await requestRealDevice();
    const scene = new DerivedUniformsScene(device);
    const N = 200;
    const heap = device.createBuffer({
      size: N * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    scene.setMainHeapForChunk(0, () => heap);

    const root = cval(Trafo3d.translation(new V3d(0, 0, 0)));
    for (let i = 0; i < N; i++) {
      registerRoDerivations(scene, {}, {
        rules: new Map(),
        chains: new Map([["ModelTrafo", [root, AVal.constant(Trafo3d.translation(new V3d(i, 0, 0)))]]]),
        trafoAvals: new Map(),
        hostUniformOffset: () => undefined,
        outputOffset: () => i * 64, // 16 floats per RO
        drawHeaderBaseByte: 0,
        chunkIdx: 0,
      });
    }
    runChain(scene, device, heap); // initial

    // Change the one root cval shared by all N ROs.
    transact(() => { root.value = Trafo3d.translation(new V3d(1000, 0, 0)); });
    scene.routeInputChanged(root);                 // heap would route this
    const dirty = scene.pullDirty(AdaptiveToken.top);
    expect(dirty.size).toBe(2);                     // root fwd+inv ONLY — not 2·N
    scene.uploadDirty(dirty);
    const enc = device.createCommandEncoder();
    scene.encode(enc);
    device.queue.submit([enc.finish()]);

    // Every RO's ModelTrafo translation column now reflects the new root.
    const out = await readFloats(device, heap, N * 16);
    for (let i = 0; i < N; i++) {
      const want = root.value.forward.mul(Trafo3d.translation(new V3d(i, 0, 0)).forward);
      const base = i * 16;
      expect(out[base + 3]!).toBeCloseTo(want.M03, 2);  // x translation
    }
    scene.dispose(); heap.destroy();
  });
});
