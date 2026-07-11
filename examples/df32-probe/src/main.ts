// Planet-scale df32 probe: composes ModelView on the GPU via the §7 derived-
// uniform chain at ECEF magnitudes and reports the absolute error vs f64.
import { AVal, AdaptiveToken } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";
import {
  DerivedUniformsScene, registerRoDerivations, type RoDerivedRequest,
} from "../../../packages/rendering/src/runtime/derivedUniforms/sceneIntegration.js";
import { runMicro } from "./micro.js";
import { STANDARD_DERIVED_RULES } from "../../../packages/rendering/src/runtime/derivedUniforms/recipes.js";

const out = document.getElementById("out")!;
const log = (s: string) => { out.textContent += "\n" + s; };

async function readFloats(device: GPUDevice, buf: GPUBuffer, byteOffset: number, count: number): Promise<Float32Array> {
  const bytes = count * 4;
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, byteOffset, staging, 0, bytes);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const o = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap(); staging.destroy();
  return o;
}

const baseReq = (over: Partial<RoDerivedRequest>): RoDerivedRequest => ({
  rules: new Map(), trafoAvals: new Map(), hostUniformOffset: () => undefined,
  outputOffset: () => undefined, drawHeaderBaseByte: 0, chunkIdx: 0, ...over,
});

(async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter!.requestDevice();
  const scene = new DerivedUniformsScene(device);
  const heap = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  scene.setMainHeapForChunk(0, () => heap);

  const tileT = new V3d(4250557, 857436, 4662583);
  // app-representative model: pure centroid translation (per-part recentring),
  // with a small local rotation FIRST (stays near the origin) — mul is "a then b".
  const model = Trafo3d.rotation(new V3d(0, 0, 1), 0.3).mul(Trafo3d.translation(tileT));
  // camera-style view: wombat.base Trafo3d.mul is Aardvark "a then b" —
  // eyeT.mul(rot) = translate to eye, THEN rotate → forward = R · T(-eye).
  // The composed ModelView is then eye-relative (small entries) like a real
  // render, so collapsing the output to f32 is harmless and errors are mm.
  const eyeT = Trafo3d.translation(new V3d(-(tileT.x + 80), -(tileT.y - 50), -(tileT.z + 60)));
  const view = eyeT.mul(Trafo3d.rotation(new V3d(0, 1, 0), 0.7));

  const reg = registerRoDerivations(scene, {}, baseReq({
    rules: new Map([["ModelViewTrafo", STANDARD_DERIVED_RULES.get("ModelViewTrafo")!]]),
    modelChain: [AVal.constant(model)],
    trafoAvals: new Map([["View", AVal.constant(view)]]),
    outputOffset: (n) => (n === "ModelViewTrafo" ? 0 : undefined),
  }));
  scene.uploadDirty(scene.pullDirty(AdaptiveToken.top));
  const enc = device.createCommandEncoder();
  scene.encode(enc);
  device.queue.submit([enc.finish()]);

  // stage bisect: the chain-composed Model constituent at FULL df32 (hi+lo)
  const rawC = await readFloats(device, scene.constituentsBuf, reg.modelLeaf!.modelPair.fwd * 128, 32);
  const gotModel: number[] = [];
  for (let i = 0; i < 16; i++) gotModel.push(rawC[i * 2]! + rawC[i * 2 + 1]!);
  const wm = model.forward;
  const wModel = [wm.M00, wm.M01, wm.M02, wm.M03, wm.M10, wm.M11, wm.M12, wm.M13,
                  wm.M20, wm.M21, wm.M22, wm.M23, wm.M30, wm.M31, wm.M32, wm.M33];
  let mErr = 0;
  for (let i = 0; i < 16; i++) mErr = Math.max(mErr, Math.abs(gotModel[i]! - wModel[i]!));
  log(`chain-stage Model maxAbsErr=${mErr.toExponential(3)}`);

  // stage bisect: the View constituent as the rule arm reads it (hi+lo)
  const viewAv = AVal.constant(view); // NOT interned with reg's — need reg's slot:
  void viewAv;
  const vPair = (scene.constituents as any).acquire((reg as any).constituentAvals.find((a: any) => a.getValue(AdaptiveToken.top) === view) ?? null);
  const rawV = await readFloats(device, scene.constituentsBuf, vPair.fwd * 128, 32);
  const vw = view.forward;
  const wView = [vw.M00, vw.M01, vw.M02, vw.M03, vw.M10, vw.M11, vw.M12, vw.M13,
                 vw.M20, vw.M21, vw.M22, vw.M23, vw.M30, vw.M31, vw.M32, vw.M33];
  let vErr = 0, vErrHiOnly = 0, vLoMax = 0;
  for (let i = 0; i < 16; i++) {
    vErr = Math.max(vErr, Math.abs((rawV[i * 2]! + rawV[i * 2 + 1]!) - wView[i]!));
    vErrHiOnly = Math.max(vErrHiOnly, Math.abs(rawV[i * 2]! - wView[i]!));
    vLoMax = Math.max(vLoMax, Math.abs(rawV[i * 2 + 1]!));
  }
  log(`View constituent maxAbsErr=${vErr.toExponential(3)} (hi-only ${vErrHiOnly.toExponential(3)}, max|lo|=${vLoMax.toExponential(3)})`);
  log(`vPair fwd=${vPair.fwd} inv=${vPair.inv} modelPair fwd=${reg.modelLeaf!.modelPair.fwd} inv=${reg.modelLeaf!.modelPair.inv}`);
  log(`rawV=${Array.from(rawV).map(v=>v.toPrecision(9)).join(",")}`);

  const got = await readFloats(device, heap, 0, 16);
  const want = view.forward.mul(model.forward);
  const w = [want.M00, want.M01, want.M02, want.M03, want.M10, want.M11, want.M12, want.M13,
             want.M20, want.M21, want.M22, want.M23, want.M30, want.M31, want.M32, want.M33];
  let maxErr = 0, maxI = -1;
  for (let i = 0; i < 16; i++) {
    const e = Math.abs(got[i]! - w[i]!);
    if (e > maxErr) { maxErr = e; maxI = i; }
  }
  log(`maxAbsErr=${maxErr.toExponential(3)} at [${Math.floor(maxI/4)},${maxI%4}]`);
  log(`got=${Array.from(got).map(v=>v.toFixed(4)).join(",")}`);
  log(`want=${w.map(v=>v.toFixed(4)).join(",")}`);
  await runMicro(log);
  (globalThis as any).__probeResult = { maxErr };
})().catch((e) => { log("ERROR " + e.message); (globalThis as any).__probeResult = { error: String(e) }; });
