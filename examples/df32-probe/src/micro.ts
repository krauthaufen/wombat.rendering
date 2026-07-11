// df32 micro-probe: one MV translation element computed three ways —
// (a) GPU WGSL df32 (the DF32_LIB verbatim), (b) bit-exact JS emulation of
// the same algorithm (fround after every op), (c) f64 reference.
import { Trafo3d, V3d } from "@aardworx/wombat.base";

const F = Math.fround;
type DF = [number, number];
const split12 = (a: number): DF => {
  const buf = new DataView(new ArrayBuffer(4));
  buf.setFloat32(0, a);
  buf.setUint32(0, buf.getUint32(0) & 0xFFFFF000);
  const hi = buf.getFloat32(0);
  return [hi, F(a - hi)];
};
const two_sum = (a: number, b: number): DF => {
  const s = F(a + b);
  const bb = F(s - a);
  const t1 = F(s - bb);
  const t2 = F(a - t1);
  const t3 = F(b - bb);
  return [s, F(t2 + t3)];
};
const quick_two_sum = (a: number, b: number): DF => {
  const s = F(a + b);
  const t = F(s - a);
  return [s, F(b - t)];
};
const two_prod = (a: number, b: number): DF => {
  const p = F(a * b);
  const A = split12(a), B = split12(b);
  const err = F(F(F(F(A[0] * B[0]) - p) + F(A[0] * B[1]) + F(A[1] * B[0])) + F(A[1] * B[1]));
  return [p, err];
};
const df_add = (a: DF, b: DF): DF => {
  const s = two_sum(a[0], b[0]);
  const t = two_sum(a[1], b[1]);
  const s3 = quick_two_sum(s[0], F(s[1] + t[0]));
  return quick_two_sum(s3[0], F(s3[1] + t[1]));
};
const df_mul = (a: DF, b: DF): DF => {
  const p = two_prod(a[0], b[0]);
  const cross1 = F(F(a[0] * b[1]) + p[1]);   // unfused-fma emulation
  const cross = F(F(a[1] * b[0]) + cross1);
  return quick_two_sum(p[0], cross);
};
const toDF = (v: number): DF => { const hi = F(v); return [hi, F(v - hi)]; };

export async function runMicro(log: (s: string) => void): Promise<void> {
  const tileT = new V3d(4250557, 857436, 4662583);
  const model = Trafo3d.translation(tileT).mul(Trafo3d.rotation(new V3d(0, 0, 1), 0.3));
  const eyeT = Trafo3d.translation(new V3d(-(tileT.x + 80), -(tileT.y - 50), -(tileT.z + 60)));
  const view = Trafo3d.rotation(new V3d(0, 1, 0), 0.7).mul(eyeT);
  const V = view.forward, M = model.forward;
  const r = 2, c = 3;
  const Vrow = [V.M20, V.M21, V.M22, V.M23];
  const Mcol = [M.M03, M.M13, M.M23, M.M33];

  // f64 reference
  const want = Vrow[0]*Mcol[0] + Vrow[1]*Mcol[1] + Vrow[2]*Mcol[2] + Vrow[3]*Mcol[3];

  // JS df32 emulation
  let acc: DF = [0, 0];
  for (let t = 0; t < 4; t++) acc = df_add(acc, df_mul(toDF(Vrow[t]!), toDF(Mcol[t]!)));
  const jsRes = acc[0] + acc[1];

  // GPU: same values through the DF32_LIB verbatim
  const DF32_LIB = `
fn split12(a: f32) -> vec2<f32> {
  let hi = bitcast<f32>(bitcast<u32>(a) & 0xFFFFF000u);
  return vec2<f32>(hi, a - hi);
}
fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let s  = a + b;
  let bb = fma(1.0, s, -a);
  let t1 = fma(1.0, s, -bb);
  let t2 = fma(1.0, a, -t1);
  let t3 = fma(1.0, b, -bb);
  return vec2<f32>(s, t2 + t3);
}
fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = a + b;
  let t = fma(1.0, s, -a);
  return vec2<f32>(s, fma(1.0, b, -t));
}
fn two_prod(a: f32, b: f32) -> vec2<f32> {
  let p = a * b;
  let A = split12(a);
  let B = split12(b);
  let err = ((A.x * B.x - p) + A.x * B.y + A.y * B.x) + A.y * B.y;
  return vec2<f32>(p, err);
}
fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let s = two_sum(a.x, b.x);
  let t = two_sum(a.y, b.y);
  let s3 = quick_two_sum(s.x, s.y + t.x);
  return quick_two_sum(s3.x, s3.y + t.y);
}
fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let p      = two_prod(a.x, b.x);
  let cross1 = fma(a.x, b.y, p.y);
  let cross  = fma(a.y, b.x, cross1);
  return quick_two_sum(p.x, cross);
}
`;
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter!.requestDevice();
  const inBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const inData = new Float32Array(16);
  for (let t = 0; t < 4; t++) {
    const dv = toDF(Vrow[t]!), dm = toDF(Mcol[t]!);
    inData[t * 4] = dv[0]; inData[t * 4 + 1] = dv[1];
    inData[t * 4 + 2] = dm[0]; inData[t * 4 + 3] = dm[1];
  }
  device.queue.writeBuffer(inBuf, 0, inData);
  const mod = device.createShaderModule({ code: `
${DF32_LIB}
@group(0) @binding(0) var<storage, read> IN: array<vec4<f32>, 4>;
@group(0) @binding(1) var<storage, read_write> OUT: vec2<f32>;
@compute @workgroup_size(1)
fn main() {
  var acc = vec2<f32>(0.0, 0.0);
  for (var t: u32 = 0u; t < 4u; t = t + 1u) {
    let v = vec2<f32>(IN[t].x, IN[t].y);
    let m = vec2<f32>(IN[t].z, IN[t].w);
    acc = df_add(acc, df_mul(v, m));
  }
  OUT = acc;
}` });
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: inBuf } },
    { binding: 1, resource: { buffer: outBuf } },
  ]});
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(1); pass.end();
  const staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  enc.copyBufferToBuffer(outBuf, 0, staging, 0, 8);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const gpu = new Float32Array(staging.getMappedRange().slice(0));
  const gpuRes = gpu[0]! + gpu[1]!;

  log(`micro[2,3]: f64=${want.toFixed(6)} js-df32=${jsRes.toFixed(6)} (err ${(jsRes-want).toExponential(2)}) gpu-df32=${gpuRes.toFixed(6)} (err ${(gpuRes-want).toExponential(2)})`);
  log(`micro slot-repr err (toDF hi+lo vs f64): ${(inData.slice(0,4).length, [0,1,2,3].map(t => Math.abs((toDF(Vrow[t]!)[0]+toDF(Vrow[t]!)[1])-Vrow[t]!)).reduce((a,b)=>Math.max(a,b),0)).toExponential(2)}`);
}
