// Real-WebGPU helpers for browser-mode tests.

export async function requestRealDevice(): Promise<GPUDevice> {
  if (!("gpu" in navigator)) {
    throw new Error("navigator.gpu unavailable — WebGPU not enabled in this browser");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no GPUAdapter available");
  return adapter.requestDevice();
}

/**
 * Read every byte of a 2D texture back to the CPU. The texture must
 * have been created with `COPY_SRC` usage. Returns one `Uint8Array`
 * with rows tightly packed (no `bytesPerRow` padding).
 */
export async function readTexturePixels(
  device: GPUDevice,
  texture: GPUTexture,
  bytesPerPixel = 4,
): Promise<Uint8Array> {
  const w = texture.width;
  const h = texture.height;
  const bytesPerRowAligned = Math.ceil((w * bytesPerPixel) / 256) * 256;
  const stagingSize = bytesPerRowAligned * h;
  const staging = device.createBuffer({
    size: stagingSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer(
    { texture },
    { buffer: staging, bytesPerRow: bytesPerRowAligned, rowsPerImage: h },
    { width: w, height: h, depthOrArrayLayers: 1 },
  );
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  // Strip row padding.
  const out = new Uint8Array(w * h * bytesPerPixel);
  for (let y = 0; y < h; y++) {
    out.set(
      padded.subarray(y * bytesPerRowAligned, y * bytesPerRowAligned + w * bytesPerPixel),
      y * w * bytesPerPixel,
    );
  }
  return out;
}
