// Device errors, made visible.
//
// WebGPU reports most failures (out of memory, validation) only as
// `uncapturederror` events on the device: without a listener a failed
// allocation leaves an invalid object behind and the page simply draws
// nothing, with no trace in the console. `watchDeviceErrors` installs one
// listener per device that logs every such error once (`console.error`)
// and hands it to subscribers (wombat.dom's render control shows a note on
// the page). Errors the renderer catches and recovers from itself (the
// texture atlas running out of memory while growing) are reported through
// `reportDeviceError` the same way, marked `recovered`.

/** One device error. */
export interface DeviceError {
  /** "out-of-memory", "validation", "internal", "unknown" (uncaptured device
   *  errors), "device-lost", "frame" (an exception in a frame callback), or
   *  "scene" (a draw left out of the scene, e.g. one missing an attribute). */
  readonly kind: string;
  readonly message: string;
  /** The renderer recovered from it (the scene keeps rendering, maybe degraded). */
  readonly recovered: boolean;
}

interface Watch {
  readonly listeners: Set<(e: DeviceError) => void>;
  readonly errors: DeviceError[];
}

const watches = new WeakMap<GPUDevice, Watch>();

function kindOf(error: unknown): string {
  if (typeof GPUOutOfMemoryError !== "undefined" && error instanceof GPUOutOfMemoryError) return "out-of-memory";
  if (typeof GPUValidationError !== "undefined" && error instanceof GPUValidationError) return "validation";
  if (typeof GPUInternalError !== "undefined" && error instanceof GPUInternalError) return "internal";
  const name = (error as { constructor?: { name?: string } } | null)?.constructor?.name ?? "";
  if (/OutOfMemory/i.test(name)) return "out-of-memory";
  if (/Validation/i.test(name)) return "validation";
  if (/Internal/i.test(name)) return "internal";
  return "unknown";
}

function watchFor(device: GPUDevice): Watch {
  let w = watches.get(device);
  if (w === undefined) {
    w = { listeners: new Set(), errors: [] };
    watches.set(device, w);
  }
  return w;
}

/** The limits that most often explain a scene that renders on one GPU and
 *  not another (texture and buffer sizes, binding counts). */
const LIMITS = [
  "maxTextureDimension2D", "maxTextureArrayLayers", "maxBufferSize", "maxStorageBufferBindingSize",
  "maxStorageBuffersPerShaderStage", "maxSampledTexturesPerShaderStage", "maxBindGroups",
  "maxUniformBufferBindingSize", "maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup",
] as const;

/** The device's adapter (vendor, architecture, description) and its
 *  key limits, as one line: what to quote next to an error report. */
export function describeDevice(device: GPUDevice): string {
  const info = (device as { adapterInfo?: { vendor?: string; architecture?: string; device?: string; description?: string } }).adapterInfo;
  const who = info === undefined
    ? "adapter info unavailable"
    : [info.vendor, info.architecture, info.device, info.description].filter((x) => x !== undefined && x !== "").join(" / ") || "adapter info empty";
  const limits = (device as unknown as { limits?: Record<string, number> }).limits;
  const mb = (n: number) => (n >= 1 << 20 ? `${Math.round(n / (1 << 20))} MB` : String(n));
  const lim = limits === undefined ? "no limits" : LIMITS
    .filter((k) => typeof limits[k] === "number")
    .map((k) => `${k} ${/Size$/.test(k) && !/Workgroup/.test(k) ? mb(limits[k]!) : limits[k]}`)
    .join(", ");
  const mem = (globalThis as { navigator?: { deviceMemory?: number } }).navigator?.deviceMemory;
  return `${who}; ${lim}${mem !== undefined ? `; deviceMemory ${mem} GB` : ""}`;
}

/** Logs and dispatches a device error (`log` false: the caller logged it).
 *  The first error of a device also logs the device (`describeDevice`). */
export function reportDeviceError(device: GPUDevice, error: DeviceError, log = true): void {
  const w = watchFor(device);
  if (w.errors.length === 0) {
    try { console.error(`[wombat] GPU device: ${describeDevice(device)}`); } catch { /* best effort */ }
  }
  w.errors.push(error);
  // keep the record bounded: a broken pass can raise one per frame
  if (w.errors.length > 256) w.errors.splice(0, w.errors.length - 256);
  if (log) console.error(`[wombat] GPU device error (${error.kind}${error.recovered ? ", recovered" : ""}): ${error.message}`);
  for (const l of w.listeners) {
    try { l(error); } catch (e) { console.error("[wombat] device error listener threw", e); }
  }
}

/** Installs the device's `uncapturederror` listener (once per device). */
export function watchDeviceErrors(device: GPUDevice): void {
  const w = watchFor(device);
  if ((w as { installed?: boolean }).installed) return;
  (w as { installed?: boolean }).installed = true;
  const target = device as unknown as { addEventListener?: (type: string, cb: (e: Event) => void) => void };
  if (typeof target.addEventListener !== "function") return; // mock devices
  target.addEventListener("uncapturederror", (e: Event) => {
    const error = (e as unknown as { error?: { message?: string } }).error;
    reportDeviceError(device, { kind: kindOf(error), message: error?.message ?? String(error), recovered: false });
  });
}

/** Subscribes to the device's errors (those so far are not replayed: see `deviceErrors`). */
export function onDeviceError(device: GPUDevice, cb: (e: DeviceError) => void): { dispose(): void } {
  const w = watchFor(device);
  w.listeners.add(cb);
  return { dispose: () => { w.listeners.delete(cb); } };
}

/** The device's errors so far (the last 256). */
export function deviceErrors(device: GPUDevice): readonly DeviceError[] {
  return watchFor(device).errors;
}
