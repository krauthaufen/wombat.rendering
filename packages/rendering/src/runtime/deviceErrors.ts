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
  /** "out-of-memory", "validation", "internal" or "unknown". */
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

/** Logs and dispatches a device error. */
export function reportDeviceError(device: GPUDevice, error: DeviceError): void {
  const w = watchFor(device);
  w.errors.push(error);
  // keep the record bounded: a broken pass can raise one per frame
  if (w.errors.length > 256) w.errors.splice(0, w.errors.length - 256);
  console.error(`[wombat] GPU device error (${error.kind}${error.recovered ? ", recovered" : ""}): ${error.message}`);
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
