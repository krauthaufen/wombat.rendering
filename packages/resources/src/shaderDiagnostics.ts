// Shader-compile error reporting. WebGPU's `createShaderModule`
// is synchronous and never throws on parse errors; the errors
// surface only via `module.getCompilationInfo()`, which is async.
// We fire-and-forget that call after creation and log structured
// messages — including a hint at the originating source map — to
// the console.
//
// A perfect line-by-line mapping from WGSL → TS would require a
// full v3 source-map decoder. For now we just surface the source
// map's `sources[]` (the user files involved) and leave the
// per-line lookup to browser dev-tools / external tooling.

import type { SourceMap } from "@aardworx/wombat.shader-ir";

export interface ShaderDiagnosticsOptions {
  /** Human-friendly tag prefixed to log messages. */
  readonly label?: string;
  /** Source map produced by wombat.shader for this stage. */
  readonly sourceMap?: SourceMap | null;
  /** Optional override for the logger. Default: console. */
  readonly logger?: { error(...args: unknown[]): void; warn(...args: unknown[]): void };
}

export function installShaderDiagnostics(
  module: GPUShaderModule,
  source: string,
  opts: ShaderDiagnosticsOptions = {},
): void {
  // getCompilationInfo is part of the WebGPU spec; some test mocks
  // may not implement it. Detect and skip silently.
  const info = (module as { getCompilationInfo?: () => Promise<GPUCompilationInfo> }).getCompilationInfo;
  if (typeof info !== "function") return;
  const log = opts.logger ?? console;
  const tag = opts.label ?? "shader";

  info.call(module).then((compInfo) => {
    if (compInfo.messages.length === 0) return;
    const sources = opts.sourceMap?.sources ?? [];
    const sourcesNote = sources.length > 0
      ? `(originated in ${sources.join(", ")})`
      : "";
    for (const m of compInfo.messages) {
      const lineText = source.split("\n")[Math.max(0, m.lineNum - 1)] ?? "";
      const head = `[${tag}] ${m.type} ${m.lineNum}:${m.linePos}: ${m.message}`;
      const body = lineText.length > 0 ? `\n  ${lineText.trim()}` : "";
      const line = `${head}${body}${sourcesNote ? "\n  " + sourcesNote : ""}`;
      if (m.type === "error") log.error(line);
      else log.warn(line);
    }
  }).catch(() => {
    // Mock environments / older browsers may reject; silent swallow.
  });
}
