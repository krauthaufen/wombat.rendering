// Decoder for the line-granular source maps emitted by
// `@aardworx/wombat.shader-ir`'s `buildSourceMap`. Each generated line
// has either zero or one segment; segments encode (genCol=0,
// sourceIdxDelta, sourceLineDelta, sourceColDelta) as base64-VLQ.
//
// We mirror only what `installShaderDiagnostics` needs: given a
// 1-based generated line, return the 1-based originating
// (file, line, col). Heavier source-map work (multi-segment lines,
// names array, sections) isn't needed because we never emit it.

import type { SourceMap } from "@aardworx/wombat.shader/ir";

const VLQ_INDEX = new Map<string, number>();
{
  const a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (let i = 0; i < a.length; i++) VLQ_INDEX.set(a.charAt(i), i);
}

function vlqDecode(s: string): number[] {
  const out: number[] = [];
  let value = 0;
  let shift = 0;
  for (let i = 0; i < s.length; i++) {
    const ch = s.charAt(i);
    const digit = VLQ_INDEX.get(ch);
    if (digit === undefined) throw new Error(`bad VLQ char: ${ch}`);
    const cont = (digit & 0b100000) !== 0;
    value |= (digit & 0b11111) << shift;
    if (cont) {
      shift += 5;
    } else {
      const negative = (value & 1) === 1;
      const magnitude = value >>> 1;
      out.push(negative ? -magnitude : magnitude);
      value = 0;
      shift = 0;
    }
  }
  return out;
}

export interface DecodedPosition {
  /** Source file as listed in `SourceMap.sources`. */
  readonly file: string;
  /** 1-based line in the original source. */
  readonly line: number;
  /** 1-based column in the original source. */
  readonly column: number;
}

/**
 * Look up the originating position for a 1-based generated line. The
 * generated column from the WGSL compiler is ignored because we emit
 * one mapping per line. Returns `null` if the generated line is
 * unmapped or out of range.
 */
export function decodeLine(map: SourceMap, generatedLine: number): DecodedPosition | null {
  const lines = map.mappings.split(";");
  let srcIdx = 0;
  let srcLine = 0;
  let srcCol = 0;
  let last: { idx: number; line: number; col: number } | null = null;

  const target = generatedLine - 1;
  if (target < 0) return null;

  for (let i = 0; i <= target && i < lines.length; i++) {
    const seg = lines[i] ?? "";
    if (seg.length === 0) continue;
    const fields = vlqDecode(seg);
    // [genCol, sourceIdxDelta, sourceLineDelta, sourceColDelta]
    if (fields.length < 4) continue;
    srcIdx += fields[1]!;
    srcLine += fields[2]!;
    srcCol += fields[3]!;
    last = { idx: srcIdx, line: srcLine, col: srcCol };
  }

  if (!last) return null;
  const file = map.sources[last.idx];
  if (file === undefined) return null;
  return { file, line: last.line + 1, column: last.col + 1 };
}
