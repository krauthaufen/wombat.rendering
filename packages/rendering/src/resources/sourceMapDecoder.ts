// Decoder for v3 source maps. Supports both line-granular maps
// (one segment per generated line at col 0) and multi-segment
// lines (per-Expr granularity). Segments encode
// (genColDelta, sourceIdxDelta, sourceLineDelta, sourceColDelta)
// as base64-VLQ, with deltas accumulating across the whole map
// except for genCol which resets at every `;` line boundary.
//
// `decodeLine(map, line)` returns the position for the *first*
// segment on the line; `decodePosition(map, line, col)` picks the
// closest preceding segment. Both fall back to the most recent
// mapped line if the queried line is unmapped, which is what
// `installShaderDiagnostics` wants for "go to source" navigation.

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

interface DecodedSegment {
  readonly genCol: number;
  readonly idx: number;
  readonly line: number;
  readonly col: number;
}

/** Walk the map up to (and including) `targetLine`, returning each line's segments. */
function segmentsUpTo(map: SourceMap, targetLine: number): DecodedSegment[][] {
  const lines = map.mappings.split(";");
  const out: DecodedSegment[][] = [];
  let srcIdx = 0;
  let srcLine = 0;
  let srcCol = 0;
  for (let i = 0; i <= targetLine && i < lines.length; i++) {
    const segs = (lines[i] ?? "").split(",").filter(s => s.length > 0);
    let prevGenCol = 0;
    const lineOut: DecodedSegment[] = [];
    for (const sStr of segs) {
      const fields = vlqDecode(sStr);
      if (fields.length < 4) continue;
      const genCol = prevGenCol + fields[0]!;
      srcIdx += fields[1]!;
      srcLine += fields[2]!;
      srcCol += fields[3]!;
      lineOut.push({ genCol, idx: srcIdx, line: srcLine, col: srcCol });
      prevGenCol = genCol;
    }
    out.push(lineOut);
  }
  return out;
}

function makePosition(map: SourceMap, s: DecodedSegment): DecodedPosition | null {
  const file = map.sources[s.idx];
  if (file === undefined) return null;
  return { file, line: s.line + 1, column: s.col + 1 };
}

/**
 * Closest-preceding-segment lookup. Given a 1-based generated
 * `line` + 0-based generated `col`, returns the originating source
 * position. Falls back to the last segment on a previous mapped
 * line if `line` itself has no segments.
 */
export function decodePosition(
  map: SourceMap,
  generatedLine: number,
  generatedCol = 0,
): DecodedPosition | null {
  const target = generatedLine - 1;
  if (target < 0) return null;
  const allLines = segmentsUpTo(map, target);
  // Prefer a segment on the target line at or before generatedCol.
  const onTarget = allLines[target];
  if (onTarget && onTarget.length > 0) {
    let pick: DecodedSegment | undefined;
    for (const s of onTarget) {
      if (s.genCol <= generatedCol) pick = s;
      else break;
    }
    if (pick) return makePosition(map, pick);
    return makePosition(map, onTarget[0]!);
  }
  // Fallback: last segment on the most recent mapped line.
  for (let i = target - 1; i >= 0; i--) {
    const segs = allLines[i];
    if (segs && segs.length > 0) return makePosition(map, segs[segs.length - 1]!);
  }
  return null;
}

/**
 * Backwards-compatible line-only decoder: returns the *first*
 * segment of `generatedLine`, or the most recent mapped line for
 * unmapped lines. Equivalent to `decodePosition(map, line, 0)`.
 */
export function decodeLine(map: SourceMap, generatedLine: number): DecodedPosition | null {
  return decodePosition(map, generatedLine, 0);
}
