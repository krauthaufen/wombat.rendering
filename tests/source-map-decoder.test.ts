// Round-trip test: build a line-granular source map with the IR's
// `buildSourceMap`, then resolve generated lines back through
// `decodeLine`. The decoder is what `installShaderDiagnostics` uses
// to translate WGSL compile errors into TS source positions, so we
// pin the round-trip behaviour here.

import { describe, it, expect } from "vitest";
import { buildSourceMap, type Span } from "@aardworx/wombat.shader-ir";
import { decodeLine } from "../packages/resources/src/sourceMapDecoder.js";

const fileA = "a.ts";
const fileB = "b.ts";

const contentsA = "line0\nline1 longer\nline2\nline3\n";
//                 0     6           19    25

const contentsB = "first\nsecond\nthird\n";
//                 0     6      13

const fileContents = new Map([
  [fileA, contentsA],
  [fileB, contentsB],
]);

function span(file: string, start: number, end: number = start): Span {
  return { file, start, end };
}

describe("sourceMapDecoder", () => {
  it("decodes mapped lines back to (file, line, col)", () => {
    const map = buildSourceMap({
      lineSpans: [
        span(fileA, 0),       // → a.ts:1:1
        span(fileA, 19),      // → a.ts:3:1
        span(fileB, 6),       // → b.ts:2:1
      ],
      fileContents,
    });

    expect(decodeLine(map, 1)).toEqual({ file: fileA, line: 1, column: 1 });
    expect(decodeLine(map, 2)).toEqual({ file: fileA, line: 3, column: 1 });
    expect(decodeLine(map, 3)).toEqual({ file: fileB, line: 2, column: 1 });
  });

  it("falls back to the most recent mapped line for unmapped lines", () => {
    const map = buildSourceMap({
      lineSpans: [
        span(fileA, 0),
        undefined, // unmapped
        undefined,
        span(fileA, 19),
      ],
      fileContents,
    });

    expect(decodeLine(map, 1)).toEqual({ file: fileA, line: 1, column: 1 });
    // Unmapped lines fall back to the previous mapped position.
    expect(decodeLine(map, 2)).toEqual({ file: fileA, line: 1, column: 1 });
    expect(decodeLine(map, 3)).toEqual({ file: fileA, line: 1, column: 1 });
    expect(decodeLine(map, 4)).toEqual({ file: fileA, line: 3, column: 1 });
  });

  it("returns null for unmapped prefix and out-of-range lines", () => {
    const map = buildSourceMap({
      lineSpans: [undefined, span(fileA, 0)],
      fileContents,
    });

    expect(decodeLine(map, 1)).toBeNull();
    expect(decodeLine(map, 2)).toEqual({ file: fileA, line: 1, column: 1 });
    expect(decodeLine(map, 99)).toEqual({ file: fileA, line: 1, column: 1 });
  });

  it("decodes columns inside a line", () => {
    const map = buildSourceMap({
      lineSpans: [span(fileA, 6 + 6) /* "longer" inside line1 */],
      fileContents,
    });
    expect(decodeLine(map, 1)).toEqual({ file: fileA, line: 2, column: 7 });
  });
});
