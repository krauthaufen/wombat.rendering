// Static-analysis-style guard: no `.force(` call sites in
// wombat.rendering's source under the render path. The legitimate
// exclusions are documented inline.
//
// Why a unit test rather than a lint rule: the constraint is small,
// rare, and easy to assert with a regex over the package source. A
// dedicated linter would be overkill.

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { execSync } from "node:child_process";

const here = dirname(fileURLToPath(import.meta.url));
const srcRoot = resolve(here, "..", "packages", "rendering", "src");

/**
 * Files that may legitimately call `.force()` — bounded "outside
 * adaptive context" boundaries. Keep this list explicit and minimal.
 *
 *   - `preparedRenderObject.ts`: one-time read of the index buffer
 *     view to discover its `indexFormat` at construction. The view
 *     aval is expected to settle on a stable format; per-frame
 *     reads inside `record(token)` use `getValue(token)` instead.
 */
const ALLOWED = new Set<string>([
  "resources/preparedRenderObject.ts",
]);

describe("no .force() on the render path", () => {
  it("scans wombat.rendering src for .force(", () => {
    const out = execSync(
      `grep -RIn --include='*.ts' '\\.force(' ${JSON.stringify(srcRoot)} || true`,
      { encoding: "utf8" },
    );
    const offenders: string[] = [];
    for (const line of out.split("\n")) {
      if (!line.trim()) continue;
      // grep output: "<absolute path>:<lineno>:<text>"
      const m = line.match(/^(.+?):(\d+):(.*)$/);
      if (m === null) continue;
      const file = m[1]!;
      const text = m[3]!;
      if (text.includes("/* allow-force */")) continue;
      const rel = file.startsWith(srcRoot) ? file.slice(srcRoot.length + 1) : file;
      if (ALLOWED.has(rel)) continue;
      offenders.push(`${rel}:${m[2]}:${text.trim()}`);
    }
    expect(offenders, `unexpected .force() call sites:\n${offenders.join("\n")}`).toEqual([]);
  });

  it("preparedRenderObject only forces at construction-boundary spots", () => {
    const file = readFileSync(resolve(srcRoot, "resources", "preparedRenderObject.ts"), "utf8");
    const matches = [...file.matchAll(/\.force\(/g)];
    // Construction-boundary forces, all at prepare-time:
    //   - `vertexAttributesAval.force()` — initial map snapshot used to
    //     decide per-binding step-mode (vertex vs instance) for the
    //     pipeline's vertex layout.
    //   - `obj.instanceAttributes.force()` — same, for the instance map.
    //   - `obj.indices.force()` — index-format discovery from the
    //     initial BufferView.
    // None of these run on the live render path; per-frame reads use
    // `getValue(token)` and chain dependencies through aval.bind / map.
    expect(matches.length).toBe(3);
    expect(file).toMatch(/vertexAttributesAval\.force\(\)/);
    expect(file).toMatch(/obj\.instanceAttributes\.force\(\)/);
    expect(file).toMatch(/obj\.indices\.force\(\)/);
  });
});
