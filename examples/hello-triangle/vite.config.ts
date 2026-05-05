import { defineConfig } from "vite";
import { wombatShader } from "@aardworx/wombat.shader-vite";
import { boperators } from "@boperators/plugin-vite";

export default defineConfig({
  server: { port: 5174 },
  plugins: [
    boperators(),
    // Scans `vertex(...)` / `fragment(...)` / `compute(...)`
    // marker calls and inlines them into `__wombat_stage(...)`
    // expressions at build time. Closure captures are baked
    // into IR `ReadInput("Closure", ...)` placeholders.
    wombatShader({
      rootDir: __dirname,
    }),
  ],
});
