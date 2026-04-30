import { defineConfig } from "vite";
import { wombatShader } from "@aardworx/wombat.shader-vite";

export default defineConfig({
  server: { port: 5174 },
  plugins: [
    // Scans `vertex(...)` / `fragment(...)` / `compute(...)`
    // marker calls and inlines them into `__wombat_stage(...)`
    // expressions at build time. Closure captures are baked
    // into IR `ReadInput("Closure", ...)` placeholders.
    wombatShader({
      rootDir: __dirname,
    }),
  ],
});
