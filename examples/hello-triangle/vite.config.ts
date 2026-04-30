import { defineConfig } from "vite";

export default defineConfig({
  server: { port: 5174 },
  // No special config needed — wombat.shader is consumed via plain
  // ESM imports (parseShader + stage at runtime). The Vite plugin
  // workflow (vertex(...) / fragment(...) markers) is a follow-up.
});
