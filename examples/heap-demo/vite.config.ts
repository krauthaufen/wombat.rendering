import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { boperators } from "@boperators/plugin-vite";
import { wombatShader } from "@aardworx/wombat.shader-vite";

const here = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  plugins: [boperators(), wombatShader({ rootDir: here })],
  server: {
    host: true,            // bind 0.0.0.0 so Tailscale / LAN clients can reach the dev server
    port: 5180,
    allowedHosts: [".ts.net", ".loca.lt", "localhost"],
  },
  define: {
    global: "globalThis",
  },
  optimizeDeps: {
    esbuildOptions: {
      define: { global: "globalThis" },
    },
  },
});
