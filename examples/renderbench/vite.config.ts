import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { wombatShader } from "@aardworx/wombat.shader-vite";
import { adaptiveMemoPlugin } from "@aardworx/wombat.adaptive/plugin";

const here = fileURLToPath(new URL(".", import.meta.url));

// Tailscale cert (same provisioning as heap-demo) — WebGPU needs a secure
// context over non-localhost, i.e. for phone access via the tailnet.
const certDir = join(homedir(), ".local/share/heap-demo-cert");
const certPath = join(certDir, "airtop.crt");
const keyPath  = join(certDir, "airtop.key");
const httpsConfig = existsSync(certPath) && existsSync(keyPath)
  ? { cert: readFileSync(certPath), key: readFileSync(keyPath) }
  : undefined;

export default defineConfig({
  plugins: [wombatShader({ rootDir: here }), adaptiveMemoPlugin()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    allowedHosts: [".ts.net", "localhost"],
    ...(httpsConfig !== undefined ? { https: httpsConfig } : {}),
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
