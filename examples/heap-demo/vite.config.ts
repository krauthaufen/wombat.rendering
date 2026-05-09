import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { boperators } from "@boperators/plugin-vite";
import { wombatShader } from "@aardworx/wombat.shader-vite";
import { adaptiveMemoPlugin } from "@aardworx/wombat.adaptive/plugin";

const here = fileURLToPath(new URL(".", import.meta.url));

// Tailscale cert + key for HTTPS — WebGPU requires a secure context
// over non-localhost. Path is host-local; absent on dev boxes that
// don't have a cert provisioned, in which case we fall back to HTTP
// (works for localhost-only access).
const certDir = join(homedir(), ".local/share/heap-demo-cert");
const certPath = join(certDir, "airtop.crt");
const keyPath  = join(certDir, "airtop.key");
const httpsConfig = existsSync(certPath) && existsSync(keyPath)
  ? { cert: readFileSync(certPath), key: readFileSync(keyPath) }
  : undefined;

export default defineConfig({
  plugins: [boperators(), wombatShader({ rootDir: here }), adaptiveMemoPlugin()],
  server: {
    host: true,
    port: 5180,
    allowedHosts: [".ts.net", ".loca.lt", "localhost"],
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
