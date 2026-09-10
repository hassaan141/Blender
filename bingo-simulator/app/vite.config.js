import {defineConfig} from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",
  // MuJoCo WASM and onnxruntime-web are dynamically imported from a CDN with
  // @vite-ignore (see game/constants.js). They must NOT be bundled: their .wasm
  // sidecars resolve relative to the importing module's URL, so a bundled copy
  // would look for them in the wrong place and fail at load.
  server: {port: 5173},
  build: {assetsDir: "bundle", chunkSizeWarningLimit: 2000},
});
