import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// The wasm runtime of MediaPipe is not reachable through the package `exports` map. Resolve the installed package
// (hoisted to the workspace root or local) and alias its wasm folder.
const mediapipeWasmDir = decodeURIComponent(new URL('./wasm', import.meta.resolve('@mediapipe/tasks-vision')).pathname).replace(
  /^\/([A-Za-z]:)/,
  '$1',
);

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@mediapipe-wasm': mediapipeWasmDir } },
  // The alias id looks like a bare package to the dependency optimizer, which would prebundle these ?url files as JS.
  optimizeDeps: { exclude: ['@mediapipe-wasm/vision_wasm_internal.js?url', '@mediapipe-wasm/vision_wasm_internal.wasm?url'] },
  worker: { format: 'es' },
  build: {
    rolldownOptions: {
      output: {
        codeSplitting: {
          includeDependenciesRecursively: false,
          groups: [
            { name: 'three-core-vendor', test: /[/\\]node_modules[/\\]three[/\\]build[/\\]three\.core\.js$/ },
            { name: 'three-vendor', test: /[/\\]node_modules[/\\]three[/\\]/ },
            { name: 'r3f-vendor', test: /[/\\]node_modules[/\\]@react-three[/\\]fiber[/\\]/ },
            { name: 'drei-vendor', test: /[/\\]node_modules[/\\]@react-three[/\\]drei[/\\]/ },
            { name: 'react-vendor', test: /[/\\]node_modules[/\\](react|react-dom|scheduler)[/\\]/ },
          ],
        },
      },
    },
  },
  server: { port: 5173, strictPort: true },
  test: { exclude: ['e2e/**', 'node_modules/**'] },
});
