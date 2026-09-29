import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
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
