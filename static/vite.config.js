import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',   // Where Vite will output the production files
    emptyOutDir: true // Remove old files before each build
  }
});
