import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',         // root 
  build: {
    outDir: 'dist',  // output folder for Vercel to deploy
    emptyOutDir: true
  }
});
