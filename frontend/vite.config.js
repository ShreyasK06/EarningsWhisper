import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  plugins: [react()],
  // GitHub Pages serves this repo at /EarningsWhisper/, not the domain
  // root — only apply that base path to the production build so
  // `npm run dev` still serves at / locally.
  base: command === 'build' ? '/EarningsWhisper/' : '/',
}))
