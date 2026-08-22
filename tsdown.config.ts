import { defineConfig } from 'tsdown/config';

export default defineConfig({
  entry: ['./src/index.ts'],
  format: ['cjs', 'esm'],
  hash: false,
  deps: {
    neverBundle: ['@kmamal/sdl', '@napi-rs/canvas'],
  },
});
