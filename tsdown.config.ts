import { defineConfig } from 'tsdown/config';

export default defineConfig({
  entry: [
    './src/index.ts',
    './src/spaces/index.ts',
    './src/wrappers/index.ts',
    './src/envs/classic_control/index.ts',
  ],
  hash: false,
});
