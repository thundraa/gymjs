import { defineConfig } from 'tsdown/config';

export default defineConfig({
  entry: [
    './src/index.ts',
    './src/spaces/index.ts',
    './src/envs/classic_control/index.ts',
    './src/envs/classic_control_web/index.ts',
  ],
  minify: {
    compress: true,
    mangle: true,
  },
});
