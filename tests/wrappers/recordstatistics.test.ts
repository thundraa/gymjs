import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { Env } from '../../src/core';
import { Box } from '../../src/spaces/box';
import { RecordEpisodeStatistics } from '../../src/wrappers';

class ExampleEnv extends Env {
  private count: number;
  constructor() {
    const observationSpace = new Box(0, 1, [1], 'float32');
    const actioSpace = new Box(0, 1, [1], 'float32');
    super(actioSpace, observationSpace, null);
    this.count = 0;
  }

  reset(options?: Record<string, any>): [tf.Tensor, null] {
    this.count = 0;
    return [tf.tensor([0]), null];
  }

  async step(
    action: tf.Tensor
  ): Promise<[tf.Tensor, number, boolean, boolean, null]> {
    this.count += 1;
    const terminated = this.count >= 5 ? true : false;
    console.log(terminated);
    return [tf.tensor([0]), 1, terminated, false, null];
  }

  async render(): Promise<void> {
    return;
  }

  close(): void {
    return;
  }
}

describe('Test Record Episode Statistics Wrapper', async () => {
  const recordedEnv = new RecordEpisodeStatistics(new ExampleEnv());
  const [_obs, _info] = recordedEnv.reset();

  for (let i = 0; i < 4; i++) {
    await recordedEnv.step(recordedEnv.actionSpace.sample());
  }
  const [obs, reward, terminated, truncated, info] = await recordedEnv.step(
    recordedEnv.actionSpace.sample()
  );

  it('Info should exist and have the correct properies', () => {
    expect(info).not.toBe(null);
    expect(info).toHaveProperty('episode');
    // @ts-ignore
    expect(info['episode']).toHaveProperty('length');
    // @ts-ignore
    expect(info['episode']).toHaveProperty('rewards');
    // @ts-ignore
    expect(info['episode']).toHaveProperty('time');
    // @ts-ignore
    expect(info['episode']['length']).toBe(5);
    // @ts-ignore
    expect(info['episode']['rewards']).toBe(5);
  });
});
