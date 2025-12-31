import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { Env } from '../../src/core';
import { Box } from '../../src/spaces/box';
import { ClipReward } from '../../src/wrappers';
import { Discrete } from '../../src/spaces';

// An env that returns the exact same reward as step
class ExampleEnv extends Env<tf.Tensor, number> {
  constructor() {
    const observationSpace = new Box(0, 1, [1], 'float32');
    const actioSpace = new Discrete(10, -5);
    super(actioSpace, observationSpace, null);
  }

  reset(options?: Record<string, any>): [tf.Tensor, null] {
    return [tf.tensor([0]), null];
  }

  async step(
    action: number
  ): Promise<[tf.Tensor, number, boolean, boolean, null]> {
    return [tf.tensor([0]), action, false, false, null];
  }

  async render(): Promise<void> {
    return;
  }

  close(): void {
    return;
  }
}

describe('Test Argument Errors', () => {
  const exampleEnv = new ExampleEnv();
  it('Should not accept if Both minReward and maxReward are null', () => {
    expect(() => new ClipReward(exampleEnv, null, null)).toThrow(
      'Both `minReward` and `maxReward` cannot be null'
    );
  });

  it('Should not accept if Both minReward is less than maxReward', () => {
    const minReward = 1;
    const maxReward = 0;
    expect(() => new ClipReward(exampleEnv, minReward, maxReward)).toThrow(
      `Min reward (${minReward}) must be smaller than max reward (${maxReward})`
    );
  });
});

describe.each([
  [0, null, -1, 0],
  [0, null, 1, 1],
  [null, 0, 1, 0],
  [null, 0, -1, -1],
  [0, 2, 1, 1],
  [0, 2, 3, 2],
  [0, 2, -1, 0],
])(
  'Test Valid Clip for min %i max %i originalReward %i expectedReward %i',
  (min, max, originalReward, expectedReward) => {
    const exampleEnv = new ExampleEnv();
    const clipedEnv = new ClipReward(exampleEnv, min, max);
    clipedEnv.reset();
    it(`Reward should be correctly clipped`, async () => {
      // Example env returns the exact same reward as step
      const [obs, reward, terminated, truncated, info] =
        await clipedEnv.step(originalReward);
      expect(reward).toBe(expectedReward);
    });
  }
);
