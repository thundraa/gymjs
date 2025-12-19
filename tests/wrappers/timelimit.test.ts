import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { Env } from '../../src/core';
import { Box } from '../../src/spaces/box';
import { TimeLimit } from '../../src/wrappers';

class ExampleEnv extends Env {
  constructor() {
    const observationSpace = new Box(0, 1, [1], 'float32');
    const actioSpace = new Box(0, 1, [1], 'float32');
    super(actioSpace, observationSpace, null);
  }

  reset(options?: Record<string, any>): [tf.Tensor, null] {
    return [tf.tensor([0]), null];
  }

  async step(
    action: tf.Tensor
  ): Promise<[tf.Tensor, number, boolean, boolean, null]> {
    return [tf.tensor([0]), 0, false, false, null];
  }

  async render(): Promise<void> {
    return;
  }

  close(): void {
    return;
  }
}

describe('Test Time Limit Reset (Info and Obs)', () => {
  const env = new ExampleEnv();
  const limitedEnv = new TimeLimit(env, 100);
  const [obs, info] = limitedEnv.reset();

  it('Should have the same observation space', () => {
    expect.assert(env.observationSpace.equals(limitedEnv.observationSpace));
  });

  it('Observation should be in the observation space', () => {
    expect.assert(env.observationSpace.contains(obs));
  });

  it('Should have the same info (null)', () => {
    expect.assert(info === null);
  });
});

describe('Test Time Limit Wrapper', () => {
  const env = new ExampleEnv();

  it('Should truncate automatically after the maximum steps', async () => {
    const maxEpisodeLength = 20;
    const limitedEnv = new TimeLimit(env, maxEpisodeLength);

    limitedEnv.reset();

    let terminated = false;
    let truncated = false;

    let nSteps = 0;

    while (!(terminated || truncated)) {
      nSteps += 1;
      let [obs, reward, nextTerminated, nextTruncated, info] =
        await limitedEnv.step(env.actionSpace.sample());
      terminated = nextTerminated;
      truncated = nextTruncated;
    }

    expect(nSteps).toBe(maxEpisodeLength);
    expect.assert(truncated);
  });

  it('Should truncate on last step', async () => {
    const maxEpisodeLength = 1;
    const limitedEnv = new TimeLimit(env, maxEpisodeLength);

    limitedEnv.reset();
    // Change step to always terminate
    env.step = async (action: tf.Tensor) => [
      tf.tensor([0]),
      0,
      true,
      false,
      null,
    ];

    let terminated = false;
    let truncated = false;

    let [obs, reward, nextTerminated, nextTruncated, info] =
      await limitedEnv.step(env.actionSpace.sample());

    terminated = nextTerminated;
    truncated = nextTruncated;

    expect.assert(terminated);
    expect.assert(truncated);
  });
});
