import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { Env } from '../../src/core';
import { Box } from '../../src/spaces/box';
import { Autoreset } from '../../src/wrappers';

class ExampleEnv extends Env {
  private count: number;
  constructor() {
    const observationSpace = new Box(0, 3, [1], 'int32');
    const actioSpace = new Box(0, 1, [1], 'float32');
    super(actioSpace, observationSpace, null);
    this.count = 0;
  }

  reset(
    seed: number | undefined,
    options: Record<string, any> | null
  ): [tf.Tensor, Record<string, number>] {
    this.count = 0;
    return [tf.tensor([0]), { count: 0 }];
  }

  async step(
    action: tf.Tensor
  ): Promise<[tf.Tensor, number, boolean, boolean, Record<string, number>]> {
    this.count += 1;
    const biggerThanTwo = this.count > 2;
    return [
      tf.tensor([this.count]),
      Number(biggerThanTwo),
      biggerThanTwo,
      false,
      { count: this.count },
    ];
  }

  async render(): Promise<void> {
    return;
  }

  close(): void {
    return;
  }
}

describe('Test Auto Reset Wrapper', () => {
  const env = new ExampleEnv();
  const autoEnv = new Autoreset(env);

  it('Should reset normally', () => {
    let [obs, info] = autoEnv.reset();

    expect.assert(obs.equal(tf.tensor([0]).dataSync()[0]));
    expect(JSON.stringify(info)).toBe(JSON.stringify({ count: 0 }));
  });

  it('Should not reset on first step', async () => {
    let [obs, reward, terminated, truncated, info] = await autoEnv.step(
      env.actionSpace.sample()
    );

    expect.assert(obs.equal(tf.tensor([1]).dataSync()[0]));
    expect(reward).toBe(0);
    expect.assert(!(terminated || truncated));
    expect(JSON.stringify(info)).toBe(JSON.stringify({ count: 1 }));
  });

  it('Should not reset on second step', async () => {
    let [obs, reward, terminated, truncated, info] = await autoEnv.step(
      env.actionSpace.sample()
    );

    expect.assert(obs.equal(tf.tensor([2]).dataSync()[0]));
    expect(reward).toBe(0);
    expect.assert(!(terminated || truncated));
    expect(JSON.stringify(info)).toBe(JSON.stringify({ count: 2 }));
  });

  it('Should terminate and reset on thrid step', async () => {
    let [obs, reward, terminated, truncated, info] = await autoEnv.step(
      env.actionSpace.sample()
    );

    expect.assert(obs.equal(tf.tensor([3]).dataSync()[0]));
    expect(reward).toBe(1);
    expect.assert(terminated || truncated);
    expect(JSON.stringify(info)).toBe(JSON.stringify({ count: 3 }));
  });

  it('Should have been reseted on previous step', async () => {
    let [obs, reward, terminated, truncated, info] = await autoEnv.step(
      env.actionSpace.sample()
    );

    expect.assert(obs.equal(tf.tensor([0]).dataSync()[0]));
    expect(reward).toBe(0);
    expect.assert(!(terminated || truncated));
    expect(JSON.stringify(info)).toBe(JSON.stringify({ count: 0 }));
  });
});
