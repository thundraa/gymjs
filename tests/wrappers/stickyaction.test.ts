import { test, expect, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { Env } from '../../src/core';
import { Box } from '../../src/spaces/box';
import { StickyAction } from '../../src/wrappers';

class ExampleEnv extends Env<tf.Tensor, tf.Tensor> {
  constructor() {
    const observationSpace = new Box(0, 1, [1], 'float32');
    const actionSpace = new Box(0, 1, [1], 'float32');
    super(actionSpace, observationSpace, null);
  }

  reset(options?: Record<string, any>): [tf.Tensor, null] {
    return [tf.tensor([0]), null];
  }

  async step(
    action: tf.Tensor
  ): Promise<[tf.Tensor, number, boolean, boolean, null]> {
    return [action, 0, false, false, null];
  }

  async render(): Promise<void> {
    return;
  }

  close(): void {
    return;
  }
}

describe('Test StickyAction Wrapper', () => {
  it('Should throw for invalid probability', () => {
    const env = new ExampleEnv();
    expect(() => new StickyAction(env, -0.1)).toThrow(RangeError);
    expect(() => new StickyAction(env, 1)).toThrow(RangeError);
  });

  it('Should throw for non-integer duration', () => {
    const env = new ExampleEnv();
    expect(() => new StickyAction(env, 0.1, 1.5)).toThrow(TypeError);
    expect(() => new StickyAction(env, 0.1, [1, 1.5])).toThrow(TypeError);
  });

  it('Should throw for invalid duration range', () => {
    const env = new ExampleEnv();
    expect(() => new StickyAction(env, 0.1, [2, 1])).toThrow(RangeError);
    expect(() => new StickyAction(env, 0.1, 0)).toThrow(RangeError);
    expect(() => new StickyAction(env, 0.1, [0, 1])).toThrow(RangeError);
  });

  it('Should not repeat actions when probability is 0', async () => {
    const env = new ExampleEnv();
    const stickyEnv = new StickyAction(env, 0, [1, 1]);
    stickyEnv.reset();

    const [obs1] = await stickyEnv.step(tf.tensor([1]));
    expect(obs1.dataSync()[0]).toBe(1);

    const [obs2] = await stickyEnv.step(tf.tensor([0]));
    expect(obs2.dataSync()[0]).toBe(0);

    const [obs3] = await stickyEnv.step(tf.tensor([0.5]));
    expect(obs3.dataSync()[0]).toBe(0.5);
  });
});
