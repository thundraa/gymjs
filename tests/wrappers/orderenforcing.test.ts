import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { Env } from '../../src/core';
import { Box } from '../../src/spaces/box';
import { OrderEnforcing } from '../../src/wrappers';

class ExampleEnv extends Env<tf.Tensor, tf.Tensor> {
  private count: number;
  constructor() {
    const observationSpace = new Box(0, 3, [1], 'int32');
    const actioSpace = new Box(0, 1, [1], 'float32');
    super(actioSpace, observationSpace, null);
  }

  reset(options?: Record<string, any>): [tf.Tensor, null] {
    return [tf.tensor([0]), null];
  }

  async step(
    action: tf.Tensor
  ): Promise<[tf.Tensor, number, boolean, boolean, null]> {
    this.count += 1;
    const biggerThanTwo = this.count > 2;
    return [tf.tensor([0]), 0, false, false, null];
  }

  async render(): Promise<void> {
    return;
  }

  close(): void {
    return;
  }
}

describe('Test Order Enforcing Wrapper', async () => {
  it('Should specify correctly if it has reseted', () => {
    const enforcedEnv = new OrderEnforcing(new ExampleEnv(), false);
    expect(enforcedEnv.hasReseted).toBe(false);
    enforcedEnv.reset();
    expect(enforcedEnv.hasReseted).toBe(true);
  });

  it('Should not be able to call step before reset', async () => {
    const enforcedEnv = new OrderEnforcing(new ExampleEnv(), false);
    await expect(
      async () => await enforcedEnv.step(enforcedEnv.actionSpace.sample())
    ).rejects.toThrow('Cannot call env.step() before calling env.reset()');
  });

  it('Should not be able to call render before reset if specified', async () => {
    const enforcedEnv = new OrderEnforcing(new ExampleEnv(), false);
    await expect(async () => await enforcedEnv.render()).rejects.toThrow(
      'Cannot call env.render() before calling env.reset(), unset disableRenderOrderEnforcing if this is intended'
    );
  });

  it('Should be able to call render before reset if specified', async () => {
    const enforcedEnv = new OrderEnforcing(new ExampleEnv(), true);
    await expect(async () => await enforcedEnv.render()).not.rejects;
  });

  it('Should be able to call render before reset if specified', async () => {
    const enforcedEnv = new OrderEnforcing(new ExampleEnv(), true);
    enforcedEnv.reset();
    await expect(
      async () => await enforcedEnv.step(enforcedEnv.actionSpace.sample())
    ).not.rejects;
  });

  it('Should be able to call step after reset', async () => {
    const enforcedEnv = new OrderEnforcing(new ExampleEnv(), false);
    enforcedEnv.reset();
    await expect(async () => await enforcedEnv.render()).not.rejects;
  });
});
