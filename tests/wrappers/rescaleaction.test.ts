import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { Env } from '../../src/core';
import { Box } from '../../src/spaces/box';
import { RescaleAction } from '../../src/wrappers';
import { Discrete } from '../../src/spaces';
import { checkTensors } from '../../src/utils';

// An env that returns the exact same reward as step
class ExampleEnv extends Env<tf.Tensor, tf.Tensor> {
  constructor() {
    const observationSpace = new Box(-Infinity, Infinity, [2], 'float32');
    const actionSpace = new Box(
      tf.tensor([0, -1]),
      tf.tensor([10, 1]),
      [2],
      'float32'
    );
    super(actionSpace, observationSpace, null);
  }

  reset(options?: Record<string, any>): [tf.Tensor, null] {
    return [tf.tensor([0]), null];
  }

  // Here we return action as the observation
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

describe('Test Argument Errors', () => {
  const exampleEnv = new ExampleEnv();
  it('Should not accept if minAction or maxAction have different shape compared to parent space', () => {
    expect(() => new RescaleAction(exampleEnv, tf.zeros([1]), 0)).toThrow(
      'minAction should have the same shape as action space!'
    );

    expect(() => new RescaleAction(exampleEnv, 0, tf.zeros([1]))).toThrow(
      'maxAction should have the same shape as action space!'
    );
  });
});
// tf.tensor([-5, 0, 4, -3]), tf.tensor([-2, 1, 10, 0])
describe.each([
  [tf.tensor([0, -1]), tf.tensor([-10, 0])],
  [tf.tensor([10, 1]), tf.tensor([-9, 1])],
  [tf.tensor([5, 0]), tf.tensor([-9.5, 0.5])],
])(
  'Test Valid Action Rescale for sampleAction %i expectedAction %i',
  (sampleAction, expectedAction) => {
    const exampleEnv = new ExampleEnv();
    new Box(tf.tensor([0, -1]), tf.tensor([10, 1]), [2], 'float32');
    const rescaledEnv = new RescaleAction(
      exampleEnv,
      tf.tensor([-10, 0]),
      tf.tensor([-9, 1])
    );
    rescaledEnv.reset();
    it(`Action should be correctly rescaled`, async () => {
      // Example env returns action taken as observation
      const [transformedAction, reward, terminated, truncated, info] =
        await rescaledEnv.step(sampleAction);
      transformedAction.print();
      expectedAction.print();
      expect.assert(checkTensors(transformedAction, expectedAction, true));
    });
  }
);
