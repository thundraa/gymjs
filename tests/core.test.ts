import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import {
  Env,
  Wrapper,
  ObservationWrapper,
  ActionWrapper,
  RewardWrapper,
} from '../src/core';
import { Box, Space } from '../src/spaces';
import { Tensor } from '@tensorflow/tfjs';

class ExampleEnv extends Env {
  constructor() {
    const observationSpace = new Box(0, 1, [1], 'float32');
    const actioSpace = new Box(0, 1, [1], 'float32');
    super(actioSpace, observationSpace, null);
  }

  reset(options?: Record<string, any>): [Tensor, null] {
    return [tf.tensor([0]), null];
  }

  async step(
    action: tf.Tensor
  ): Promise<[tf.Tensor, number, boolean, boolean, null]> {
    let obs = tf.tensor([0]);
    if (action.dataSync()[0] === 1.0) {
      obs = tf.tensor([0.5]);
    }
    return [obs, 0, false, false, null];
  }

  async render(): Promise<void> {
    return;
  }

  close(): void {
    return;
  }
}

describe('Test Env', () => {
  const exampleEnv = new ExampleEnv();
  it('Rendermode should be null', () => {
    expect(exampleEnv.renderMode).toBe(null);
  });
});

class ExampleWrapper extends Wrapper {
  constructor(env: Env | Wrapper) {
    super(env);
  }

  reset(
    options?: Record<string, any>
  ): [tf.Tensor, Record<string, any> | null] {
    return super.reset(options);
  }

  async step(
    action: tf.Tensor | number
  ): Promise<
    [tf.Tensor, number, boolean, boolean, Record<string, any> | null]
  > {
    let [obs, reward, terminated, truncated, info] = await super.step(action);

    return [obs, 3, terminated, truncated, info];
  }
}

class ExampleWrapperDifferent extends Wrapper {
  constructor(env: Env | Wrapper) {
    super(env);
    this._observationSpace = new Box(0, 2, [1], 'float32');
    this._actionSpace = new Box(0, 2, [1], 'float32');
    this._renderMode = 'human';
  }

  reset(
    options?: Record<string, any>
  ): [tf.Tensor, Record<string, any> | null] {
    return super.reset(options);
  }

  async step(
    action: tf.Tensor | number
  ): Promise<
    [tf.Tensor, number, boolean, boolean, Record<string, any> | null]
  > {
    let [obs, reward, terminated, truncated, info] = await super.step(action);

    return [obs, 3, terminated, truncated, info];
  }
}

describe('Test Wrapper', () => {
  const exampleEnv = new ExampleEnv();
  const exampleWrapper = new ExampleWrapper(exampleEnv);
  const exampleWrapperDifferent = new ExampleWrapperDifferent(exampleEnv);
  it('Should have the same render mode', () => {
    expect.assert(exampleEnv.renderMode === exampleWrapper.renderMode);
  });

  it('Should have the same observation space', () => {
    expect.assert(
      exampleEnv.observationSpace.equals(exampleWrapper.observationSpace)
    );
  });

  it('Should have the same action space', () => {
    expect.assert(exampleEnv.actionSpace.equals(exampleWrapper.actionSpace));
  });

  it('Should have different render mode', () => {
    expect.assert(
      !(exampleEnv.renderMode === exampleWrapperDifferent.renderMode)
    );
  });

  it('Should have different observation space', () => {
    expect.assert(
      !exampleEnv.observationSpace.equals(
        exampleWrapperDifferent.observationSpace
      )
    );
  });

  it('Should have different action space', () => {
    expect.assert(
      !exampleEnv.actionSpace.equals(exampleWrapperDifferent.actionSpace)
    );
  });
});

class ExampleRewardWrapper extends RewardWrapper {
  rewardTransform(reward: number): number {
    return 1;
  }
}

class ExampleObservationWrapper extends ObservationWrapper {
  observarionTransform(obs: tf.Tensor): tf.Tensor {
    return tf.tensor([1], [1], 'float32');
  }
}

class ExampleActionWrapper extends ActionWrapper {
  actionTransform(action: tf.Tensor): tf.Tensor {
    return tf.tensor([1], [1], 'float32');
  }
}

describe('Test Specific Wrappers', () => {
  const exampleEnv = new ExampleEnv();
  const exampleRewardWrapper = new ExampleRewardWrapper(exampleEnv);
  const exampleObservationWrapper = new ExampleObservationWrapper(exampleEnv);
  const exampleActionWrapper = new ExampleActionWrapper(exampleEnv);

  exampleEnv.reset();

  it('Reward wrapper should change reward to 1', async () => {
    const [obs, reward, terminated, truncated, info] =
      await exampleRewardWrapper.step(tf.tensor([1]));
    expect(reward).toBe(1);
  });

  it('Observation wrapper should change observation to [1]', async () => {
    const [obs, info] = exampleObservationWrapper.reset();
    expect(obs.dataSync()[0]).toBe(1);
    const [newObs, reward, terminated, truncated, newInfo] =
      await exampleObservationWrapper.step(tf.tensor([1]));
    expect(newObs.dataSync()[0]).toBe(1);
  });

  it('Action wrapper should change to action [1] which changes observation to [0.5]', async () => {
    const [obs, reward, terminated, truncated, info] =
      await exampleActionWrapper.step(tf.tensor([1]));
    expect(obs.dataSync()[0]).toBe(0.5);
  });
});
