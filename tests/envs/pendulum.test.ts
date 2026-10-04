import { describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { PendulumEnv } from '../../src/envs/classic_control/pendulum';

describe('Test Pendulum Angle Normalization', () => {
  it('Reward should use the angle normalized into [-pi, pi)', async () => {
    const env = new PendulumEnv();
    env.reset();
    // theta below -pi: JS `%` does not wrap it into [-pi, pi), NumPy `%` does
    (env as any).state = [-4, 0];

    const [, reward] = await env.step(tf.tensor([0]));

    // costs = angleNormalize(-4) ** 2 = 2.2831853071795862 ** 2
    expect(reward).toBeCloseTo(-5.212935, 5);
  });
});
