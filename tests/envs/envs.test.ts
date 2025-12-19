import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { CartPoleEnv, PendulumEnv } from '../../src/envs/classic_control';
import { Env } from '../../src/core';

describe.each([
  [CartPoleEnv, 'CartPoleEnv'],
  [PendulumEnv, 'PendulumEnv'],
])('Testing Environment $name', (env, name) => {
  it(`${name} should construct successfully`, async () => {
    expect(() => new env()).not.toThrowError();
  });

  const environment = new env();

  it(`${name} should reset successfully`, () => {
    expect(() => environment.reset()).not.toThrowError();
  });

  it(`${name} reset observation should be in observation space`, () => {
    const [obs, info] = environment.reset();
    expect.assert(environment.observationSpace.contains(obs));
  });

  it(`${name} should step successfully`, () => {
    expect(
      // @ts-ignore
      async () => await environment.step(environment.actionSpace.sample())
    ).not.toThrowError();
  });

  it(`${name} step observation should be in observation space`, async () => {
    const [obs, reward, terminate, truncate, info] = await environment.step(
      // @ts-ignore
      environment.actionSpace.sample()
    );
    expect.assert(environment.observationSpace.contains(obs));
  });
});
