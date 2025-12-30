import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { Dict, Box, Discrete } from '../../src/spaces';

describe('Test Contain', () => {
  const space = new Dict({
    key1: new Box(-1, 1, [4], 'float32'),
    key2: new Discrete(4),
  });

  it('Sample should be contained in the space', () => {
    const sample = space.sample();

    expect.assert(space.contains(sample));
  });

  it('Example should not be in the space', () => {
    expect.assert(
      !space.contains({
        key1: tf.tensor(0),
        key2: -1,
      })
    );
  });
});

describe('Test Equality', () => {
  const space = new Dict({
    key1: new Box(-1, 1, [4], 'float32'),
    key2: new Discrete(4),
  });

  it('Spaces should be equal', () => {
    const differentSpace = new Dict({
      key1: new Box(-1, 1, [4], 'float32'),
      key2: new Discrete(4),
    });

    expect.assert(space.equals(differentSpace));
  });

  it('Spaces should not be equal for different key names', () => {
    const differentSpace = new Dict({
      key1_d: new Box(-1, 1, [4], 'float32'),
      key2: new Discrete(4),
    });

    expect.assert(!space.equals(differentSpace));
  });

  it('Spaces should not be equal for space values', () => {
    const differentSpace = new Dict({
      key1: new Box(-1, 1, [4], 'float32'),
      key2: new Discrete(2),
    });

    expect.assert(!space.equals(differentSpace));
  });
});
