import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { Tuple, Box, Discrete } from '../../src/spaces';

describe('Test Contain', () => {
  const space = new Tuple([new Box(-1, 1, [4], 'float32'), new Discrete(4)]);

  it('Sample should be contained in the space', () => {
    const sample = space.sample();

    expect.assert(space.contains(sample));
  });

  it('Example should not be in the space', () => {
    expect.assert(!space.contains([tf.tensor(0), -1]));
  });
});

describe('Test Equality', () => {
  const space = new Tuple([new Box(-1, 1, [4], 'float32'), new Discrete(4)]);

  it('Spaces should be equal', () => {
    const differentSpace = new Tuple([
      new Box(-1, 1, [4], 'float32'),
      new Discrete(4),
    ]);

    expect.assert(space.equals(differentSpace));
  });

  it('Spaces should not be equal for different length', () => {
    const differentSpace = new Tuple([
      new Box(-1, 1, [4], 'float32'),
      new Discrete(4),
      new Discrete(4),
    ]);

    expect.assert(!space.equals(differentSpace));
  });

  it('Spaces should not be equal for space elements', () => {
    const differentSpace = new Tuple([
      new Box(-1, 1, [4], 'float32'),
      new Discrete(2),
    ]);

    expect.assert(!space.equals(differentSpace));
  });
});
