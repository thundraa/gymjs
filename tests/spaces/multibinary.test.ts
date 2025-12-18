import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { MultiBinary } from '../../src/spaces/multibinary';

describe('Test Shape Errors', () => {
  it('nVec datatype must be an integer', () => {
    expect(() => new MultiBinary([3, -2])).toThrow(
      'n (counts) have to be positive'
    );
  });
});

describe('Test Contain', () => {
  it('Sample should be contained in the space', () => {
    const space = new MultiBinary([3, 2]);
    const sample = space.sample();

    expect.assert(space.contains(sample));
  });
});

describe('Test Equality', () => {
  const space = new MultiBinary([3, 2]);
  it('Spaces should be equal', () => {
    const differentSpace = new MultiBinary([3, 2]);

    expect.assert(space.equals(differentSpace));
  });

  it('Spaces should not be equal for different shapes', () => {
    const differentSpace = new MultiBinary([2, 2]);

    expect.assert(!space.equals(differentSpace));
  });
});
