import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { MultiDiscrete } from '../../src/spaces/multidiscrete';

describe('Test Shape and Type Errors', () => {
  it('nVec datatype must be an integer', () => {
    expect(() => new MultiDiscrete(tf.tensor([2], [1], 'float32'))).toThrow(
      'nVec data type must be integer!'
    );
  });

  it('nVec elements must all be positive', () => {
    expect(() => new MultiDiscrete(tf.tensor([2, -1], [2], 'int32'))).toThrow(
      'nVec elements must all be positive!'
    );
  });

  it('nVec and start should have the same dimensions', () => {
    expect(
      () =>
        new MultiDiscrete(
          tf.tensor([2], [1], 'int32'),
          tf.tensor([2, 2], [2], 'int32')
        )
    ).toThrow('nVec and start have different dimensions!');
  });
});

describe('Test Contain', () => {
  it('Sample should be contained in the space', () => {
    const space = new MultiDiscrete(
      tf.tensor([3, 4], [2], 'int32'),
      tf.tensor([-2, 1], [2], 'int32')
    );
    const sample = space.sample();

    expect.assert(space.contains(sample));
  });
});

describe('Test Equality', () => {
  const space = new MultiDiscrete(tf.tensor([2], [1], 'int32'));
  it('Spaces should be equal', () => {
    const differentSpace = new MultiDiscrete(tf.tensor([2], [1], 'int32'));

    expect.assert(space.equals(differentSpace));
  });

  it('Spaces should not be equal for different sizes', () => {
    const differentSpace = new MultiDiscrete(tf.tensor([1], [1], 'int32'));

    expect.assert(!space.equals(differentSpace));
  });

  it('Spaces should not be equal for different starts', () => {
    const differentSpace = new MultiDiscrete(
      tf.tensor([2], [1], 'int32'),
      tf.tensor([1], [1], 'int32')
    );

    expect.assert(!space.equals(differentSpace));
  });
});
