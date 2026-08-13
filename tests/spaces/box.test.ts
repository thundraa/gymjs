import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { Box } from '../../src/spaces/box';

describe('Test Shape Errors', () => {
  it("Box low.shape doesn't match provided shape, low.shape=[2], shape=[3]", () => {
    expect(() => new Box(tf.zeros([2]), tf.ones([3]), [3], 'float32')).toThrow(
      'Low should have the same shape as Box!'
    );
  });

  it("Box high.shape doesn't match provided shape, high.shape=[2], shape=[3]", () => {
    expect(() => new Box(tf.zeros([3]), tf.ones([2]), [3], 'float32')).toThrow(
      'High should have the same shape as Box!'
    );
  });
});

describe('Test Low and High Differene Errors', () => {
  it('Low is higher than high', () => {
    expect(() => new Box(1, 0, [1], 'float32')).toThrow(
      'Not all values in high are higher than low!'
    );
  });

  it('Tensor low is higher than tensor high', () => {
    expect(() => new Box(tf.ones([1]), tf.zeros([1]), [1], 'float32')).toThrow(
      'Not all values in high are higher than low!'
    );
  });
});

describe.each([
  [0, 1, [1], 'float32'],
  [tf.zeros([2]), tf.ones([2]), [2], 'float32'],
  [tf.tensor([-Infinity]), tf.tensor([Infinity]), [1], 'float32'],
  [tf.tensor([-Infinity]), tf.zeros([1]), [1], 'float32'],
  [tf.zeros([1]), tf.tensor([Infinity]), [1], 'float32'],
  [0, 1, [1], 'int32'],
  [tf.zeros([2], 'int32'), tf.ones([2], 'int32'), [2], 'int32'],
])(
  'Test Valid Low and High for low %i high %i shape %i dtype %i',
  (low, high, shape, dtype) => {
    const space = new Box(low, high, shape, dtype as tf.DataType);
    it('Space, low and high dtype should be the same as provided dtype', () => {
      expect.assert(space.dtype === dtype);
      if (low instanceof tf.Tensor) {
        expect.assert((space.low as tf.Tensor).dtype === dtype);
      }
      if (high instanceof tf.Tensor) {
        expect.assert((space.high as tf.Tensor).dtype === dtype);
      }
    });

    const sample = space.sample();
    it('Sample should have the same dtype as provided dtype', () => {
      expect.assert(sample.dtype === dtype);
    });

    it('Samples should be in the bounds of the space', () => {
      expect.assert(space.contains(sample));
    });
  }
);

describe('Test Contains All Elements', () => {
  const space = new Box(0, 1, [2], 'float32');

  it('Should contain a tensor with all elements in bounds', () => {
    expect.assert(space.contains(tf.tensor([0.25, 0.75])));
  });

  it('Should not contain a tensor with only some elements in bounds', () => {
    expect.assert(!space.contains(tf.tensor([0.5, 5])));
  });

  it('Should not contain a tensor with all elements out of bounds', () => {
    expect.assert(!space.contains(tf.tensor([-1, 5])));
  });

  it('Should contain a tensor sitting on the bounds', () => {
    expect.assert(space.contains(tf.tensor([0, 1])));
  });
});

describe('Test Contains Rejects NaN', () => {
  it('Bounded space should not contain NaN', () => {
    const space = new Box(0, 1, [1], 'float32');
    expect.assert(!space.contains(tf.tensor([NaN])));
  });

  it('Unbounded space should not contain NaN', () => {
    const space = new Box(-Infinity, Infinity, [2], 'float32');
    expect.assert(!space.contains(tf.tensor([0.5, NaN])));
  });

  it('Bounded space should not contain a mix of finite and NaN values', () => {
    const space = new Box(0, 1, [2], 'float32');
    expect.assert(!space.contains(tf.tensor([0.5, NaN])));
  });
});

describe('Test Equality', () => {
  const space = new Box(0, 1, [2], 'float32');
  it('Spaces should be equal', () => {
    const differentSpace = new Box(0, 1, [2], 'float32');

    expect.assert(space.equals(differentSpace));
  });

  it('Spaces should not be equal for different shapes', () => {
    const differentSpace = new Box(0, 1, [3], 'float32');

    expect.assert(!space.equals(differentSpace));
  });

  it('Spaces should not be equal for different dtypes', () => {
    const differentSpace = new Box(0, 1, [2], 'int32');

    expect.assert(!space.equals(differentSpace));
  });

  it('Spaces should not be equal for different low or high', () => {
    const differentSpace = new Box(-1, 1, [2], 'float32');

    expect.assert(!space.equals(differentSpace));
  });
});
