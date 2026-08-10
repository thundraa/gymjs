import * as tf from '@tensorflow/tfjs';
import { Space } from './space';
import { checkTensors } from '../utils';

/**
 * Box is the representation of the Cartesian product of n closed intervals
 */
export class Box extends Space<tf.Tensor> {
  /** The lower bound for the value */
  public low: tf.Tensor;
  /** The upper bound for the value */
  public high: tf.Tensor;

  constructor(
    low: tf.Tensor | number,
    high: tf.Tensor | number,
    shape: number[],
    dtype: tf.DataType
  ) {
    super(shape, dtype);

    if (typeof low === 'number') {
      this.low = tf.fill(shape, low, dtype);
    } else {
      this.low = low;
    }

    if (typeof high === 'number') {
      this.high = tf.fill(shape, high, dtype);
    } else {
      this.high = high;
    }

    if (JSON.stringify(this.low.shape) !== JSON.stringify(this.shape)) {
      throw new Error('Low should have the same shape as Box!');
    }
    if (JSON.stringify(this.high.shape) !== JSON.stringify(shape)) {
      throw new Error('High should have the same shape as Box!');
    }
    if (this.high.less(this.low).any().dataSync()[0]) {
      throw new Error('Not all values in high are higher than low!');
    }
  }

  /**
   * Gets a sample of the box space.
   *
   * @returns a random tensor in the space range
   *
   * @override
   */
  sample(): tf.Tensor {
    return tf.tidy(() => {
      let low = this.low as tf.Tensor;
      let high = this.high as tf.Tensor;
      if (high.dtype === 'int32') {
        high = high.add(tf.scalar(1));
      }

      low = low.asType('float32');
      high = high.asType('float32');

      let sample = tf.zeros(this.shape, 'float32');

      let boundedBelow = tf.neg(low).less(tf.scalar(Infinity));
      let boundedAbove = high.less(tf.scalar(Infinity));

      let unbounded = boundedBelow
        .logicalNot()
        .logicalAnd(boundedAbove.logicalNot());
      let boundedBelowOnly = boundedBelow.logicalAnd(boundedAbove.logicalNot());
      let boundedAboveOnly = boundedAbove.logicalAnd(boundedBelow.logicalNot());
      let bounded = boundedBelow.logicalAnd(boundedAbove);

      sample = tf.where(
        unbounded,
        tf.randomNormal(this.shape, 0, 1, 'float32'),
        sample
      );
      let boundedTensor = tf.randomUniform(this.shape, 0, 1, 'float32');
      boundedTensor = boundedTensor.mul(high.sub(low));
      boundedTensor = boundedTensor.add(low);
      sample = tf.where(bounded, boundedTensor, sample);

      let unboundedBelowTensor = randomExponential(this.shape);
      unboundedBelowTensor = unboundedBelowTensor.add(low);
      sample = tf.where(boundedBelowOnly, unboundedBelowTensor, sample);

      let unboundedAboveTensor = randomExponential(this.shape);
      unboundedAboveTensor = tf.neg(unboundedAboveTensor).add(high);
      sample = tf.where(boundedAboveOnly, unboundedAboveTensor, sample);

      return sample.asType(this.dtype);
    });
  }

  /**
   * Determines whether a value is in the space or not
   *
   * @returns A boolean that specifies if the value is in the space
   *
   * @override
   */
  contains(x: any): boolean {
    // Same type
    if (!(x instanceof tf.Tensor)) {
      return false;
    }

    // Same shape
    if (JSON.stringify(this.shape) !== JSON.stringify(x.shape)) {
      return false;
    }

    // Same element type
    if (x.dtype !== this.dtype) {
      return false;
    }

    const inBound = x
      .lessEqual(this.high)
      .logicalAnd(tf.logicalNot(x.less(this.low)));

    if (inBound.any().dataSync()[0]) {
      return true;
    }
    return false;
  }

  /**
   * Determines if the two discrete are the same
   *
   * @returns A boolean that specifies if the two discrete are the same
   */
  equals(other: Space<any>): boolean {
    if (!(other instanceof Box)) {
      return false;
    }

    if (this.dtype !== other.dtype) {
      return false;
    } else if (JSON.stringify(this.shape) !== JSON.stringify(other.shape)) {
      return false;
    } else {
      return (
        checkTensors(this.low, other.low, true) &&
        checkTensors(this.high, other.high, true)
      );
    }
  }
}

function randomExponential(shape: number[]) {
  let randomTensor = tf.randomUniform(shape, 0, 1, 'float32');
  randomTensor = tf.neg(tf.log(randomTensor));
  return randomTensor;
}
