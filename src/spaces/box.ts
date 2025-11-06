import * as tf from '@tensorflow/tfjs';
import { Space } from './space';

/**
 * Box is the representation of the Cartesian product of n closed intervals
 */
export class Box extends Space {
  /** The lower bound for the value */
  public low: number;
  /** The upper bound for the value */
  public high: number;

  constructor(low: number, high: number, shape: number[], dtype: tf.DataType) {
    super(shape, dtype);
    this.low = low;
    this.high = high;
  }

  /**
   * Gets a sample of the box space.
   *
   * @returns a random tensor in the space range
   *
   * @override
   */
  sample(): tf.Tensor {
    return tf.randomUniform(this.shape, this.low, this.high, this.dtype);
  }
}
