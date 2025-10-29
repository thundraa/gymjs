import * as tf from '@tensorflow/tfjs';

/**
 * An abstract class that represents the structure of a space.
 */
export abstract class Space {
  /** The dimensions of the space */
  public shape: number[];
  /** The datatype of the space */
  public dtype: tf.DataType;

  constructor(shape: number[], dtype: tf.DataType) {
    this.shape = shape;
    this.dtype = dtype;
  }

  /**
   * Returns a sample of the space.
   *
   * @returns A tensor or number that is acceptable in the space
   */
  abstract sample(): tf.Tensor | number;
}
