import * as tf from '@tensorflow/tfjs';

/**
 * An abstract class that represents the structure of a space.
 */
export abstract class Space {
  /** The dimensions of the space */
  public shape: number[];
  /** The datatype of the space */
  public dtype: tf.DataType;

  protected seed: number | undefined;

  constructor(
    shape: number[],
    dtype: tf.DataType,
    seed: number | undefined = undefined
  ) {
    this.shape = shape;
    this.dtype = dtype;
    this.seed = seed;
  }

  /**
   * Returns a sample of the space.
   *
   * @returns A tensor or number that is acceptable in the space
   */
  abstract sample(): tf.Tensor | number;

  /**
   * Determines whether a value is in the space or not
   *
   * @returns A boolean that specifies if the value is in the space
   */
  abstract contains(x: tf.Tensor | number): boolean;

  /**
   * Determines if the two spaces are the same
   *
   * @returns A boolean that specifies if the two spaces are the same
   */
  abstract equals(other: Space): boolean;
}
