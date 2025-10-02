import * as tf from '@tensorflow/tfjs';

export abstract class Space {
  public shape: number[];
  public dtype: tf.DataType;

  constructor(shape: number[], dtype: tf.DataType) {
    this.shape = shape;
    this.dtype = dtype;
  }

  abstract sample(): tf.Tensor | number;
}
