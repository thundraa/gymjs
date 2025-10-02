import * as tf from '@tensorflow/tfjs';
import { Space } from './space';

export class Box extends Space {
  public low: number;
  public high: number;

  constructor(low: number, high: number, shape: number[], dtype: tf.DataType) {
    super(shape, dtype);
    this.low = low;
    this.high = high;
  }

  sample(): tf.Tensor {
    return tf.randomUniform(this.shape, this.low, this.high, this.dtype);
  }
}
