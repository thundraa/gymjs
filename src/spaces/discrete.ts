import * as tf from '@tensorflow/tfjs';
import { Space } from './space';

export class Discrete extends Space {
  public n: number;
  public start: number;

  constructor(n: number, start: number = 0) {
    super([], 'int32');
    this.n = n;
    this.start = start;
  }

  sample(): number {
    // Might be a bit too elaborate to use TF but in case we use seeds this is the way
    let randomNumTensor = tf.randomUniformInt(
      [1],
      this.start,
      this.start + this.n
    );
    let [randomNumber] = randomNumTensor.dataSync();

    return randomNumber;
  }
}
