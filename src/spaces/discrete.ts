import * as tf from '@tensorflow/tfjs';
import { Space } from './space';

/**
 * A space consisting of finitely many elements.
 */
export class Discrete extends Space {
  /** The number of the discrete elements in the space */
  public n: number;
  /** The smallest element of the space */
  public start: number;

  constructor(n: number, start: number = 0) {
    super([], 'int32');
    this.n = n;
    this.start = start;
  }
  /**
   * Gets a sample of the discrete space.
   *
   * @returns a random integer in the space range
   *
   * @override
   */
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
