import * as tf from '@tensorflow/tfjs';
import { Space } from './space';
import { SpaceType } from '../core';

/**
 * A space consisting of finitely many elements.
 */
export class Discrete extends Space<number> {
  /** The number of the discrete elements in the space */
  public n: number;
  /** The smallest element of the space */
  public start: number;

  constructor(n: number, start: number = 0) {
    if (n <= 0) {
      throw new Error('The nummber of discrete elements must be positive!');
    }
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
    let randomNumber = tf.tidy(() => {
      let randomNumTensor = tf.randomUniformInt(
        [1],
        this.start,
        this.start + this.n
      );
      let [randomNumber] = randomNumTensor.dataSync();

      return randomNumber;
    });

    return randomNumber;
  }

  /**
   * Determines whether a value is in the space or not
   *
   * @returns A boolean that specifies if the value is in the space
   *
   * @override
   */
  contains(x: any): boolean {
    if (!(typeof x === 'number')) {
      return false;
    }

    if (x >= this.start && x < this.start + this.n) {
      return true;
    }

    return false;
  }

  /**
   * Determines if the two discrete are the same
   *
   * @returns A boolean that specifies if the two discrete are the same
   */
  equals(other: SpaceType): boolean {
    if (!(other instanceof Discrete)) {
      return false;
    }

    if (this.n === other.n && this.start === other.start) {
      return true;
    } else {
      return false;
    }
  }
}
