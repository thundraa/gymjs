import * as tf from '@tensorflow/tfjs';
import { Space } from './space';
import { SpaceType } from '../core';

/**
 * MultiBinary is the representation of n-shape binary space
 */
export class MultiBinary extends Space<tf.Tensor> {
  n: number[] | number;
  constructor(n: number[] | number) {
    let inputN: number[];
    if (typeof n === 'number') {
      inputN = [n];
    } else {
      inputN = n;
    }

    if (!inputN.every((num) => num > 0)) {
      throw Error('n (counts) have to be positive');
    }

    super(inputN, 'int32');
    this.n = n;
  }

  /**
   * Gets a sample of the discrete space.
   *
   * @returns a random integer in the space range
   *
   * @override
   */
  sample(): tf.Tensor {
    return tf.randomUniformInt(this.shape, 0, 2);
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

    return x.equal(1).logicalOr(x.equal(0)).all().dataSync()[0] === 1;
  }

  /**
   * Determines if the two multi binary are the same
   *
   * @returns A boolean that specifies if the two multi binary are the same
   */
  equals(other: SpaceType): boolean {
    if (!(other instanceof MultiBinary)) {
      return false;
    }

    if (typeof this.n === 'number' && typeof other.n === 'number') {
      return this.n === other.n;
    } else if (Array.isArray(this.n) && Array.isArray(other.n)) {
      return JSON.stringify(this.n) === JSON.stringify(other.n);
    } else {
      return false;
    }
  }
}
