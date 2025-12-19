import * as tf from '@tensorflow/tfjs';
import { Space } from './space';
import { checkTensors } from '../utils';

/**
 * MultiDiscrete is the representation of the Cartesian product of n discrete spaces
 */
export class MultiDiscrete extends Space {
  /** The number of the discrete elements in each dimension of the space */
  public nVec: tf.Tensor;
  /** The smallest element of the space in each dimension */
  public start: tf.Tensor;

  constructor(nVec: tf.Tensor, start: tf.Tensor | null = null) {
    super(nVec.shape, 'int32');
    this.nVec = nVec;

    if (this.nVec.dtype !== 'int32') {
      throw Error('nVec data type must be integer!');
    }

    if (start === null) {
      this.start = tf.zeros(nVec.shape, 'int32');
    } else {
      this.start = start;
    }

    if (JSON.stringify(this.nVec.shape) !== JSON.stringify(this.start.shape)) {
      throw Error('nVec and start have different dimensions!');
    }

    const allPositive = !this.nVec.lessEqual(0).any().dataSync()[0];
    if (!allPositive) {
      throw Error('nVec elements must all be positive!');
    }
  }

  /**
   * Gets a sample of the multi discrete space.
   *
   * @returns a random tensor in the space range
   *
   * @override
   */
  sample(): tf.Tensor {
    return tf.tidy(() => {
      // TODO: find a way without using randomUniform (which is float based)
      let random = tf.randomUniform(this.shape, 0, 1, 'float32');
      random = random.mul(this.nVec).add(this.start);

      return random.floor().asType(this.dtype);
    });
  }

  /**
   * Determines whether a value is in the space or not
   *
   * @returns A boolean that specifies if the value is in the space
   *
   * @override
   */
  contains(x: tf.Tensor): boolean {
    // Same type
    if (!(x instanceof tf.Tensor)) {
      return false;
    }

    // Same shape
    if (JSON.stringify(this.nVec.shape) !== JSON.stringify(x.shape)) {
      return false;
    }

    // Same element type
    if (x.dtype !== this.dtype) {
      return false;
    }

    return tf.tidy(() => {
      const sameShape =
        JSON.stringify(x.shape) === JSON.stringify(this.nVec.shape);
      const biggerThanStart = !x.less(this.start).any().dataSync()[0];
      const lessThanMax = !this.nVec
        .lessEqual(x.sub(this.start))
        .any()
        .dataSync()[0];

      return sameShape && biggerThanStart && lessThanMax;
    });
  }

  /**
   * Determines if the two multi discrete are the same
   *
   * @returns A boolean that specifies if the two multi discrete are the same
   */
  equals(other: MultiDiscrete): boolean {
    if (
      checkTensors(this.nVec, other.nVec, true) &&
      checkTensors(this.start, other.start, true)
    ) {
      return true;
    } else {
      return false;
    }
  }
}
