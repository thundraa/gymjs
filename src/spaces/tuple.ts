import * as tf from '@tensorflow/tfjs';
import { Space } from './space';

/**
 * A tuple of several spaces
 */
export class Tuple extends Space<Record<string, any>> {
  public spaces: Space<any>[];

  constructor(spaces: Space<any>[]) {
    super([], 'string');
    this.spaces = spaces;
  }

  /**
   * Gets a sample of the tuple space.
   *
   * @returns a tuple with samples from the dict spaces with corresponding keys
   *
   * @override
   */
  sample(): Space<any>[] {
    const sample = this.spaces.map((space) => space.sample());
    return sample;
  }

  /**
   * Determines whether a tuple is in the space or not
   *
   * @returns A boolean that specifies if the value is in the space
   *
   * @override
   */
  contains(x: any[]): boolean {
    // Same type
    if (!Array.isArray(x)) {
      return false;
    }

    // Same length
    if (this.spaces.length !== x.length) {
      return false;
    }

    // Corresponding elements
    return this.spaces.every((space, index) =>
      this.spaces[index].contains(x[index])
    );
  }

  /**
   * Determines if the two spaces are the same
   *
   * @returns A boolean that specifies if the two spaces are the same
   */
  equals(other: Space<any>): boolean {
    // Same type
    if (!(other instanceof Tuple)) {
      return false;
    }

    // Same length
    if (this.spaces.length !== other.spaces.length) {
      return false;
    }

    // Corresponding elements
    return this.spaces.every((space, index) =>
      this.spaces[index].equals(other.spaces[index])
    );
  }
}
