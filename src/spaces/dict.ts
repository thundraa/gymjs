import * as tf from '@tensorflow/tfjs';
import { Space } from './space';

/**
 * A dict of several spaces
 */
export class Dict extends Space<Record<string, any>> {
  public spaces: Record<string, Space<any>>;

  constructor(spaces: Record<string, Space<any>>) {
    super([], 'string');
    this.spaces = spaces;
  }

  /**
   * Gets a sample of the dict space.
   *
   * @returns a dict with samples from the dict spaces with corresponding keys
   *
   * @override
   */
  sample(): Record<string, any> {
    const sample: Record<string, any> = {};

    Object.keys(this.spaces).forEach((key) => {
      sample[key] = this.spaces[key].sample(); // Transform the value as needed
    });

    return sample;
  }

  /**
   * Determines whether a dict is in the space or not
   *
   * @returns A boolean that specifies if the value is in the space
   *
   * @override
   */
  contains(x: Record<string, any>): boolean {
    // Same type
    if (typeof x !== 'object') {
      return false;
    }

    // Same keys
    if (!haveSameKeys(this.spaces, x)) {
      return false;
    }

    // Corresponding items
    return Object.keys(this.spaces).every((key) =>
      this.spaces[key].contains(x[key])
    );
  }

  /**
   * Determines if the two spaces are the same
   *
   * @returns A boolean that specifies if the two spaces are the same
   */
  equals(other: Space<any>): boolean {
    // Same type
    if (!(other instanceof Dict)) {
      return false;
    }

    // Same keys
    if (!haveSameKeys(this.spaces, other.spaces)) {
      return false;
    }

    // Corresponding items
    return Object.keys(this.spaces).every((key) =>
      this.spaces[key].equals(other.spaces[key])
    );
  }
}

function haveSameKeys(
  recordOne: Record<string, any>,
  recordTwo: Record<string, any>
) {
  const recordOneLength = Object.keys(recordOne).length;
  const recordTwoLength = Object.keys(recordTwo).length;

  if (recordOneLength === recordTwoLength) {
    return Object.keys(recordOne).every((key) => recordTwo.hasOwnProperty(key));
  }

  return false;
}
