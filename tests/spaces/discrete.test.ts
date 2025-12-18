import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { Discrete } from '../../src/spaces/discrete';

describe('Test Shape Errors', () => {
  it('Number of discrete elements must be positive', () => {
    expect(() => new Discrete(-1)).toThrow(
      'The nummber of discrete elements must be positive!'
    );
  });
});

describe('Test Contain', () => {
  it('Sample should be contained in the space', () => {
    const space = new Discrete(5, -2);
    const sample = space.sample();

    expect.assert(space.contains(sample));
  });
});

describe('Test Equality', () => {
  const space = new Discrete(5, -2);
  it('Spaces should be equal', () => {
    const differentSpace = new Discrete(5, -2);

    expect.assert(space.equals(differentSpace));
  });

  it('Spaces should not be equal for different sizes', () => {
    const differentSpace = new Discrete(4, -2);

    expect.assert(!space.equals(differentSpace));
  });

  it('Spaces should not be equal for different starts', () => {
    const differentSpace = new Discrete(5, 0);

    expect.assert(!space.equals(differentSpace));
  });
});
