import * as tf from '@tensorflow/tfjs';

import { Env, Wrapper } from '../core';


/**
 * A wrapper that places a step limit on the environment
 */
export class TimeLimit extends Wrapper {
  private maxEpisodeSteps: number;
  private elapsedSteps: number;

  constructor(env: Env | Wrapper, maxEpisodeSteps: number) {
    super(env);
    this.maxEpisodeSteps = maxEpisodeSteps;
    this.elapsedSteps = -1; // Env hasn't began yet
  }

  /**
   * Resets the wrapper
   *
   * @returns An array of the observation of the initial state and info
   */
  reset(): [tf.Tensor, Record<string, any> | null] {
    this.elapsedSteps = 0;
    return super.reset();
  }

  /**
   * Takes one step in the wrapper
   *
   * @param action - action to take in the environment
   * @returns A tuple of the observation of the initial state, reward, termination, truncation and info
   */
  async step(
    action: tf.Tensor | number
  ): Promise<
    [tf.Tensor, number, boolean, boolean, Record<string, any> | null]
  > {
    let [obs, reward, terminated, truncated, info] =
      await this.env.step(action);
    this.elapsedSteps += 1;

    if (this.elapsedSteps >= this.maxEpisodeSteps) {
      truncated = true;
    }

    return [obs, reward, terminated, truncated, info];
  }
}
