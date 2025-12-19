import * as tf from '@tensorflow/tfjs';

import { Env, Wrapper, ActType, ObsType } from '../core';

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
   * Resets the wrapper.
   *
   * @param options - additional informatiom to specify how the environment resets
   * @returns An array of the observation of the initial state and info
   */
  reset(options?: Record<string, any>): [ObsType, Record<string, any> | null] {
    this.elapsedSteps = 0;
    return super.reset(options);
  }

  /**
   * Takes one step in the wrapper
   *
   * @param action - action to take in the environment
   * @returns A tuple of the observation of the initial state, reward, termination, truncation and info
   */
  async step(
    action: ActType
  ): Promise<[ObsType, number, boolean, boolean, Record<string, any> | null]> {
    let [obs, reward, terminated, truncated, info] =
      await this.env.step(action);
    this.elapsedSteps += 1;

    if (this.elapsedSteps >= this.maxEpisodeSteps) {
      truncated = true;
    }

    return [obs, reward, terminated, truncated, info];
  }
}

/**
 * A wrapper resets the environment automatically once the environment finishes
 */
export class Autoreset extends Wrapper {
  private autoReset: boolean;

  constructor(env: Env | Wrapper) {
    super(env);
    this.autoReset = false;
  }

  /**
   * Resets the wrapper.
   *
   * @param options - additional informatiom to specify how the environment resets
   * @returns An array of the observation of the initial state and info
   */
  reset(options?: Record<string, any>): [ObsType, Record<string, any> | null] {
    this.autoReset = false;
    return super.reset(options);
  }

  /**
   * Takes one step in the wrapper
   *
   * @param action - action to take in the environment
   * @returns A tuple of the observation of the initial state, reward, termination, truncation and info
   */
  async step(
    action: ActType
  ): Promise<[ObsType, number, boolean, boolean, Record<string, any> | null]> {
    let obs: tf.Tensor;
    let reward: number;
    let terminated: boolean;
    let truncated: boolean;
    let info: Record<string, any> | null;

    if (this.autoReset) {
      [obs, info] = this.env.reset();
      [reward, terminated, truncated] = [0.0, false, false];
    } else {
      [obs, reward, terminated, truncated, info] = await this.env.step(action);
    }

    this.autoReset = terminated || truncated;

    return [obs, reward, terminated, truncated, info];
  }
}
