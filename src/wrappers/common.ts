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

/**
 * A wrapper that enforcws thw correct order of the environment functions
 */
export class OrderEnforcing extends Wrapper {
  private hasReset: boolean;
  private disableRenderOrderEnforcing: boolean;

  constructor(
    env: Env | Wrapper,
    disableRenderOrderEnforcing: boolean = false
  ) {
    super(env);
    this.disableRenderOrderEnforcing = disableRenderOrderEnforcing;
    this.hasReset = false;
  }

  /**
   * Resets the wrapper.
   *
   * @param options - additional informatiom to specify how the environment resets
   * @returns An array of the observation of the initial state and info
   */
  reset(
    options?: Record<string, any>
  ): [tf.Tensor, Record<string, any> | null] {
    this.hasReset = true;
    return super.reset(options);
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
    if (!this.hasReset) {
      throw new Error('Cannot call env.step() before calling env.reset()');
    }

    return await this.env.step(action);
  }

  /**
   * Renders the environment
   */
  async render(): Promise<void | tf.Tensor> {
    if (!this.disableRenderOrderEnforcing && !this.hasReset) {
      throw new Error(
        'Cannot call env.render() before calling env.reset(), unset disableRenderOrderEnforcing if this is intended'
      );
    }

    return await this.env.render();
  }

  get hasReseted() {
    return this.hasReset;
  }
}

/**
 * A wrapper that records the episode's statistics (episode length, cumulative reward and time elapsed since the beginning)
 * in info with the stats_key as key to a record of rewards, length and time
 */
export class RecordEpisodeStatistics extends Wrapper {
  private statsKey: string;
  private episodeStartTime: number;
  private episodeReturns: number;
  private episodeLengths: number;

  constructor(env: Env | Wrapper, statsKey: string = 'episode') {
    super(env);
    this.statsKey = statsKey;
    this.episodeStartTime = -1;
    this.episodeReturns = 0;
    this.episodeLengths = 0;
  }

  /**
   * Resets the wrapper.
   *
   * @param options - additional informatiom to specify how the environment resets
   * @returns An array of the observation of the initial state and info
   */
  reset(
    options?: Record<string, any>
  ): [tf.Tensor, Record<string, any> | null] {
    this.episodeStartTime = Date.now();
    this.episodeReturns = 0;
    this.episodeLengths = 0;
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
    this.episodeReturns += reward;
    this.episodeLengths += 1;

    if (terminated || truncated) {
      if (info === null) {
        info = {};
      }

      if (this.statsKey in info) {
        throw new Error('Stats key already exists in info!');
      }

      const elapsedTime = (Date.now() - this.episodeStartTime) / 1000;

      info[this.statsKey] = {
        rewards: this.episodeReturns,
        length: this.episodeLengths,
        time: elapsedTime,
      };
    }

    return [obs, reward, terminated, truncated, info];
  }
}
