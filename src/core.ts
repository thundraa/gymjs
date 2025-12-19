import * as tf from '@tensorflow/tfjs';

import { Space } from './spaces/space';

export type ActType = number | tf.Tensor;
export type ObsType = tf.Tensor;

/**
 * An abstract class that represents the structure of an environment.
 */
export abstract class Env {
  /** The action space of the environment */
  public actionSpace: Space;
  /** The observation space of the environment */
  public observationSpace: Space;
  /** The render mode of the environment */
  private readonly _renderMode: string | null;

  constructor(
    actionSpace: Space,
    observationSpace: Space,
    renderMode: string | null
  ) {
    this.actionSpace = actionSpace;
    this.observationSpace = observationSpace;
    this._renderMode = renderMode;
  }

  /**
   * Resets the environment.
   *
   * @param options - additional informatiom to specify how the environment resets
   * @returns An array of the observation of the initial state and info
   */
  abstract reset(
    options?: Record<string, any>
  ): [ObsType, Record<string, any> | null];
  /**
   * Takes one step in the environment
   *
   * @param action - action to take in the environment
   * @returns A tuple of the observation of the initial state, reward, termination, truncation and info
   */
  abstract step(
    action: ActType
  ): Promise<[ObsType, number, boolean, boolean, Record<string, any> | null]>;
  /**
   * Renders the environment graphically.
   *
   * @returns Either no return or an array of the screen of the environment
   */
  abstract render(): Promise<void | tf.Tensor>;
  /**
   * Closes the environment.
   */
  abstract close(): void;

  get unwrapped(): Env {
    return this;
  }

  get renderMode(): string | null {
    return this._renderMode;
  }
}

/**
 * Wrapper around an environment, meant to change/add the behaviour or funtionality
 */
export abstract class Wrapper {
  /** Environemnt to wrap */
  env: Env | Wrapper;
  /** Substitute action space */
  protected _actionSpace: Space | null;
  /** Substitute observation space */
  protected _observationSpace: Space | null;
  /** Substitute Rendermode */
  protected _renderMode: string | null;

  constructor(env: Env | Wrapper) {
    this.env = env;
    this._actionSpace = null;
    this._observationSpace = null;
    this._renderMode = null;
  }

  /**
   * Resets the wrapper.
   *
   * @param options - additional informatiom to specify how the environment resets
   * @returns An array of the observation of the initial state and info
   */
  reset(options?: Record<string, any>): [ObsType, Record<string, any> | null] {
    return this.env.reset(options);
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
    return this.env.step(action);
  }

  /**
   * Renders the wrapper graphically.
   *
   * @returns Either no return or an array of the screen of the environment
   */
  async render(): Promise<void | tf.Tensor> {
    return this.env.render();
  }
  /**
   * Closes the environment.
   */
  close(): void {
    this.env.close();
  }
  /**
   * @returns the action space of the wrapper.
   */
  get actionSpace(): Space {
    if (this._actionSpace === null) {
      return this.env.actionSpace;
    } else {
      return this._actionSpace;
    }
  }
  /**
   * @returns the observation space of the wrapper.
   */
  get observationSpace(): Space {
    if (this._observationSpace === null) {
      return this.env.observationSpace;
    } else {
      return this._observationSpace;
    }
  }
  /**
   * Sets the action space
   */
  set actionSpace(space: Space) {
    this._actionSpace = space;
  }
  /**
   * Sets the observation space
   */
  set observationSpace(space: Space) {
    this._observationSpace = space;
  }
  /**
   * @returns the unwrapped environment
   */
  get unwrapped(): Env | Wrapper {
    return this.env.unwrapped;
  }
  /**
   * @returns the rendermode
   */
  get renderMode(): string | null {
    if (this._renderMode === null) {
      return this.env.renderMode;
    } else {
      return this._renderMode;
    }
  }
}

/**
 * A specifc wrapper that only makes changes to the observation
 */
export abstract class ObservationWrapper extends Wrapper {
  constructor(env: Env | Wrapper) {
    super(env);
  }

  /**
   * Resets the wrapper.
   *
   * @param options - additional informatiom to specify how the environment resets
   * @returns An array of the observation of the initial state and info
   */
  reset(options?: Record<string, any>): [ObsType, Record<string, any> | null] {
    let [obs, info] = this.env.reset(options);
    return [this.observarionTransform(obs), info];
  }

  async step(
    action: ActType
  ): Promise<[ObsType, number, boolean, boolean, Record<string, any> | null]> {
    let [obs, reward, terminated, truncated, info] =
      await this.env.step(action);
    return [
      this.observarionTransform(obs),
      reward,
      terminated,
      truncated,
      info,
    ];
  }
  /**
   * Makes changes to the observation of the environment
   *
   * @param obs - The original observation to change
   * @returns the transformed observation
   */
  abstract observarionTransform(obs: tf.Tensor): tf.Tensor;
}

/**
 * A specifc wrapper that only makes changes to the rewards
 */
export abstract class RewardWrapper extends Wrapper {
  constructor(env: Env | Wrapper) {
    super(env);
  }

  async step(
    action: ActType
  ): Promise<[ObsType, number, boolean, boolean, Record<string, any> | null]> {
    let [obs, reward, terminated, truncated, info] =
      await this.env.step(action);
    return [obs, this.rewardTransform(reward), terminated, truncated, info];
  }
  /**
   * Makes changes to the rewards of the environment
   *
   * @param reward - The original reward to change
   * @returns the transformed reward
   */
  abstract rewardTransform(reward: number): number;
}

/**
 * A specifc wrapper that only makes changes to the actions
 */
export abstract class ActionWrapper extends Wrapper {
  constructor(env: Env | Wrapper) {
    super(env);
  }

  async step(
    action: ActType
  ): Promise<[ObsType, number, boolean, boolean, Record<string, any> | null]> {
    action = this.actionTransform(action);
    let [obs, reward, terminated, truncated, info] =
      await this.env.step(action);
    return [obs, reward, terminated, truncated, info];
  }
  /**
   * Makes changes to the actions of the environment
   *
   * @param action - The original action to change
   * @returns the transformed action
   */
  abstract actionTransform(action: ActType): ActType;
}
