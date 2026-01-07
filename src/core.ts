import * as tf from '@tensorflow/tfjs';

import { Space } from './spaces';

/**
 * An abstract class that represents the structure of an environment.
 */
abstract class BaseEnv<ObsType, ActType> {
  /** The action space of the environment */
  protected _actionSpace?: Space<ActType>;
  /** The observation space of the environment */
  protected _observationSpace?: Space<ObsType>;
  /** The render mode of the environment */
  public renderMode: string | null = null;

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

  get unwrapped(): BaseEnv<unknown, unknown> {
    return this;
  }

  abstract get observationSpace(): Space<unknown>;
  abstract get actionSpace(): Space<unknown>;
}

export abstract class Env<ObsType, ActType> extends BaseEnv<ObsType, ActType> {
  /** The action space of the environment */
  public actionSpace: Space<ActType>;
  /** The observation space of the environment */
  public observationSpace: Space<ObsType>;
  /** The render mode of the environment */
  public renderMode: string | null;

  constructor(
    actionSpace: Space<ActType>,
    observationSpace: Space<ObsType>,
    renderMode: string | null
  ) {
    super();
    this.actionSpace = actionSpace;
    this.observationSpace = observationSpace;
    this.renderMode = renderMode;
  }

  get unwrapped(): Env<unknown, unknown> {
    return this;
  }
}

/**
 * Wrapper around an environment, meant to change/add the behaviour or funtionality
 */
export abstract class Wrapper<
  WrapperObsType,
  WrapperActType,
  ObsType,
  ActType,
> extends BaseEnv<WrapperObsType, WrapperActType> {
  /** Environemnt to wrap */
  env: BaseEnv<ObsType, ActType>;

  constructor(env: BaseEnv<ObsType, ActType>) {
    super();
    this.env = env;
  }

  /**
   * Resets the wrapper.
   *
   * @param options - additional informatiom to specify how the environment resets
   * @returns An array of the observation of the initial state and info
   */
  reset(
    options?: Record<string, any>
  ): [WrapperObsType, Record<string, any> | null] {
    const [obs, info] = this.env.reset(options);
    // TODO: fix ...as unknown as...
    return [obs as unknown as WrapperObsType, info];
  }

  /**
   * Takes one step in the wrapper
   *
   * @param action - action to take in the environment
   * @returns A tuple of the observation of the initial state, reward, termination, truncation and info
   */
  async step(
    action: WrapperActType
  ): Promise<
    [WrapperObsType, number, boolean, boolean, Record<string, any> | null]
  > {
    // TODO: fix ...as unknown as...
    const [obs, reward, terminated, truncated, info] = await this.env.step(
      action as unknown as ActType
    );
    return [
      obs as unknown as WrapperObsType,
      reward,
      terminated,
      truncated,
      info,
    ];
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
   * @returns the unwrapped environment
   */
  get unwrapped(): Env<unknown, unknown> {
    return this.env.unwrapped;
  }
  /**
   * @returns the action space of the wrapper.
   */
  get actionSpace(): Space<WrapperActType> | Space<ActType> {
    if (this._actionSpace === undefined) {
      // We know that the type of action space of the wrapper must be Space<ActType>
      return this.env.actionSpace as Space<ActType>;
    } else {
      return this._actionSpace;
    }
  }
  /**
   * @returns the observation space of the wrapper.
   */
  get observationSpace(): Space<WrapperObsType> | Space<ObsType> {
    if (this._observationSpace === undefined) {
      // We know that the type of observation space of the wrapper must be Space<ObsType>
      return this.env.observationSpace as Space<ObsType>;
    } else {
      return this._observationSpace;
    }
  }
  /**
   * Sets the action space of the wrapper.
   */
  set actionSpace(space: Space<WrapperActType>) {
    this._actionSpace = space;
  }
  /**
   * Sets the observation space of the wrapper.
   */
  set observationSpace(space: Space<WrapperObsType>) {
    this._observationSpace = space;
  }
}

/**
 * A specifc wrapper that only makes changes to the observation
 */
export abstract class ObservationWrapper<
  WrapperObsType,
  ObsType,
  ActType,
> extends Wrapper<WrapperObsType, ActType, ObsType, ActType> {
  constructor(env: BaseEnv<ObsType, ActType>) {
    super(env);
  }

  /**
   * Resets the wrapper.
   *
   * @param options - additional informatiom to specify how the environment resets
   * @returns An array of the observation of the initial state and info
   */
  reset(
    options?: Record<string, any>
  ): [WrapperObsType, Record<string, any> | null] {
    let [obs, info] = this.env.reset(options);
    return [this.observarionTransform(obs), info];
  }

  async step(
    action: ActType
  ): Promise<
    [WrapperObsType, number, boolean, boolean, Record<string, any> | null]
  > {
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
  abstract observarionTransform(obs: ObsType): WrapperObsType;
}

/**
 * A specifc wrapper that only makes changes to the rewards
 */
export abstract class RewardWrapper<ObsType, ActType> extends Wrapper<
  ObsType,
  ActType,
  ObsType,
  ActType
> {
  constructor(env: BaseEnv<ObsType, ActType>) {
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
export abstract class ActionWrapper<
  WrapperActType,
  ObsType,
  ActType,
> extends Wrapper<ObsType, WrapperActType, ObsType, ActType> {
  constructor(env: BaseEnv<ObsType, ActType>) {
    super(env);
  }

  async step(
    action: WrapperActType
  ): Promise<[ObsType, number, boolean, boolean, Record<string, any> | null]> {
    return await this.env.step(this.actionTransform(action));
  }
  /**
   * Makes changes to the actions of the environment
   *
   * @param action - The original action to change
   * @returns the transformed action
   */
  abstract actionTransform(action: WrapperActType): ActType;
}
