import * as tf from '@tensorflow/tfjs';

import { Discrete } from './spaces/discrete';
import { Box } from './spaces/box';
import { Space } from './spaces';

type ActSpace = Discrete;
type ObsSpace = Box;

export type InfoType<T> = Record<string, T>;

export abstract class Env<T> {
  protected renderMode: string | null;
  public actionSpace: ActSpace;
  public observationSpace: ObsSpace;

  constructor(
    actionSpace: ActSpace,
    observationSpace: ObsSpace,
    renderMode: string | null
  ) {
    this.actionSpace = actionSpace;
    this.observationSpace = observationSpace;
    this.renderMode = renderMode;
  }

  abstract reset(): [tf.Tensor, InfoType<T> | null];
  abstract step(
    action: tf.Tensor | number
  ): Promise<[tf.Tensor, number, boolean, boolean, InfoType<T> | null]>; // Action is number for now
  abstract render(): Promise<void | tf.Tensor>;
  abstract close(): void;

  get unwrapped(): Env<T> {
    return this;
  }
}

export abstract class Wrapper<T> {
  env: Env<T> | Wrapper<T>;
  protected _actionSpace: Space | null;
  protected _observationSpace: Space | null;

  constructor(env: Env<T> | Wrapper<T>) {
    this.env = env;
    this._actionSpace = null;
    this._observationSpace = null;
  }

  reset(): [tf.Tensor, InfoType<T> | null] {
    return this.env.reset();
  }

  async step(
    action: tf.Tensor | number
  ): Promise<[tf.Tensor, number, boolean, boolean, InfoType<T> | null]> {
    return this.env.step(action);
  }

  async render(): Promise<void | tf.Tensor> {
    return this.env.render();
  }

  close(): void {
    this.env.close();
  }

  get actionSpace(): Space {
    if (this._actionSpace === null) {
      return this.env.actionSpace;
    } else {
      return this._actionSpace;
    }
  }

  get observationSpace(): Space {
    if (this._observationSpace === null) {
      return this.env.observationSpace;
    } else {
      return this._observationSpace;
    }
  }

  get unwrapped(): Env<T> | Wrapper<T> {
    return this.env.unwrapped;
  }
}

export abstract class ObservationWrapper<T> extends Wrapper<T> {
  constructor(env: Env<T> | Wrapper<T>) {
    super(env);
  }

  reset(): [tf.Tensor, InfoType<T> | null] {
    let [obs, info] = this.env.reset();
    return [this.observarionTransform(obs), info];
  }

  async step(
    action: tf.Tensor | number
  ): Promise<[tf.Tensor, number, boolean, boolean, InfoType<T> | null]> {
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

  abstract observarionTransform(obs: tf.Tensor): tf.Tensor;
}

export abstract class RewardWrapper<T> extends Wrapper<T> {
  constructor(env: Env<T> | Wrapper<T>) {
    super(env);
  }

  async step(
    action: tf.Tensor | number
  ): Promise<[tf.Tensor, number, boolean, boolean, InfoType<T> | null]> {
    let [obs, reward, terminated, truncated, info] =
      await this.env.step(action);
    return [obs, this.rewardTransform(reward), terminated, truncated, info];
  }

  abstract rewardTransform(reward: number): number;
}

export abstract class ActionWrapper<T> extends Wrapper<T> {
  constructor(env: Env<T> | Wrapper<T>) {
    super(env);
  }

  async step(
    action: tf.Tensor | number
  ): Promise<[tf.Tensor, number, boolean, boolean, InfoType<T> | null]> {
    action = this.actionTransform(action);
    let [obs, reward, terminated, truncated, info] =
      await this.env.step(action);
    return [obs, reward, terminated, truncated, info];
  }

  abstract actionTransform(action: tf.Tensor | number): tf.Tensor | number;
}
