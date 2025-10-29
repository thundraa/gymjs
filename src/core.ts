import * as tf from '@tensorflow/tfjs';

import { Discrete } from './spaces/discrete';
import { Box } from './spaces/box';

type ActSpace = Discrete;
type ObsSpace = Box;

export type InfoType<T> = Record<string, T>;

/**
 * An abstract class that represents the structure of an environment.
 */
export abstract class Env<T> {
  /** The render mode of the environment */
  protected renderMode: string | null;
  /** The action space of the environment */
  public actionSpace: ActSpace;
  /** The observation space of the environment */
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

  /**
   * Resets the environment.
   *
   * @returns An array of the observation of the initial state and info
   */
  abstract reset(): [tf.Tensor, InfoType<T> | null];
  /**
   * Takes one step in the environment
   *
   * @param action - action to take in the environment
   * @returns A tuple of the observation of the initial state, reward, termination, truncation and info
   */
  abstract step(
    action: tf.Tensor | number
  ): Promise<[tf.Tensor, number, boolean, boolean, InfoType<T> | null]>;
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
}
