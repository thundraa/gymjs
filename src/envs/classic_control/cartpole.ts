import * as tf from '@tensorflow/tfjs';

import { Discrete } from '../../spaces/discrete';
import { Box } from '../../spaces/box';
import { Env } from '../../core';

/**
 * CartPole, an environment that corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
 * The Cartpole problem in reinforcement learning involves balancing a pole on a moving cart along a track, where the agent
 * must learn to keep the pole upright by choosing to move the cart left or right based on the state of the system.
 */
export class CartPoleEnv extends Env {
  // A bunch of environment constants
  static readonly gravity = 9.8;
  static readonly massCart = 1.0;
  static readonly massPole = 0.1;
  static readonly totalMass = CartPoleEnv.massPole + CartPoleEnv.massCart;
  static readonly poleLength = 0.5;
  static readonly polemassLength =
    CartPoleEnv.massPole * CartPoleEnv.poleLength;
  static readonly forceMag = 10.0;
  static readonly tau = 0.02;
  static readonly kinematicsIntegrator = 'euler';
  static readonly thetaThresholdRadians = (12 * 2 * Math.PI) / 360;
  static readonly xThreshold = 2.4;
  static readonly screenWidth = 600;
  static readonly screenHeight = 400;
  static readonly frameRate = 60;

  // Per-instance variables
  private readonly suttonBartoReward: boolean;
  private state: [number, number, number, number] | null;

  /**
   * Creates an instance of CartPoleEnv.
   *
   * @param suttonBartoReward - If `True` the reward function matches the original sutton barto implementation
   */
  constructor(suttonBartoReward: boolean = false) {
    let actionSpace = new Discrete(2);

    const high = tf.tensor([
      CartPoleEnv.xThreshold * 2,
      Infinity,
      CartPoleEnv.thetaThresholdRadians * 2,
      Infinity,
    ]);

    let observationSpace = new Box(tf.neg(high), high, [4], 'float32');

    super(actionSpace, observationSpace, null);
    this.suttonBartoReward = suttonBartoReward;
    this.state = null;
  }

  /**
   * Resets the environment.
   *
   * @param suttonBartoReward - If `True` the reward function matches the original sutton barto implementation
   *
   * @returns a tuple of observation (type float32 and shape [4]) and info (null)
   */
  reset(): [tf.Tensor, null] {
    let randomState = tf.randomUniform(
      this.observationSpace.shape,
      -0.05,
      0.05,
      this.observationSpace.dtype
    );
    let [x, xDot, theta, thetaDot] = randomState.dataSync();
    this.state = [x, xDot, theta, thetaDot];

    return [randomState, null];
  }

  /**
   * Takes one step in the environment.
   *
   * @param action - The action chosen, 0 means push to the left and 1 means push to the right
   *
   * @returns A tuple of observation (type float32 and shape [4]), reward, terminated, truncated and info (null)
   */
  async step(
    action: number
  ): Promise<[tf.Tensor, number, boolean, boolean, null]> {
    if (this.state === null) {
      throw new Error('State variables must be defined.');
    }

    // Logic taken from:
    // https://github.com/sheilaschoepp/gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py
    // I have no idea how it works.

    let [x, xDot, theta, thetaDot] = this.state;
    let force = action - 0.5 > 0 ? CartPoleEnv.forceMag : -CartPoleEnv.forceMag;
    const costheta = Math.cos(theta);
    const sintheta = Math.sin(theta);

    let temp =
      (force + CartPoleEnv.polemassLength * (thetaDot * thetaDot) * sintheta) /
      CartPoleEnv.totalMass;

    let thetaacc =
      (CartPoleEnv.gravity * sintheta - costheta * temp) /
      (CartPoleEnv.poleLength *
        (4.0 / 3.0 -
          (CartPoleEnv.massPole * (costheta * costheta)) /
            CartPoleEnv.totalMass));

    let xacc =
      temp -
      (CartPoleEnv.polemassLength * thetaacc * costheta) /
        CartPoleEnv.totalMass;

    if (CartPoleEnv.kinematicsIntegrator == 'euler') {
      x = x + CartPoleEnv.tau * xDot;
      xDot = xDot + CartPoleEnv.tau * xacc;
      theta = theta + CartPoleEnv.tau * thetaDot;
      thetaDot = thetaDot + CartPoleEnv.tau * thetaacc;
    } else {
      // semi-implicit euler
      xDot = xDot + CartPoleEnv.tau * xacc;
      x = x + CartPoleEnv.tau * xDot;
      thetaDot = thetaDot + CartPoleEnv.tau * thetaacc;
      theta = theta + CartPoleEnv.tau * thetaDot;
    }

    this.state = [x, xDot, theta, thetaDot];
    let tensorState = tf.tensor(
      this.state,
      this.observationSpace.shape,
      this.observationSpace.dtype
    );

    let terminated =
      x < -CartPoleEnv.xThreshold ||
      x > CartPoleEnv.xThreshold ||
      theta < -CartPoleEnv.thetaThresholdRadians ||
      theta > CartPoleEnv.thetaThresholdRadians;

    let reward: number;
    if (!terminated) {
      reward = this.suttonBartoReward ? 0.0 : 1.0;
    } else {
      reward = this.suttonBartoReward ? -1.0 : 0.0;
    }

    if (this.renderMode === 'human') {
      await new Promise((resolve) =>
        setTimeout(resolve, 1000 / CartPoleEnv.frameRate)
      );
    }

    return [tensorState, reward, terminated, false, null];
  }

  async render(): Promise<void> {
    return;
  }

  close(): void {
    return;
  }

  getState(): [number, number, number, number] | null {
    return this.state;
  }
}
