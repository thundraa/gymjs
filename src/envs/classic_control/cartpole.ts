import * as tf from '@tensorflow/tfjs';

import type { Canvas } from '@napi-rs/canvas';
import type { Sdl } from '@kmamal/sdl';

let sdl: typeof import('@kmamal/sdl') | undefined = undefined;
let createCanvas: typeof import('@napi-rs/canvas').createCanvas | undefined =
  undefined;

if (typeof process !== 'undefined' && process.platform !== 'darwin') {
  sdl = require('@kmamal/sdl');
  createCanvas = require('@napi-rs/canvas').createCanvas;
}

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
  protected state: [number, number, number, number] | null;

  // Instance variables related to rendering
  private canvas: HTMLCanvasElement | Canvas | null;
  private window: Sdl.Video.Window | undefined;

  /**
   * Creates an instance of CartPoleEnv.
   *
   * @param suttonBartoReward - If `True` the reward function matches the original sutton barto implementation
   * @param renderMode - Specify the render mode, null means no rendering and "human" means rendering on a canvas.
   * @param canvas - Specify which canvas to render on, must be specified on web if the rendering mode is human
   */
  constructor(
    suttonBartoReward: boolean = false,
    renderMode: 'human' | null = null,
    canvas: HTMLCanvasElement | null = null
  ) {
    let actionSpace = new Discrete(2);

    const high = tf.tensor([
      CartPoleEnv.xThreshold * 2,
      Infinity,
      CartPoleEnv.thetaThresholdRadians * 2,
      Infinity,
    ]);

    let observationSpace = new Box(tf.neg(high), high, [4], 'float32');

    super(actionSpace, observationSpace, renderMode);
    this.suttonBartoReward = suttonBartoReward;
    this.state = null;
    this.canvas = canvas;
    this.window = undefined;

    if (this.renderMode === 'human') {
      const isNode = typeof process === 'object';
      const isMac = isNode && process.platform === 'darwin';
      if (!isNode && canvas === null) {
        throw Error('Canvas must be provied in human rendering mode!');
      }

      if (isMac) {
        throw Error(
          'Unfortunately SDL does not currently work on Mac OS! Disable human mode.'
        );
      }

      if (isNode) {
        this.window = sdl?.video.createWindow({
          title: 'Cart Pole',
          width: CartPoleEnv.screenWidth,
          height: CartPoleEnv.screenHeight,
        });
        if (createCanvas !== undefined) {
          this.canvas = createCanvas(
            CartPoleEnv.screenWidth,
            CartPoleEnv.screenHeight
          );
        }
      }
    }
  }

  /**
   * Resets the environment.
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

    if (!this.actionSpace.contains(action)) {
      throw Error(`Action invalid`);
    }

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
      await this.render();
      await new Promise((resolve) =>
        setTimeout(resolve, 1000 / CartPoleEnv.frameRate)
      );
    }

    return [tensorState, reward, terminated, false, null];
  }

  /**
   * Renders the environment on the canvas.
   */
  async render(): Promise<void> {
    this.draw();
  }

  /**
   * Closes the window on Node JS
   */
  close(): void {
    this.window?.destroy();
  }

  private draw(): void {
    if (this.canvas === null) {
      throw Error("Can't draw without a canvas!");
    }

    const ctx = this.canvas.getContext('2d');
    if (ctx === null) {
      throw Error('Context must not be bull!');
    }

    this.canvas.width = CartPoleEnv.screenWidth;
    this.canvas.height = CartPoleEnv.screenHeight;

    // Background color
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    if (this.state === null) {
      return;
    }

    let worldWidth = CartPoleEnv.xThreshold * 2;
    let scale = CartPoleEnv.screenWidth / worldWidth;
    let poleWidth = 10.0;
    let poleLen = scale * (2 * CartPoleEnv.poleLength);
    let cartWidth = 50.0;
    let cartHeight = 30.0;

    let [x, _, theta, __] = this.state;

    // Left, right, top, bottom
    let [l, r, t, b] = [
      -cartWidth / 2,
      cartWidth / 2,
      cartHeight / 2,
      -cartHeight / 2,
    ];

    let axleOffset = cartHeight / 4.0;
    let cartX = x * scale + CartPoleEnv.screenWidth / 2.0;
    let cartY = CartPoleEnv.screenHeight - 100;

    // Draw a horizontal line
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(0, cartY);
    ctx.lineTo(CartPoleEnv.screenWidth, cartY);
    ctx.stroke();

    // Draw the cart
    let cartCoords = [
      [l, b], // Bottom-left
      [l, t], // Top-left
      [r, t], // Top-right
      [r, b], // Bottom-right
    ];

    cartCoords = cartCoords.map((c) => [c[0] + cartX, c[1] + cartY]);
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(cartCoords[0][0], cartCoords[0][1]);
    for (let i = 1; i < 4; i++) {
      ctx.lineTo(cartCoords[i][0], cartCoords[i][1]);
    }
    ctx.closePath();
    ctx.fill();

    // Draw the pole
    [l, r, t, b] = [
      -poleWidth / 2,
      poleWidth / 2,
      -(poleLen - poleWidth / 2),
      -poleWidth / 2,
    ];
    let poleCoords = [
      [l, b], // Bottom-left
      [l, t], // Top-left
      [r, t], // Top-right
      [r, b], // Bottom-right
    ];

    poleCoords = poleCoords.map((c) => {
      const cos = Math.cos(theta);
      const sin = Math.sin(theta);
      return [
        c[0] * cos - c[1] * sin + cartX,
        c[0] * sin + c[1] * cos + cartY + axleOffset,
      ];
    });

    // Draw the pole
    ctx.fillStyle = '#ca9895';
    ctx.beginPath();
    ctx.moveTo(poleCoords[0][0], poleCoords[0][1]);
    for (let i = 1; i < 4; i++) {
      ctx.lineTo(poleCoords[i][0], poleCoords[i][1]);
    }
    ctx.closePath();
    ctx.fill();

    // Draw the circle inside the cart
    ctx.beginPath();
    ctx.fillStyle = '#8184cb';
    ctx.arc(cartX, cartY + axleOffset, poleWidth / 2, 0, Math.PI * 2);
    ctx.fill();

    // Render to window on node js
    if (this.window !== undefined) {
      const width = CartPoleEnv.screenWidth;
      const height = CartPoleEnv.screenHeight;
      const buffer = Buffer.from(ctx.getImageData(0, 0, width, height).data);
      this.window.render(width, height, width * 4, 'rgba32', buffer);
    }
  }
}
