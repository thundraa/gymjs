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

import { Box } from '../../spaces/box';
import { Env } from '../../core';

/**
 * CartPole, an environment that corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
 * The Cartpole problem in reinforcement learning involves balancing a pole on a moving cart along a track, where the agent
 * must learn to keep the pole upright by choosing to move the cart left or right based on the state of the system.
 */
export class PendulumEnv extends Env {
  // A bunch of environment constants
  static readonly maxSpeed = 8;
  static readonly maxTorque = 2.0;
  static readonly dt = 0.05;
  static readonly m = 1.0;
  static readonly l = 1.0;
  static readonly screenDim = 500;
  static readonly defaultX = Math.PI;
  static readonly defaulty = 1.0;
  static readonly frameRate = 30;

  // Per-instance variables
  private readonly g: number;
  protected state: [number, number] | null;
  private lastU: number | null;

  // Instance variables related to rendering
  private canvas: HTMLCanvasElement | Canvas | null;
  private window: Sdl.Video.Window | undefined;

  /**
   * Creates an instance of CartPoleEnv.
   *
   * @param g - Gravity, 10 by default
   * @param renderMode - Specify the render mode, null means no rendering and "human" means rendering on a canvas.
   * @param canvas - Specify which canvas to render on, must be specified on web if the rendering mode is human
   */
  constructor(
    g: number = 10.0,
    renderMode: 'human' | 'rgb_array' | null = null,
    canvas: HTMLCanvasElement | null = null
  ) {
    let actionSpace = new Box(
      -PendulumEnv.maxTorque,
      PendulumEnv.maxTorque,
      [1],
      'float32'
    );

    const high = tf.tensor([1.0, 1.0, PendulumEnv.maxSpeed]);

    let observationSpace = new Box(tf.neg(high), high, high.shape, 'float32');

    super(actionSpace, observationSpace, renderMode);
    this.g = g;
    this.state = null;
    this.lastU = null;
    this.canvas = canvas;
    this.window = undefined;

    const isNode = typeof process === 'object';
    if (this.renderMode === 'human' || this.renderMode === 'rgb_array') {
      const isMac = isNode && process.platform === 'darwin';
      if (!isNode && canvas === null) {
        throw Error('Canvas must be provied in rendering mode in web!');
      }

      if (isMac) {
        throw Error(
          'Unfortunately Rendering does not currently work on Mac OS! Disable rendering mode.'
        );
      }

      if (isNode && this.renderMode === 'human') {
        this.window = sdl?.video.createWindow({
          title: 'Cart Pole',
          width: PendulumEnv.screenDim,
          height: PendulumEnv.screenDim,
        });
      }

      if (createCanvas !== undefined) {
        this.canvas = createCanvas(
          PendulumEnv.screenDim,
          PendulumEnv.screenDim
        );
      }
    }
  }

  /**
   * Resets the environment.
   *
   * @returns a tuple of observation (type float32 and shape [4]) and info (null)
   */
  reset(): [tf.Tensor, null] {
    const high = tf.tensor([PendulumEnv.defaultX, PendulumEnv.defaulty]);
    const low = tf.neg(high);

    const random = tf.randomUniform(high.shape, 0, 1, 'float32');
    const randomState = random.mul(high.sub(low)).add(low);

    let [theta, thetaDot] = randomState.dataSync();
    this.state = [theta, thetaDot];
    this.lastU = null;

    return [this.getObs(), null];
  }

  /**
   * Takes one step in the environment.
   *
   * @param action - The force applied, between -2 and 2. The absolute value is the strength of the force and
   * positive value means that the force is rightward and negative means that it's leftward.
   *
   * @returns A tuple of observation (type float32 and shape [4]), reward, terminated, truncated and info (null)
   */
  async step(
    action: tf.Tensor
  ): Promise<[tf.Tensor, number, boolean, boolean, null]> {
    if (this.state === null) {
      throw new Error('State variables must be defined.');
    }

    // Logic taken from:
    // https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py
    // I have no idea how it works.

    if (!this.actionSpace.contains(action)) {
      throw Error(`Action invalid`);
    }

    let [theta, thetaDot] = this.state;

    action = tf.clipByValue(
      action,
      -PendulumEnv.maxTorque,
      PendulumEnv.maxTorque
    );
    const u = action.dataSync()[0];
    this.lastU = u;

    const costs =
      angleNormalize(theta) ** 2 + 0.1 * thetaDot ** 2 + 0.001 * u ** 2;

    let newThetaDot =
      thetaDot +
      (((3 * this.g) / (2 * PendulumEnv.l)) * Math.sin(theta) +
        (3.0 / (PendulumEnv.m * PendulumEnv.l ** 2)) * u) *
        PendulumEnv.dt;
    newThetaDot = tf.util.clamp(
      -PendulumEnv.maxSpeed,
      newThetaDot,
      PendulumEnv.maxSpeed
    );
    const newTheta = theta + newThetaDot * PendulumEnv.dt;

    this.state = [newTheta, newThetaDot];

    if (this.renderMode === 'human') {
      await this.render();
      await new Promise((resolve) =>
        setTimeout(resolve, 1000 / PendulumEnv.frameRate)
      );
    }

    return [this.getObs(), -costs, false, false, null];
  }

  /**
   * Renders the environment on the canvas.
   */
  async render(): Promise<void | tf.Tensor> {
    if (this.renderMode === 'human') {
      this.draw(false);
    } else if (this.renderMode === 'rgb_array') {
      return this.draw(true);
    }
  }

  /**
   * Closes the window on Node JS
   */
  close(): void {
    this.window?.destroy();
  }

  private getObs(): tf.Tensor {
    if (this.state === null) {
      throw Error("State can't be null!");
    }

    let [theta, thetaDot] = this.state;
    const obs = tf.tensor([Math.cos(theta), Math.sin(theta), thetaDot]);
    return obs;
  }

  private draw(returnTensor: boolean): tf.Tensor | void {
    // Drawing is mostly the direct translation of gymnasium's Pygame rendering with ChatGPT
    if (this.state === null) {
      throw Error("Can't draw without a state!");
    }

    if (this.canvas === null) {
      throw Error("Can't draw without a canvas!");
    }

    const ctx = this.canvas.getContext('2d');
    if (ctx === null) {
      throw Error('Context must not be bull!');
    }

    this.canvas.width = PendulumEnv.screenDim;
    this.canvas.height = PendulumEnv.screenDim;

    ctx.save();
    ctx.clearRect(0, 0, PendulumEnv.screenDim, PendulumEnv.screenDim);

    ctx.fillStyle = 'rgb(0,0,0)';
    ctx.fillRect(0, 0, PendulumEnv.screenDim, PendulumEnv.screenDim);

    // --- Geometry --------------------------------------------------------------

    const bound = 2.2;
    const scale = PendulumEnv.screenDim / (bound * 2);
    const offset = PendulumEnv.screenDim / 2;

    const rodLength = 1 * scale;
    const rodWidth = 0.2 * scale;

    const l = 0;
    const r = rodLength;
    const t = rodWidth / 2;
    const b = -rodWidth / 2;

    const angle = this.state[0] + Math.PI / 2;

    // Utility — rotate around origin
    function rotate(x: number, y: number, a: number): [number, number] {
      return [
        x * Math.cos(a) - y * Math.sin(a),
        x * Math.sin(a) + y * Math.cos(a),
      ];
    }

    // Base rod corners → transformed
    const corners: [number, number][] = [
      [l, b],
      [l, t],
      [r, t],
      [r, b],
    ].map(([x, y]) => {
      const [rx, ry] = rotate(x, y, angle);
      return [rx + offset, ry + offset];
    });

    // --- Rod polygon -----------------------------------------------------------

    ctx.fillStyle = 'rgb(204,77,77)';
    ctx.strokeStyle = 'rgb(204,77,77)';
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    for (let i = 1; i < corners.length; i++) {
      ctx.lineTo(corners[i][0], corners[i][1]);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // --- Center circle ---------------------------------------------------------

    const radius = rodWidth / 2;

    ctx.beginPath();
    ctx.arc(offset, offset, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // --- Rod end circle ---------------------------------------------------------

    let [ex, ey] = rotate(rodLength, 0, angle);
    ex += offset;
    ey += offset;

    ctx.beginPath();
    ctx.arc(ex, ey, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // --- Torque / image arrow ---------------------------------------------------

    if (this.lastU !== null) {
      // Arrow size proportional to |u|
      const arrowSize = (scale * Math.abs(this.lastU)) / 4 + 15;

      ctx.save();
      ctx.translate(offset, offset);

      // If this.lastU > 0, draw arrow flipped horizontally (matches image flip)
      if (this.lastU > 0) {
        ctx.scale(-1, 1);
      }

      // Arrow style
      ctx.strokeStyle = 'rgb(255, 255, 255)';
      ctx.fillStyle = 'rgb(255, 255, 255)';
      ctx.lineWidth = 2;

      // Draw a curved arrow like a torque direction indicator
      const R = arrowSize * 0.6; // radius of curve
      const head = arrowSize * 0.25; // arrowhead size

      ctx.beginPath();
      ctx.arc(0, 0, R, Math.PI * 0.5, -Math.PI, true); // curved path
      ctx.stroke();

      // Arrow head at end of arc
      const endAngle = -Math.PI;
      const ex = R * Math.cos(endAngle);
      const ey = R * Math.sin(endAngle);

      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex + head / 2, ey);
      ctx.lineTo(ex, ey + head);
      ctx.lineTo(ex - head / 2, ey);
      ctx.closePath();
      ctx.fill();

      ctx.restore();
    }

    // --- Axle (black center) ----------------------------------------------------

    const axleR = 0.05 * scale;

    ctx.fillStyle = 'black';
    ctx.strokeStyle = 'black';

    ctx.beginPath();
    ctx.arc(offset, offset, axleR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // --- Vertical flip (pygame.transform.flip False, True) ----------------------
    // This must occur *after* drawing everything.

    ctx.translate(0, PendulumEnv.screenDim);
    ctx.scale(1, -1);

    // Now the content we drew earlier appears flipped because
    // the transform affects subsequent blits.
    // We simply re-draw the flipped buffer onto itself:
    //
    // Trick: use drawImage on the same canvas.
    // (This is allowed as long as source/destination don't overlap exactly.)

    // @ts-ignore
    ctx.drawImage(
      ctx.canvas,
      0,
      0,
      PendulumEnv.screenDim,
      PendulumEnv.screenDim,
      0,
      0,
      PendulumEnv.screenDim,
      PendulumEnv.screenDim
    );

    ctx.restore(); // restore to original transform

    const width = PendulumEnv.screenDim;
    const height = PendulumEnv.screenDim;
    if (returnTensor) {
      const imageArray = ctx.getImageData(0, 0, width, height).data;
      return tf.tensor(imageArray).reshape([width, height, 4]);
    }

    // Render to window on node js
    if (this.window !== undefined) {
      const buffer = Buffer.from(ctx.getImageData(0, 0, width, height).data);
      this.window.render(width, height, width * 4, 'rgba32', buffer);
    }
  }
}

function angleNormalize(x: number): number {
  return ((x + Math.PI) % (2 * Math.PI)) - Math.PI;
}
