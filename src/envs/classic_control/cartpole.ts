import * as tf from '@tensorflow/tfjs';
import Phaser from 'phaser';

import { Discrete } from "../../spaces/discrete";
import { Box } from "../../spaces/box";
import { Env } from "../../core";


export class CartPoleEnv extends Env<null> {
    public static readonly gravity = 9.8;
    public static readonly massCart = 1.0;
    public static readonly massPole = 0.1;
    public static readonly totalMass = CartPoleEnv.massPole + CartPoleEnv.massCart;
    public static readonly poleLength = 0.5;
    public static readonly polemassLength = CartPoleEnv.massPole * CartPoleEnv.poleLength;
    public static readonly forceMag = 10.0;
    public static readonly tau = 0.02;
    public static readonly kinematicsIntegrator = "euler";
    public static readonly thetaThresholdRadians = 12 * 2 * Math.PI / 360;
    public static readonly xThreshold = 2.4;
    public static readonly screenWidth = 600;
    public static readonly screenHeight = 400;
    public static readonly frameRate = 60;

    private readonly suttonBartoReward: boolean;
    private state: [number, number, number, number] | null;
    private game: Phaser.Game | null;
    private canvas: HTMLCanvasElement | null;

    constructor(suttonBartoReward: boolean = false, renderMode = null, canvas: HTMLCanvasElement | null = null) {
        let actionSpace = new Discrete(2);
        let observationSpace = new Box(-Infinity, Infinity, [4], "float32");

        super(actionSpace, observationSpace, renderMode);
        this.suttonBartoReward = suttonBartoReward;
        this.state = null;
        this.game = null;
        this.canvas = canvas;

        if(renderMode === "human") {
            this.render();
        }
    }

    reset(): [tf.Tensor, null] {
        let randomState = tf.randomUniform(this.observationSpace.shape, -0.05, 0.05, this.observationSpace.dtype);
        let [x, xDot, theta, thetaDot] = randomState.dataSync();
        this.state = [x, xDot, theta, thetaDot];

        return [randomState, null]
    }

    async step(action: number): Promise<[tf.Tensor, number, boolean, boolean, null]> {
        if(this.state === null) {
            throw new Error("State variables must be defined.");
        }

        // Logic taken from:
        // https://github.com/sheilaschoepp/gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py
        // I have no idea how it works.

        let [x, xDot, theta, thetaDot] = this.state;
        let force = (action - 0.5 > 0) ? CartPoleEnv.forceMag : -CartPoleEnv.forceMag;
        const costheta = Math.cos(theta);
        const sintheta = Math.sin(theta);

        let temp = (
            force + CartPoleEnv.polemassLength * (thetaDot * thetaDot) * sintheta
        ) / CartPoleEnv.totalMass

        let thetaacc = (CartPoleEnv.gravity * sintheta - costheta * temp) / (
            CartPoleEnv.poleLength
            * (4.0 / 3.0 - CartPoleEnv.massPole * (costheta * costheta) / CartPoleEnv.totalMass)
        )

        let xacc = temp - CartPoleEnv.polemassLength * thetaacc * costheta / CartPoleEnv.totalMass;

        if(CartPoleEnv.kinematicsIntegrator == "euler") {
            x = x + CartPoleEnv.tau * xDot
            xDot = xDot + CartPoleEnv.tau * xacc
            theta = theta + CartPoleEnv.tau * thetaDot
            thetaDot = thetaDot + CartPoleEnv.tau * thetaacc
        } else {  // semi-implicit euler
            xDot = xDot + CartPoleEnv.tau * xacc
            x = x + CartPoleEnv.tau * xDot
            thetaDot = thetaDot + CartPoleEnv.tau * thetaacc
            theta = theta + CartPoleEnv.tau * thetaDot
        }

        this.state = [x, xDot, theta, thetaDot];
        let tensorState = tf.tensor(this.state, this.observationSpace.shape, this.observationSpace.dtype);

        let terminated = x < -CartPoleEnv.xThreshold || x > CartPoleEnv.xThreshold ||
            theta < -CartPoleEnv.thetaThresholdRadians || theta > CartPoleEnv.thetaThresholdRadians;

        let reward: number;
        if(!terminated) {
            reward = this.suttonBartoReward ? 0.0 : 1.0;
        } else {
            reward = this.suttonBartoReward ? -1.0 : 0.0;
        }

        if(this.renderMode === "human") {
            await new Promise(resolve => setTimeout(resolve, 1000 / CartPoleEnv.frameRate));
        }

        return [tensorState, reward, terminated, false, null]
    }

    async render(): Promise<void> {
        if(this.game === null) {
            this.game = this.createGame();
        }
    }

    close(): void {
        if(this.game !== null) {
            this.game.destroy(false);
        }
    }

    getState(): [number, number, number, number] | null {
        return this.state;
    }

    private createGame(): Phaser.Game {
        const cartPoleScene = new CartPoleScene(this);
        if(this.canvas !== null) {
            const config = {
                type: Phaser.CANVAS,
                width: CartPoleEnv.screenWidth,
                height: CartPoleEnv.screenHeight,
                canvas: this.canvas,
                scene: cartPoleScene
            };

            return new Phaser.Game(config);
        } else {
            const config = {
                type: Phaser.AUTO,
                width: CartPoleEnv.screenWidth,
                height: CartPoleEnv.screenHeight,
                scene: cartPoleScene
            };

            return new Phaser.Game(config);
        }
    }
}


class CartPoleScene extends Phaser.Scene {
    pole: Phaser.Geom.Rectangle | null = null;
    cart: Phaser.Geom.Rectangle | null = null;
    cartPoleEnv: CartPoleEnv;
    previousGraphics: Phaser.GameObjects.Graphics | null = null;

    constructor(cartPoleEnv : CartPoleEnv) {
        super();
        this.cartPoleEnv = cartPoleEnv;
    }

    update() {
        let state = this.cartPoleEnv.getState();
        if(state === null) {
            throw Error("State must not be null");
        }

        const graphics = this.add.graphics();
        if(this.previousGraphics !== null) {
            this.previousGraphics.clear();
        }
        this.previousGraphics = graphics;

        let worldWidth = CartPoleEnv.xThreshold * 2
        let scale = CartPoleEnv.screenWidth / worldWidth
        let poleWidth = 10.0
        let poleLen = scale * (2 * CartPoleEnv.poleLength)
        let cartWidth = 50.0
        let cartHeight = 30.0

        let [x, _, theta, __] = state;

        graphics.fillStyle(0x000000);

        // Left, right, top, bottom
        let [l, r, t, b] = [-cartWidth / 2, cartWidth / 2, cartHeight / 2, -cartHeight / 2]
        let axleOffset = cartHeight / 4.0;
        let cartX = x * scale + CartPoleEnv.screenWidth / 2.0;
        let cartY = CartPoleEnv.screenHeight - 100;

        // Set the line style (color and width)
        graphics.lineStyle(2, 0xffffff, 1); // 2 pixels wide, red color

        // Draw a horizontal line
        graphics.moveTo(0, cartY); // Starting point (x, y)
        graphics.lineTo(CartPoleEnv.screenWidth, cartY); // Ending point (x, y)

        // Render the line
        graphics.strokePath();

        // Initialize the cartesian coordinates
        let cartCoords = [
            [l, b], // Bottom-left
            [l, t], // Top-left
            [r, t], // Top-right
            [r, b]  // Bottom-right
        ];

        // Update the cartCoords with the new coordinates
        cartCoords = cartCoords.map((c) => [c[0] + cartX, c[1] + cartY]);

        // Begin drawing the polygon
        graphics.fillStyle(0xffffff);
        graphics.beginPath();
        graphics.moveTo(cartCoords[0][0], cartCoords[0][1]);

        // Loop through the coordinates to create the polygon
        for (let i = 1; i < 4; i++) {
            graphics.lineTo(cartCoords[i][0], cartCoords[i][1]);
        }

        // Close the path and fill the polygon
        graphics.lineTo(cartCoords[0][0], cartCoords[0][1]); // Close the polygon
        graphics.fillPath();

        [l, r, t, b] = [-poleWidth / 2, poleWidth / 2, -(poleLen - poleWidth / 2), -poleWidth / 2]

        // Initialize the cartesian coordinates
        let poleCoords = [
            [l, b], // Bottom-left
            [l, t], // Top-left
            [r, t], // Top-right
            [r, b]  // Bottom-right
        ];

        // Update the poleCoords with the new coordinates
        poleCoords = poleCoords.map((c) => {
            // const vector = new Vector2(c[0], c[1]).rotateRad(theta);
            // let newCoord = c;
            const cos = Math.cos(theta);
            const sin = Math.sin(theta);
            const newCoord = [
                c[0] * cos - c[1] * sin + cartX,
                c[0] * sin + c[1] * cos + cartY + axleOffset
            ]
            return newCoord
        });

        // Begin drawing the polygon
        graphics.fillStyle(0xca9895);
        graphics.beginPath();
        graphics.moveTo(poleCoords[0][0], poleCoords[0][1]);

        // Loop through the coordinates to create the polygon
        for (let i = 1; i < 4; i++) {
            graphics.lineTo(poleCoords[i][0], poleCoords[i][1]);
        }

        // Close the path and fill the polygon
        graphics.lineTo(poleCoords[0][0], poleCoords[0][1]); // Close the polygon
        graphics.fillPath();

        // Set the fill style (color and alpha)
        graphics.fillStyle(0x8184cb, 1); // Red color with full opacity

        // Draw a circle at (400, 300) with a radius of 100
        graphics.fillCircle(cartX, cartY + axleOffset, poleWidth / 2);
    }
}