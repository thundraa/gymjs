import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { Env, Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper } from '../src/core'
import { Box } from '../src/spaces'
import { Tensor } from '@tensorflow/tfjs';

class ExampleEnv extends Env {
    constructor() {
        const observationSpace = new Box(0, 1, [1], "float32");
        const actioSpace = new Box(0, 1, [1], "float32");
        super(actioSpace, observationSpace, null);
    }

    reset(seed: number | undefined, options: Record<string, any> | null): [Tensor, null] {
        return [tf.tensor([0]), null];
    }

    async step(action: tf.Tensor): Promise<[tf.Tensor, number, boolean, boolean, null]> {
        return [tf.tensor([0]), 0, false, false, null];
    }

    async render(): Promise<void> {
        return;
    }

    close(): void {
        return;
    }
}

class ExampleWrapper extends Wrapper {
    constructor(env: Env | Wrapper) {
        super(env);
    }

    reset(seed?: number | undefined, options?: Record<string, any> | null): [tf.Tensor, Record<string, any> | null] {
        return super.reset(seed, options);
    }

    async step(action: tf.Tensor | number): Promise<[tf.Tensor, number, boolean, boolean, Record<string, any> | null]> {
        let [obs, reward, terminated, truncated, info] = await super.step(action);

        return [obs, 3, terminated, truncated, info];
    }
}


describe('Test Wrapper', () => {
    const exampleEnv = new ExampleEnv();
    const exampleWrapper = new ExampleWrapper(exampleEnv);
    it('Should have the same observation', () => {
        expect.assert(exampleEnv.observationSpace.dtype === exampleWrapper.observationSpace.dtype);
    })
})
