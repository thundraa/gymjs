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

describe('Test Env', () => {
    const exampleEnv = new ExampleEnv();
    it('Rendermode should be null', () => {
        expect(exampleEnv.renderMode).toBe(null);
    })
})

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

class ExampleWrapperDifferent extends Wrapper {
    constructor(env: Env | Wrapper) {
        super(env);
        this._observationSpace = new Box(0, 2, [1], "float32");
        this._actionSpace = new Box(0, 2, [1], "float32");
        this._renderMode = "human";
        
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
    const exampleWrapperDifferent = new ExampleWrapperDifferent(exampleEnv);
    it('Should have the same render mode', () => {
        expect.assert(exampleEnv.renderMode === exampleWrapper.renderMode);
    })

    it('Should have the same observation space', () => {
        expect.assert(exampleEnv.observationSpace.equals(exampleWrapper.observationSpace));
    })

    it('Should have the same action space', () => {
        expect.assert(exampleEnv.actionSpace.equals(exampleWrapper.actionSpace));
    })

    it('Should have different render mode', () => {
        expect.assert(!(exampleEnv.renderMode === exampleWrapperDifferent.renderMode));
    })

    it('Should have different observation space', () => {
        expect.assert(!(exampleEnv.observationSpace.equals(exampleWrapperDifferent.observationSpace)));
    })

    it('Should have different action space', () => {
        expect.assert(!(exampleEnv.actionSpace.equals(exampleWrapperDifferent.actionSpace)));
    })
})
