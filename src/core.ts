import * as tf from '@tensorflow/tfjs';

import { Discrete } from './spaces/discrete';
import { Box } from './spaces/box';

type ActSpace = Discrete;
type ObsSpace = Box;

type InfoType<T> = Record<string, T>;

abstract class Env {
    protected renderMode: string;
    public actionSpace: ActSpace;
    public observationSpace: ObsSpace;

    constructor(actionSpace: ActSpace, observationSpace: ObsSpace, renderMode: string) {
        this.actionSpace = actionSpace;
        this.observationSpace = observationSpace;
        this.renderMode = renderMode;
    }

    abstract reset(): [tf.Tensor, InfoType<any> | null];
    abstract step(action: tf.Tensor | number): Promise<[tf.Tensor, number, boolean, boolean, InfoType<any> | null]>; // Action is number for now
    abstract render(): Promise<void | tf.Tensor>;
    abstract close(): void;
}

export { Env }