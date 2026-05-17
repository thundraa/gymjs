import * as tf from '@tensorflow/tfjs';

import { ActionWrapper, Env } from '../core';
import { Box } from '../spaces';

export class ClipAction<ObsType> extends ActionWrapper<
  tf.Tensor,
  ObsType,
  tf.Tensor
> {
  constructor(env: Env<ObsType, tf.Tensor>) {
    super(env);

    if (!(this.env.actionSpace instanceof Box)) {
      throw new Error('Clip action only works for Box space');
    }

    this.actionSpace = new Box(
      -Infinity,
      Infinity,
      this.actionSpace.shape,
      this.actionSpace.dtype
    );
  }

  actionTransform(action: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      if (!(this.env.actionSpace instanceof Box)) {
        throw new Error('Clip action only works for Box space');
      }

      let newAction = action.clone();

      const lower = action.less(this.env.actionSpace.low);
      const higher = this.env.actionSpace.high.less(action);

      newAction = tf.where(lower, this.env.actionSpace.low, newAction);
      newAction = tf.where(higher, this.env.actionSpace.high, newAction);

      return newAction;
    });
  }
}

/**
 * A wrapper that rescales action space into a specified min and max, currently only works for finite numbers
 */
export class RescaleAction<ObsType> extends ActionWrapper<
  tf.Tensor,
  ObsType,
  tf.Tensor
> {
  private minAction: tf.Tensor;
  private maxAction: tf.Tensor;
  private gradient: tf.Tensor;
  private intercept: tf.Tensor;

  // Currently it doesn't support Infinite values
  constructor(
    env: Env<ObsType, tf.Tensor>,
    minAction: number | tf.Tensor,
    maxAction: number | tf.Tensor
  ) {
    super(env);

    if (!(this.env.actionSpace instanceof Box)) {
      throw new Error('Clip action only works for Box space');
    }

    if (typeof minAction === 'number') {
      this.minAction = tf.fill(
        env.actionSpace.shape,
        minAction,
        env.actionSpace.dtype
      );
    } else {
      this.minAction = minAction;
    }

    if (typeof maxAction === 'number') {
      this.maxAction = tf.fill(
        env.actionSpace.shape,
        maxAction,
        env.actionSpace.dtype
      );
    } else {
      this.maxAction = maxAction;
    }

    if (
      JSON.stringify(this.minAction.shape) !==
      JSON.stringify(env.actionSpace.shape)
    ) {
      throw new Error('minAction should have the same shape as action space!');
    }
    if (
      JSON.stringify(this.maxAction.shape) !==
      JSON.stringify(env.actionSpace.shape)
    ) {
      throw new Error('maxAction should have the same shape as action space!');
    }

    this.actionSpace = new Box(
      this.minAction,
      this.maxAction,
      this.actionSpace.shape,
      this.actionSpace.dtype
    );

    const highLowDiff = this.env.actionSpace.high.sub(this.env.actionSpace.low);
    this.gradient = tf.div(this.maxAction.sub(this.minAction), highLowDiff);

    this.intercept = tf.add(
      this.gradient.mul(tf.neg(this.env.actionSpace.low)),
      this.minAction
    );

    tf.dispose(highLowDiff);
  }

  actionTransform(action: tf.Tensor): tf.Tensor {
    return action.mul(this.gradient).add(this.intercept);
  }
}
