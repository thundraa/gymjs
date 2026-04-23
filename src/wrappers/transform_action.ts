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

      let low: tf.Tensor;
      let high: tf.Tensor;

      if (
        typeof this.env.actionSpace.low === 'number' &&
        typeof this.env.actionSpace.high === 'number'
      ) {
        low = tf.ones(this.env.actionSpace.shape).mul(this.env.actionSpace.low);
        high = tf
          .ones(this.env.actionSpace.shape)
          .mul(this.env.actionSpace.high);
      } else if (
        this.env.actionSpace.low instanceof tf.Tensor &&
        this.env.actionSpace.high instanceof tf.Tensor
      ) {
        low = this.env.actionSpace.low;
        high = this.env.actionSpace.high;
      } else {
        throw new Error('Low and high must be of the same type');
      }

      const lower = action.less(low);
      const higher = high.less(action);

      newAction = tf.where(lower, low, newAction);
      newAction = tf.where(higher, high, newAction);

      return newAction;
    });
  }
}
