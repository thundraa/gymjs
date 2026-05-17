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
