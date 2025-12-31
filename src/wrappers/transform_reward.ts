import * as tf from '@tensorflow/tfjs';

import { Env, RewardWrapper } from '../core';

export class ClipReward<ObsType, ActType> extends RewardWrapper<
  ObsType,
  ActType
> {
  private minReward: number | null;
  private maxReward: number | null;

  constructor(
    env: Env<ObsType, ActType>,
    minReward: number | null,
    maxReward: number | null
  ) {
    super(env);

    if (minReward === null && maxReward === null) {
      throw new Error('Both `minReward` and `maxReward` cannot be null');
    }

    if (minReward !== null && maxReward !== null && maxReward < minReward) {
      throw new Error(
        `Min reward (${minReward}) must be smaller than max reward (${maxReward})`
      );
    }

    this.minReward = minReward;
    this.maxReward = maxReward;
  }

  rewardTransform(reward: number): number {
    if (this.minReward === null) {
      return Math.min(reward, this.maxReward as number);
    } else if (this.maxReward === null) {
      return Math.max(reward, this.minReward as number);
    } else {
      return tf.util.clamp(this.minReward, reward, this.maxReward);
    }
  }
}
