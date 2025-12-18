# Gymjs

Gymjs is an open source JS library for developing environments for reinforcement learning by providing a standard API, similar to Python's gym, and a couple of compliant environments like Cartpole and Pendulum.

## Installation

```bash
npm install gymjs
```

## API

Gymjs's API is very similar to that of gymnasium. Python code for running CartPole's environment:

```py
import gymnasium as gym
env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

Equivalent gymjs code:

```ts
import { CartPoleEnv } from 'gymjs/classic_control';
const env = new CartPoleEnv();

let [observation, info] = env.reset();
for (let i = 0; i < 1000; i++) {
  let action = env.actionSpace.sample();
  let [observation, reward, terminated, truncated, info] =
    await env.step(action);

  if (terminated || truncated) {
    let [observation, info] = env.reset();
  }
}
env.close();
```

## Environment Example

An example implementation of an environment:

```ts
import { Env } from 'gymjs';
import { Box } from 'gymjs/spaces';
import { TimeLimit } from 'gymjs/wrappers';

class Walker extends Env {
  agent: tf.Tensor;
  goal: tf.Tensor;
  constructor() {
    const actionSpace = new Box(0, 1, [2], 'float32');
    const observationSpace = new Box(0, 1, [4], 'float32');
    super(actionSpace, observationSpace, null);

    this.agent = tf.tensor([0, 0]);
    this.goal = tf.tensor([0, 0]);
  }

  reset(
    seed: number | undefined = undefined,
    options: Record<string, any> | null = null
  ): [tf.Tensor, null] {
    this.agent = tf.randomUniform([2], 0, 1, 'float32');
    this.goal = tf.randomUniform([2], 0, 1, 'float32');
    const obs = this.agent.concat(this.goal);

    return [obs, null];
  }

  async step(
    action: tf.Tensor
  ): Promise<
    [tf.Tensor, number, boolean, boolean, Record<string, any> | null]
  > {
    if (!this.actionSpace.contains(action)) {
      throw Error('Action not in action space.');
    }

    this.agent = this.agent.add(action.mul(0.05));
    const obs = this.agent.concat(this.goal);
    const distance = this.agent.sub(this.goal).norm().asScalar().dataSync()[0];
    const reward = -distance;

    let done = distance < 0.01;

    return [obs, reward, done, false, null];
  }

  close(): void {
    return;
  }

  async render(): Promise<void> {
    return;
  }
}

const walker = new Walker(); // Create an instance of the environment
const limitedWalker = new TimeLimit(walker, 30); // Automatically truncate the environment after 30 steps if the environment hasn't terminated already
```

**Disclaimer:** The project is still in its initial stages; expect a lot of bugs. The API is subject to change.
