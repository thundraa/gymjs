# Gymjs

Make RL environments for JS with gymjs, gym's equivalent for Javascript.

## Installation

```bash
npm install gymjs
```

## API

Gymjs's API is very similar to gymnasium. Python code for running CartPole's environment:

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
let env = CartPoleEnv();

let [observation, info] = env.reset();
for (let i = 0; i < 1000; i++) {
  let action = env.action_space.sample();
  let [observation, reward, terminated, truncated, info] =
    await env.step(action);

  if (terminated || truncated) {
    let [observation, info] = env.reset();
  }
}
env.close();
```
