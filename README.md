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
import * as gym from 'gymjs';
const env = new gym.envs.classic_control.CartPoleEnv();

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
import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';

class Walker extends gym.Env {
  agent: tf.Tensor;
  goal: tf.Tensor;
  constructor() {
    const actionSpace = new gym.spaces.Box(0, 1, [2], 'float32');
    const observationSpace = new gym.spaces.Box(0, 1, [4], 'float32');
    super(actionSpace, observationSpace, null);

    this.agent = tf.tensor([0, 0]);
    this.goal = tf.tensor([0, 0]);
  }

  reset(): [tf.Tensor, null] {
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
const limitedWalker = new gym.wrappers.TimeLimit(walker, 30); // Automatically truncate the environment after 30 steps if the environment hasn't terminated already
```

**Disclaimer:** The project is still in its initial stages; expect a lot of bugs. The API is subject to change.

## Contents

- [gymjs.Env](#gymjssrcoreenv)
- [gymjs.Wrapper](#gymjssrcorewrapper)
- [gymjs.ObservationWrapper](#gymjssrcoreobservationwrapper)
- [gymjs.ActionWrapper](#gymjssrcoreactionwrapper)
- [gymjs.RewardWrapper](#gymjssrcorerewardwrapper)
- [gymjs.spaces](#gymjsspacesspace)
  - [Space](#space)
  - [Box](#box)
  - [Discrete](#discrete)
  - [MultiDiscrete](#multidiscrete)
  - [MultiBinary](#multibinary)
  - [Dict](#dict)
  - [Tuple](#tuple)
- [gymjs.wrappers](#gymjsswrapperswrapper)
  - [TimeLimit](#timelimit)
  - [Autoreset](#autoreset)
  - [OrderEnforcing](#orderenforcing)
  - [RecordEpisodeStatistics](#recordepisodestatistics)
  - [ClipReward](#clipreward)
  - [ClipAction](#clipaction)
  - [RescaleAction](#rescaleaction)
  - [StickyAction](#stickyaction)
- [gymjs.envs](#gymjssenvsenv)
  - [classic_control.CartPoleEnv](#classic_controlcartpoleenv)
  - [classic_control.PendulumEnv](#classic_controlpendulumenv)

# API Reference

## gymjs.Env

An abstract class that represents the structure of an environment.

### gymjs.Env(actionSpace, observationSpace, renderMode)

- `actionSpace: <Space>` The action space of the environment.
- `observationSpace: <Space>` The observation space of the environment.
- `renderMode: <string>|<null>` The render mode of the environment.

Constructor for an environment.

### env.reset([options])

- `options: <object>` Additional information to specify how the environment resets.
- Returns: `[<ObsType>, <Record<string, any> | null>]` An array of the observation of the initial state and info.

Resets the environment.

### env.step(action)

- `action: <ActType>` The action to take in the environment.
- Returns: `Promise<[<ObsType>, <number>, <boolean>, <boolean>, <Record<string, any> | null]>` A tuple of the observation, reward, termination, truncation and info.

Takes one step in the environment.

### env.render()

- Returns: `Promise<void | tf.Tensor>` Either no return or an array of the screen of the environment.

Renders the environment graphically.

### env.close()

Closes the environment.

### env.unwrapped

- Returns: `Env` The unwrapped environment.

## gymjs.Wrapper

A wrapper around an environment, meant to change/add the behaviour or functionality.

### gymjs.Wrapper(env)

- `env: <Env>` The environment to wrap.

### wrapper.reset([options])

- `options: <object>` Additional information to specify how the environment resets.
- Returns: `[<WrapperObsType>, <Record<string, any> | null>]` An array of the observation of the initial state and info.

Resets the wrapper.

### wrapper.step(action)

- `action: <WrapperActType>` The action to take in the environment.
- Returns: `Promise<[<WrapperObsType>, <number>, <boolean>, <boolean>, <Record<string, any> | null]>` A tuple of the observation, reward, termination, truncation and info.

Takes one step in the wrapper.

### wrapper.render()

- Returns: `Promise<void | tf.Tensor>` Either no return or an array of the screen of the environment.

Renders the wrapper graphically.

### wrapper.close()

Closes the environment.

### wrapper.unwrapped

- Returns: `Env` The unwrapped environment.

### wrapper.actionSpace

- Type: `Space<WrapperActType>`

The action space of the wrapper.

### wrapper.observationSpace

- Type: `Space<WrapperObsType>`

The observation space of the wrapper.

## gymjs.ObservationWrapper

A specific wrapper that only makes changes to the observation.

### gymjs.ObservationWrapper(env)

- `env: <Env>` The environment to wrap.

### wrapper.observationTransform(obs)

- `obs: <ObsType>` The original observation to change.
- Returns: `<WrapperObsType>` The transformed observation.

Makes changes to the observation of the environment.

## gymjs.ActionWrapper

A specific wrapper that only makes changes to the actions.

### gymjs.ActionWrapper(env)

- `env: <Env>` The environment to wrap.

### wrapper.actionTransform(action)

- `action: <WrapperActType>` The original action to change.
- Returns: `<ActType>` The transformed action.

Makes changes to the actions of the environment.

## gymjs.RewardWrapper

A specific wrapper that only makes changes to the rewards.

### gymjs.RewardWrapper(env)

- `env: <Env>` The environment to wrap.

### wrapper.rewardTransform(reward)

- `reward: <number>` The original reward to change.
- Returns: `<number>` The transformed reward.

Makes changes to the rewards of the environment.

## gymjs.spaces

### Space

An abstract class that represents the structure of a space.

#### Space(shape, dtype)

- `shape: <number[]>` The dimensions of the space.
- `dtype: <tf.DataType>` The datatype of the space.

#### space.sample()

- Returns: `<T>` A tensor or number that is acceptable in the space.

Returns a sample of the space.

#### space.contains(x)

- `x: <any>` The value to check.
- Returns: `<boolean>` A boolean that specifies if the value is in the space.

Determines whether a value is in the space or not.

#### space.equals(other)

- `other: <Space<any>>` The space to compare with.
- Returns: `<boolean>` A boolean that specifies if the two spaces are the same.

Determines if the two spaces are the same.

### Box

Box is the representation of the Cartesian product of n closed intervals.

#### Box(low, high, shape, dtype)

- `low: <tf.Tensor | number>` The lower bound for the value.
- `high: <tf.Tensor | number>` The upper bound for the value.
- `shape: <number[]>` The dimensions of the space.
- `dtype: <tf.DataType>` The datatype of the space.

#### box.sample()

- Returns: `<tf.Tensor>` A random tensor in the space range.

Gets a sample of the box space.

#### box.contains(x)

- `x: <any>` The value to check.
- Returns: `<boolean>` A boolean that specifies if the value is in the space.

Determines whether a value is in the space or not.

#### box.equals(other)

- `other: <Space<any>>` The space to compare with.
- Returns: `<boolean>` A boolean that specifies if the two boxes are the same.

Determines if the two boxes are the same.

### Discrete

A space consisting of finitely many elements.

#### Discrete(n, start)

- `n: <number>` The number of the discrete elements in the space.
- `start: <number>` The smallest element of the space. Default: `0`

#### discrete.sample()

- Returns: `<number>` A random integer in the space range.

Gets a sample of the discrete space.

#### discrete.contains(x)

- `x: <any>` The value to check.
- Returns: `<boolean>` A boolean that specifies if the value is in the space.

Determines whether a value is in the space or not.

#### discrete.equals(other)

- `other: <Space<any>>` The space to compare with.
- Returns: `<boolean>` A boolean that specifies if the two discrete are the same.

Determines if the two discrete are the same.

### MultiDiscrete

MultiDiscrete is the representation of the Cartesian product of n discrete spaces.

#### MultiDiscrete(nVec, start)

- `nVec: <tf.Tensor>` The number of the discrete elements in each dimension of the space.
- `start: <tf.Tensor | null>` The smallest element of the space in each dimension. Default: `null` (zeros)

#### multiDiscrete.sample()

- Returns: `<tf.Tensor>` A random tensor in the space range.

Gets a sample of the multi discrete space.

#### multiDiscrete.contains(x)

- `x: <any>` The value to check.
- Returns: `<boolean>` A boolean that specifies if the value is in the space.

Determines whether a value is in the space or not.

#### multiDiscrete.equals(other)

- `other: <Space<any>>` The space to compare with.
- Returns: `<boolean>` A boolean that specifies if the two multi discrete are the same.

Determines if the two multi discrete are the same.

### MultiBinary

MultiBinary is the representation of n-shape binary space.

#### MultiBinary(n)

- `n: <number[] | number>` The number of binary elements in the space.

#### multiBinary.sample()

- Returns: `<tf.Tensor>` A random integer in the space range.

Gets a sample of the multi binary space.

#### multiBinary.contains(x)

- `x: <any>` The value to check.
- Returns: `<boolean>` A boolean that specifies if the value is in the space.

Determines whether a value is in the space or not.

#### multiBinary.equals(other)

- `other: <Space<any>>` The space to compare with.
- Returns: `<boolean>` A boolean that specifies if the two multi binary are the same.

Determines if the two multi binary are the same.

### Dict

A dict of several spaces.

#### Dict(spaces)

- `spaces: <Record<string, Space<any>>>` A record of spaces.

#### dict.sample()

- Returns: `<Record<string, any>>` A dict with samples from the dict spaces with corresponding keys.

Gets a sample of the dict space.

#### dict.contains(x)

- `x: <Record<string, any>>` The value to check.
- Returns: `<boolean>` A boolean that specifies if the value is in the space.

Determines whether a dict is in the space or not.

#### dict.equals(other)

- `other: <Space<any>>` The space to compare with.
- Returns: `<boolean>` A boolean that specifies if the two spaces are the same.

Determines if the two spaces are the same.

### Tuple

A tuple of several spaces.

#### Tuple(spaces)

- `spaces: <Space<any>[]>` An array of spaces.

#### tuple.sample()

- Returns: `<Space<any>[]>` A tuple with samples from the dict spaces with corresponding keys.

Gets a sample of the tuple space.

#### tuple.contains(x)

- `x: <any[]>` The value to check.
- Returns: `<boolean>` A boolean that specifies if the value is in the space.

Determines whether a tuple is in the space or not.

#### tuple.equals(other)

- `other: <Space<any>>` The space to compare with.
- Returns: `<boolean>` A boolean that specifies if the two spaces are the same.

Determines if the two spaces are the same.

## gymjs.wrappers

### TimeLimit

A wrapper that places a step limit on the environment.

#### TimeLimit(env, maxEpisodeSteps)

- `env: <Env>` The environment to wrap.
- `maxEpisodeSteps: <number>` The maximum number of steps per episode.

#### timeLimit.reset([options])

- `options: <object>` Additional information to specify how the environment resets.
- Returns: `[<ObsType>, <Record<string, any> | null>]` An array of the observation of the initial state and info.

Resets the wrapper.

#### timeLimit.step(action)

- `action: <ActType>` The action to take in the environment.
- Returns: `Promise<[<ObsType>, <number>, <boolean>, <boolean>, <Record<string, any> | null]>` A tuple of the observation, reward, termination, truncation and info.

Takes one step in the wrapper.

### Autoreset

A wrapper that resets the environment automatically once the environment finishes.

#### Autoreset(env)

- `env: <Env>` The environment to wrap.

#### autoreset.reset([options])

- `options: <object>` Additional information to specify how the environment resets.
- Returns: `[<ObsType>, <Record<string, any> | null>]` An array of the observation of the initial state and info.

Resets the wrapper.

#### autoreset.step(action)

- `action: <ActType>` The action to take in the environment.
- Returns: `Promise<[<ObsType>, <number>, <boolean>, <boolean>, <Record<string, any> | null]>` A tuple of the observation, reward, termination, truncation and info.

Takes one step in the wrapper.

### OrderEnforcing

A wrapper that enforces the correct order of the environment functions.

#### OrderEnforcing(env, disableRenderOrderEnforcing)

- `env: <Env>` The environment to wrap.
- `disableRenderOrderEnforcing: <boolean>` Whether to disable the render order enforcing. Default: `false`

#### orderEnforcing.reset([options])

- `options: <object>` Additional information to specify how the environment resets.
- Returns: `[<ObsType>, <Record<string, any> | null>]` An array of the observation of the initial state and info.

Resets the wrapper.

#### orderEnforcing.step(action)

- `action: <ActType>` The action to take in the environment.
- Returns: `Promise<[<ObsType>, <number>, <boolean>, <boolean>, <Record<string, any> | null]>` A tuple of the observation, reward, termination, truncation and info.

Takes one step in the wrapper.

#### orderEnforcing.render()

- Returns: `Promise<void | tf.Tensor>` Either no return or an array of the screen of the environment.

Renders the environment.

#### orderEnforcing.hasReseted

- Type: `<boolean>`

A boolean that specifies if the environment has been reset.

### RecordEpisodeStatistics

A wrapper that records the episode's statistics (episode length, cumulative reward and time elapsed since the beginning) in info with the stats_key as key to a record of rewards, length and time.

#### RecordEpisodeStatistics(env, statsKey)

- `env: <Env>` The environment to wrap.
- `statsKey: <string>` The key to use for the episode statistics. Default: `'episode'`

#### recordEpisodeStatistics.reset([options])

- `options: <object>` Additional information to specify how the environment resets.
- Returns: `[<ObsType>, <Record<string, any> | null>]` An array of the observation of the initial state and info.

Resets the wrapper.

#### recordEpisodeStatistics.step(action)

- `action: <ActType>` The action to take in the environment.
- Returns: `Promise<[<ObsType>, <number>, <boolean>, <boolean>, <Record<string, any> | null]>` A tuple of the observation, reward, termination, truncation and info.

Takes one step in the wrapper.

### ClipReward

A wrapper that clips the reward to a specified range.

#### ClipReward(env, minReward, maxReward)

- `env: <Env>` The environment to wrap.
- `minReward: <number | null>` The minimum reward. If `null`, only the maximum is enforced.
- `maxReward: <number | null>` The maximum reward. If `null`, only the minimum is enforced.

### ClipAction

A wrapper that clips the action to the action space bounds.

#### ClipAction(env)

- `env: <Env>` The environment to wrap. The action space must be a Box.

### RescaleAction

A wrapper that rescales action space into a specified min and max, currently only works for finite numbers.

#### RescaleAction(env, minAction, maxAction)

- `env: <Env>` The environment to wrap. The action space must be a Box.
- `minAction: <number | tf.Tensor>` The minimum action.
- `maxAction: <number | tf.Tensor>` The maximum action.

### StickyAction

A wrapper that introduces sticky actions, where the previous action is repeated with a given probability for a given duration.

#### StickyAction(env, repeatActionProbability, repeatActionDuration)

- `env: <Env>` The environment to wrap.
- `repeatActionProbability: <number>` The probability of repeating the previous action. Must be in `[0, 1)`.
- `repeatActionDuration: <number | [number, number]>` The number of times to repeat the action. Default: `1`

## gymjs.envs

### classic_control.CartPoleEnv

CartPole, an environment that corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson. The Cartpole problem in reinforcement learning involves balancing a pole on a moving cart along a track, where the agent must learn to keep the pole upright by choosing to move the cart left or right based on the state of the system.

#### CartPoleEnv(suttonBartoReward, renderMode, canvas)

- `suttonBartoReward: <boolean>` If `True` the reward function matches the original sutton barto implementation. Default: `false`
- `renderMode: <'human' | 'rgb_array' | null>` Specify the render mode, `null` means no rendering and `'human'` means rendering on a canvas. Default: `null`
- `canvas: <HTMLCanvasElement | null>` Specify which canvas to render on, must be specified on web if the rendering mode is `'human'`. Default: `null`

#### cartPoleEnv.reset()

- Returns: `[<tf.Tensor>, <null>]` A tuple of observation (type float32 and shape [4]) and info (null).

Resets the environment.

#### cartPoleEnv.step(action)

- `action: <number>` The action chosen, `0` means push to the left and `1` means push to the right.
- Returns: `Promise<[<tf.Tensor>, <number>, <boolean>, <boolean>, <null>]>` A tuple of observation (type float32 and shape [4]), reward, terminated, truncated and info (null).

Takes one step in the environment.

#### cartPoleEnv.render()

- Returns: `Promise<void | tf.Tensor>` Either no return or an array of the screen of the environment.

Renders the environment on the canvas.

#### cartPoleEnv.close()

Closes the window on Node JS.

### classic_control.PendulumEnv

Pendulum, an environment where the agent must learn to swing a pendulum upright and keep it balanced.

#### PendulumEnv(g, renderMode, canvas)

- `g: <number>` Gravity, `10` by default.
- `renderMode: <'human' | 'rgb_array' | null>` Specify the render mode, `null` means no rendering and `'human'` means rendering on a canvas. Default: `null`
- `canvas: <HTMLCanvasElement | null>` Specify which canvas to render on, must be specified on web if the rendering mode is `'human'`. Default: `null`

#### pendulumEnv.reset()

- Returns: `[<tf.Tensor>, <null>]` A tuple of observation (type float32 and shape [3]) and info (null).

Resets the environment.

#### pendulumEnv.step(action)

- `action: <tf.Tensor>` The force applied, between `-2` and `2`. The absolute value is the strength of the force and positive value means that the force is rightward and negative means that it's leftward.
- Returns: `Promise<[<tf.Tensor>, <number>, <boolean>, <boolean>, <null>]>` A tuple of observation (type float32 and shape [3]), reward, terminated, truncated and info (null).

Takes one step in the environment.

#### pendulumEnv.render()

- Returns: `Promise<void | tf.Tensor>` Either no return or an array of the screen of the environment.

Renders the environment on the canvas.

#### pendulumEnv.close()

Closes the window on Node JS.
