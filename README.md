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

---

# API Reference

## Contents

- [class Env](#class-env)
  - [new Env(actionSpace, observationSpace, renderMode)](#new-envactionspace-observationspace-rendermode)
  - [env.actionSpace](#envactionspace)
  - [env.observationSpace](#envobservationspace)
  - [env.renderMode](#envrendermode)
  - [env.unwrapped](#envunwrapped)
  - [env.reset([options])](#envresetoptions)
  - [env.step(action)](#envstepaction)
  - [env.render()](#envrender)
  - [env.close()](#envclose)
- [class Wrapper](#class-wrapper)
  - [new Wrapper(env)](#new-wrapperenv)
  - [wrapper.env](#wrapperenv)
  - [wrapper.actionSpace](#wrapperactionspace)
  - [wrapper.observationSpace](#wrapperobservationspace)
  - [wrapper.unwrapped](#wrapperunwrapped)
  - [wrapper.reset([options])](#wrapperresetoptions)
  - [wrapper.step(action)](#wrapperstepaction)
  - [wrapper.render()](#wrapperrender)
  - [wrapper.close()](#wrapperclose)
- [class ObservationWrapper](#class-observationwrapper)
  - [observationWrapper.observationTransform(obs)](#observationwrapperobservationtransformobs)
- [class ActionWrapper](#class-actionwrapper)
  - [actionWrapper.actionTransform(action)](#actionwrapperactiontransformaction)
- [class RewardWrapper](#class-rewardwrapper)
  - [rewardWrapper.rewardTransform(reward)](#rewardwrapperrewardtransformreward)
- [class Space](#class-space)
  - [new Space(shape, dtype)](#new-spaceshape-dtype)
  - [space.shape](#spaceshape)
  - [space.dtype](#spacedtype)
  - [space.sample()](#spacesample)
  - [space.contains(x)](#spacecontainsx)
  - [space.equals(other)](#spaceequalsother)
- [class Box](#class-box)
  - [new Box(low, high, shape, dtype)](#new-boxlow-high-shape-dtype)
  - [box.low](#boxlow)
  - [box.high](#boxhigh)
  - [box.sample()](#boxsample)
  - [box.contains(x)](#boxcontainsx)
  - [box.equals(other)](#boxequalsother)
- [class Discrete](#class-discrete)
  - [new Discrete(n[, start])](#new-discreten-start)
  - [discrete.n](#discreten)
  - [discrete.start](#discretestart)
  - [discrete.sample()](#discretesample)
  - [discrete.contains(x)](#discretecontainsx)
  - [discrete.equals(other)](#discreteequalsother)
- [class MultiDiscrete](#class-multidiscrete)
  - [new MultiDiscrete(nVec[, start])](#new-multidiscretenvec-start)
  - [multiDiscrete.nVec](#multidiscretenvec)
  - [multiDiscrete.start](#multidiscretestart)
  - [multiDiscrete.sample()](#multidiscretesample)
  - [multiDiscrete.contains(x)](#multidiscretecontainsx)
  - [multiDiscrete.equals(other)](#multidiscreteequalsother)
- [class MultiBinary](#class-multibinary)
  - [new MultiBinary(n)](#new-multibinaryn)
  - [multiBinary.n](#multibinaryn)
  - [multiBinary.sample()](#multibinarysample)
  - [multiBinary.contains(x)](#multibinarycontainsx)
  - [multiBinary.equals(other)](#multibinaryequalsother)
- [class Dict](#class-dict)
  - [new Dict(spaces)](#new-dictspaces)
  - [dict.spaces](#dictspaces)
  - [dict.sample()](#dictsample)
  - [dict.contains(x)](#dictcontainsx)
  - [dict.equals(other)](#dictequalsother)
- [class Tuple](#class-tuple)
  - [new Tuple(spaces)](#new-tuplespaces)
  - [tuple.spaces](#tuplespaces)
  - [tuple.sample()](#tuplesample)
  - [tuple.contains(x)](#tuplecontainsx)
  - [tuple.equals(other)](#tupleequalsother)
- [class TimeLimit](#class-timelimit)
  - [new TimeLimit(env, maxEpisodeSteps)](#new-timelimitenv-maxepisodesteps)
  - [timeLimit.reset([options])](#timelimitresetoptions)
  - [timeLimit.step(action)](#timelimitstepaction)
- [class Autoreset](#class-autoreset)
  - [new Autoreset(env)](#new-autoresetenv)
  - [autoreset.reset([options])](#autoresetresetoptions)
  - [autoreset.step(action)](#autoresetstepaction)
- [class OrderEnforcing](#class-orderenforcing)
  - [new OrderEnforcing(env[, disableRenderOrderEnforcing])](#new-orderenforcingenv-disablerenderorderenforcing)
  - [orderEnforcing.reset([options])](#orderenforcingresetoptions)
  - [orderEnforcing.step(action)](#orderenforcingstepaction)
  - [orderEnforcing.render()](#orderenforcingrender)
  - [orderEnforcing.hasReseted](#orderenforcinghasreseted)
- [class RecordEpisodeStatistics](#class-recordepisodestatistics)
  - [new RecordEpisodeStatistics(env[, statsKey])](#new-recordepisodestatisticsenv-statskey)
  - [recordEpisodeStatistics.reset([options])](#recordepisodestatisticsresetoptions)
  - [recordEpisodeStatistics.step(action)](#recordepisodestatisticsstepaction)
- [class ClipReward](#class-clipreward)
  - [new ClipReward(env, minReward, maxReward)](#new-cliprewardenv-minreward-maxreward)
  - [clipReward.rewardTransform(reward)](#cliprewardrewardtransformreward)
- [class ClipAction](#class-clipaction)
  - [new ClipAction(env)](#new-clipactionenv)
  - [clipAction.actionTransform(action)](#clipactionactiontransformaction)
- [class RescaleAction](#class-rescaleaction)
  - [new RescaleAction(env, minAction, maxAction)](#new-rescaleactionenv-minaction-maxaction)
  - [rescaleAction.actionTransform(action)](#rescaleactionactiontransformaction)
- [class StickyAction](#class-stickyaction)
  - [new StickyAction(env, repeatActionProbability[, repeatActionDuration])](#new-stickyactionenv-repeatactionprobability-repeatactionduration)
  - [stickyAction.actionTransform(action)](#stickyactionactiontransformaction)
- [class CartPoleEnv](#class-cartpoleenv)
  - [new CartPoleEnv(suttonBartoReward[, renderMode[, canvas]])](#new-cartpoleenvsuttonbartoreward-rendermode-canvas)
  - [cartPoleEnv.reset()](#cartpoleenvreset)
  - [cartPoleEnv.step(action)](#cartpoleenvstepaction)
  - [cartPoleEnv.render()](#cartpoleenvrender)
  - [cartPoleEnv.close()](#cartpoleenvclose)
  - [CartPoleEnv constants](#cartpoleenv-constants)
- [class PendulumEnv](#class-pendulumenv)
  - [new PendulumEnv(g[, renderMode[, canvas]])](#new-pendulumenvg-rendermode-canvas)
  - [pendulumEnv.reset()](#pendulumenvreset)
  - [pendulumEnv.step(action)](#pendulumenvstepaction)
  - [pendulumEnv.render()](#pendulumenvrender)
  - [pendulumEnv.close()](#pendulumenvclose)
  - [PendulumEnv constants](#pendulumenv-constants)
- [Rendering](#rendering)
  - [Render modes](#render-modes)
  - [Canvas](#canvas)
  - [Optional dependencies](#optional-dependencies)
  - [Frame timing](#frame-timing)

## class Env

`Env` is the abstract base class for all environments.
It is generic over the observation and action types:

```ts
class MyEnv extends gym.Env<ObsType, ActType> {
  // ...
}
```

Subclasses must implement [`reset()`](#envresetoptions), [`step()`](#envstepaction), [`render()`](#envrender) and [`close()`](#envclose), and must call `super()` with an action space, an observation space, and a render mode.

### new Env(actionSpace, observationSpace, renderMode)

- `actionSpace: `[`<Space>`](#class-space) The action space of the environment.
- `observationSpace: `[`<Space>`](#class-space) The observation space of the environment.
- `renderMode: <string>|<null>` The render mode. Use `null` to disable rendering.
- Returns: `<Env>` the new environment.

Creates the base part of an environment.
Prefer extending this class rather than instantiating it directly.

### env.actionSpace

- [`<Space>`](#class-space)

The action space of the environment.

### env.observationSpace

- [`<Space>`](#class-space)

The observation space of the environment.

### env.renderMode

- `<string>|<null>`

The render mode of the environment.
Possible values used by the built-in environments are `null`, `'human'`, and `'rgb_array'`.
See [Render modes](#render-modes).

### env.unwrapped

- [`<Env>`](#class-env)

The innermost environment.
For a plain `Env` this is the environment itself.
For a wrapper this delegates to the wrapped environment's `unwrapped` property.

### env.reset([options])

- `options: <object>` Additional information specifying how the environment resets. Optional.
- Returns: `[<ObsType>, <Record<string, any>>|<null>]` the initial observation and an info object.

Resets the environment to its initial state.
This method is synchronous and returns the observation directly (not a promise).

### env.step(action)

- `action: <ActType>` The action to take in the environment.
- Returns: `Promise<[<ObsType>, <number>, <boolean>, <boolean>, <Record<string, any>>|<null>]>` a tuple of observation, reward, termination, truncation, and info.

Takes one step in the environment.
The result tuple follows the Gymnasium convention:

| Position | Meaning                                                           |
| -------- | ----------------------------------------------------------------- |
| `0`      | The observation of the new state.                                 |
| `1`      | The reward produced by the action.                                |
| `2`      | `terminated`: `true` if the episode reached a terminal state.     |
| `3`      | `truncated`: `true` if the episode was cut short by a step limit. |
| `4`      | An info object, or `null`.                                        |

### env.render()

- Returns: `Promise<void | tf.Tensor>` nothing in `'human'` mode, or the current frame in `'rgb_array'` mode.

Renders the environment.
See [Rendering](#rendering).

### env.close()

Closes the environment and releases any resources, such as native windows.

## class Wrapper

`Wrapper` is the abstract base class for environment wrappers.
A wrapper delegates to an inner environment, changing or adding behavior while keeping the same interface.

```ts
class MyWrapper extends gym.Wrapper<ObsType, ActType, ObsType, ActType> {
  // ...
}
```

The type parameters are, in order: the wrapper's observation type, the wrapper's action type, the inner environment's observation type, and the inner environment's action type.

### new Wrapper(env)

- `env: `[`<Env>`](#class-env) The environment to wrap.
- Returns: `<Wrapper>` the new wrapper.

### wrapper.env

- [`<Env>`](#class-env)

The wrapped environment.

### wrapper.actionSpace

- [`<Space>`](#class-space)

The action space exposed by the wrapper.
By default this is the wrapped environment's action space, but subclasses may assign a different value.

### wrapper.observationSpace

- [`<Space>`](#class-space)

The observation space exposed by the wrapper.
By default this is the wrapped environment's observation space, but subclasses may assign a different value.

### wrapper.unwrapped

- [`<Env>`](#class-env)

The innermost wrapped environment.

### wrapper.reset([options])

- `options: <object>` Additional information specifying how the environment resets. Optional.
- Returns: `[<ObsType>, <Record<string, any>>|<null>]` the initial observation and info.

Delegates to the wrapped environment's [`reset()`](#envresetoptions).

### wrapper.step(action)

- `action: <ActType>` The action to take in the environment.
- Returns: `Promise<[<ObsType>, <number>, <boolean>, <boolean>, <Record<string, any>>|<null>]>` the step result.

Delegates to the wrapped environment's [`step()`](#envstepaction).

### wrapper.render()

- Returns: `Promise<void | tf.Tensor>`

Delegates to the wrapped environment's [`render()`](#envrender).

### wrapper.close()

Delegates to the wrapped environment's [`close()`](#envclose).

## class ObservationWrapper

An `ObservationWrapper` is a [`Wrapper`](#class-wrapper) that only transforms observations.
`reset()` and `step()` apply [`observationTransform()`](#observationwrapperobservationtransformobs) to the observation returned by the inner environment.

```ts
class ScaleObservation extends gym.ObservationWrapper<
  tf.Tensor,
  tf.Tensor,
  number
> {
  observationTransform(obs: tf.Tensor): tf.Tensor {
    return obs.mul(2);
  }
}
```

### observationWrapper.observationTransform(obs)

- `obs: <ObsType>` The original observation.
- Returns: `<WrapperObsType>` the transformed observation.

Abstract method that subclasses must implement.
It is called on every observation, both from `reset()` and `step()`.

## class ActionWrapper

An `ActionWrapper` is a [`Wrapper`](#class-wrapper) that only transforms actions.
[`step()`](#wrapperstepaction) applies [`actionTransform()`](#actionwrapperactiontransformaction) before passing the action to the inner environment.

```ts
class ShiftAction extends gym.ActionWrapper<number, tf.Tensor, number> {
  actionTransform(action: number): number {
    return action + 1;
  }
}
```

### actionWrapper.actionTransform(action)

- `action: <WrapperActType>` The action supplied by the caller.
- Returns: `<ActType>` the action to pass to the inner environment.

Abstract method that subclasses must implement.

## class RewardWrapper

A `RewardWrapper` is a [`Wrapper`](#class-wrapper) that only transforms rewards.
[`step()`](#wrapperstepaction) applies [`rewardTransform()`](#rewardwrapperrewardtransformreward) to the reward returned by the inner environment.

```ts
class DoubleReward extends gym.RewardWrapper<tf.Tensor, number> {
  rewardTransform(reward: number): number {
    return reward * 2;
  }
}
```

### rewardWrapper.rewardTransform(reward)

- `reward: <number>` The original reward.
- Returns: `<number>` the transformed reward.

Abstract method that subclasses must implement.

## class Space

`Space` is the abstract base class for observation and action spaces.

```ts
class MySpace extends gym.spaces.Space<number> {
  sample(): number {
    return 0;
  }

  contains(x: any): boolean {
    return x === 0;
  }

  equals(other: gym.spaces.Space<any>): boolean {
    return other instanceof MySpace;
  }
}
```

### new Space(shape, dtype)

- `shape: <number>[]` The dimensions of the space.
- `dtype: <tf.DataType>` The data type of elements in the space, for example `'float32'` or `'int32'`.
- Returns: `<Space>` the new space.

### space.shape

- `<number>[]`

The dimensions of the space.
An empty array means a scalar space.

### space.dtype

- `<tf.DataType>`

The data type of the space.

### space.sample()

- Returns: `<T>` a random value from the space.

Abstract method that returns a value contained in the space.

### space.contains(x)

- `x: <any>` The value to check.
- Returns: `<boolean>` `true` if the value is a member of the space.

Abstract method that checks whether a value belongs to the space.

### space.equals(other)

- `other: `[`<Space>`](#class-space) The space to compare against.
- Returns: `<boolean>` `true` if the two spaces are the same.

Abstract method that checks whether two spaces have the same structure and bounds.

## class Box

A `Box` represents the Cartesian product of closed intervals.
It is typically used for continuous observations and actions.
The sample is a [`tf.Tensor`](https://js.tensorflow.org/api/latest/#class:Tensor).

### new Box(low, high, shape, dtype)

- `low: <tf.Tensor>|<number>` The lower bound. A number is broadcast to every element.
- `high: <tf.Tensor>|<number>` The upper bound. A number is broadcast to every element.
- `shape: <number>[]` The shape of the space.
- `dtype: <tf.DataType>` The data type of the space.
- Returns: `<Box>` the new box space.

Throws if `low` or `high` do not have the requested shape, or if any element of `high` is lower than the corresponding element of `low`.

```ts
const space = new gym.spaces.Box(-1, 1, [2, 2], 'float32');
```

### box.low

- `<tf.Tensor>`

The lower bound of every element in the space.

### box.high

- `<tf.Tensor>`

The upper bound of every element in the space.

### box.sample()

- Returns: `<tf.Tensor>` a random tensor in the space range.

Samples each element from the corresponding interval.
Infinite bounds are handled by sampling from a normal or exponential distribution, matching Gymnasium's behavior.
Values are floored when the data type is `'int32'`.

### box.contains(x)

- `x: <any>` The value to check.
- Returns: `<boolean>` `true` if `x` is a tensor with the same shape and type and every element is within `[low, high]`.

### box.equals(other)

- `other: `[`<Space>`](#class-space) The space to compare against.
- Returns: `<boolean>` `true` if both spaces are boxes with the same data type, shape, and bounds.

## class Discrete

A `Discrete` space contains a finite number of integers.
The sample is a plain `number`.

### new Discrete(n[, start])

- `n: <number>` The number of elements in the space. Must be positive.
- `start: <number>` The smallest element of the space. Default: `0`
- Returns: `<Discrete>` the new discrete space.

Throws if `n` is not positive.
The shape is always `[]` and the data type is always `'int32'`.

```ts
const actionSpace = new gym.spaces.Discrete(4); // {0, 1, 2, 3}
```

### discrete.n

- `<number>`

The number of elements in the space.

### discrete.start

- `<number>`

The smallest element of the space.

### discrete.sample()

- Returns: `<number>` a random integer in `[start, start + n)`.

### discrete.contains(x)

- `x: <any>` The value to check.
- Returns: `<boolean>` `true` if `x` is an integer in `[start, start + n)`.

### discrete.equals(other)

- `other: `[`<Space>`](#class-space) The space to compare against.
- Returns: `<boolean>` `true` if both spaces are discrete with the same `n` and `start`.

## class MultiDiscrete

A `MultiDiscrete` space is the Cartesian product of several discrete spaces.
The sample is a [`tf.Tensor`](https://js.tensorflow.org/api/latest/#class:Tensor) with data type `'int32'`.

### new MultiDiscrete(nVec[, start])

- `nVec: <tf.Tensor>` The number of elements in each dimension. Must have data type `'int32'` and contain only positive values.
- `start: <tf.Tensor>|<null>` The smallest element in each dimension. Default: a tensor of zeros with the same shape as `nVec`.
- Returns: `<MultiDiscrete>` the new multi-discrete space.

Throws if `nVec` is not `'int32'`, if `start` does not have the same shape as `nVec`, or if any element of `nVec` is not positive.
The shape of the space is `nVec.shape`.

```ts
const space = new gym.spaces.MultiDiscrete(tf.tensor([2, 3, 4], [3], 'int32'));
```

### multiDiscrete.nVec

- `<tf.Tensor>`

The number of elements in each dimension.

### multiDiscrete.start

- `<tf.Tensor>`

The smallest element in each dimension.

### multiDiscrete.sample()

- Returns: `<tf.Tensor>` a random integer tensor where element `i` is in `[start[i], start[i] + nVec[i])`.

### multiDiscrete.contains(x)

- `x: <any>` The value to check.
- Returns: `<boolean>` `true` if `x` is an `'int32'` tensor with the same shape as `nVec` and every element is within its dimension's range.

### multiDiscrete.equals(other)

- `other: `[`<Space>`](#class-space) The space to compare against.
- Returns: `<boolean>` `true` if both spaces have the same `nVec` and `start`.

## class MultiBinary

A `MultiBinary` space contains tensors whose elements are either `0` or `1`.

### new MultiBinary(n)

- `n: <number>|<number>[]` The number of binary elements, or the shape of the space. Must be positive.
- Returns: `<MultiBinary>` the new multi-binary space.

If `n` is a number, the shape is `[n]`.
If `n` is an array, the array is used as the shape.
The data type is always `'int32'`.

```ts
const space = new gym.spaces.MultiBinary([3, 2]);
```

### multiBinary.n

- `<number>|<number>[]`

The value passed to the constructor.

### multiBinary.sample()

- Returns: `<tf.Tensor>` a random tensor of zeros and ones with the space's shape.

### multiBinary.contains(x)

- `x: <any>` The value to check.
- Returns: `<boolean>` `true` if `x` is an `'int32'` tensor with the same shape whose elements are all `0` or `1`.

### multiBinary.equals(other)

- `other: `[`<Space>`](#class-space) The space to compare against.
- Returns: `<boolean>` `true` if both spaces were created with the same `n`.

## class Dict

A `Dict` space is a record of named spaces.
The sample is a plain object whose keys match the space's keys.

### new Dict(spaces)

- `spaces: <Record<string, Space<any>>>` An object mapping names to spaces.
- Returns: `<Dict>` the new dictionary space.

The shape is always `[]` and the data type is always `'string'`.

```ts
const space = new gym.spaces.Dict({
  position: new gym.spaces.Box(-1, 1, [2], 'float32'),
  velocity: new gym.spaces.Box(-10, 10, [2], 'float32'),
});
```

### dict.spaces

- `<Record<string, Space<any>>>`

The named spaces that make up the dictionary.

### dict.sample()

- Returns: `<Record<string, any>>` an object with the same keys, where each value is a sample from the corresponding space.

### dict.contains(x)

- `x: <any>` The value to check.
- Returns: `<boolean>` `true` if `x` is an object with exactly the same keys and each value is contained in its corresponding space.

### dict.equals(other)

- `other: `[`<Space>`](#class-space) The space to compare against.
- Returns: `<boolean>` `true` if both spaces have the same keys and each corresponding pair of spaces is equal.

## class Tuple

A `Tuple` space is an ordered collection of spaces.
The sample is an array whose elements are samples from the constituent spaces.

### new Tuple(spaces)

- `spaces: <Space<any>[]>` The spaces that make up the tuple.
- Returns: `<Tuple>` the new tuple space.

The shape is always `[]` and the data type is always `'string'`.

```ts
const space = new gym.spaces.Tuple([
  new gym.spaces.Discrete(2),
  new gym.spaces.Box(-1, 1, [2], 'float32'),
]);
```

### tuple.spaces

- `<Space<any>[]>`

The spaces that make up the tuple.

### tuple.sample()

- Returns: `<any[]>` an array containing a sample from each constituent space.

### tuple.contains(x)

- `x: <any>` The value to check.
- Returns: `<boolean>` `true` if `x` is an array with the same length and each element is contained in the corresponding space.

### tuple.equals(other)

- `other: `[`<Space>`](#class-space) The space to compare against.
- Returns: `<boolean>` `true` if both spaces have the same length and each corresponding pair of spaces is equal.

## class TimeLimit

A wrapper that puts a maximum step count on an episode.

### new TimeLimit(env, maxEpisodeSteps)

- `env: `[`<Env>`](#class-env) The environment to wrap.
- `maxEpisodeSteps: <number>` The maximum number of steps per episode.
- Returns: `<TimeLimit>` the new wrapper.

### timeLimit.reset([options])

- `options: <object>` Options passed to the inner environment. Optional.
- Returns: `[<ObsType>, <Record<string, any>>|<null>]` the reset result.

Resets the step counter and delegates to the inner environment's `reset()`.

### timeLimit.step(action)

- `action: <ActType>` The action to take.
- Returns: `Promise<[<ObsType>, <number>, <boolean>, <boolean>, <Record<string, any>>|<null>]>` the step result.

Increments the step counter and sets `truncated` to `true` once `maxEpisodeSteps` has been reached.

## class Autoreset

A wrapper that automatically resets the inner environment after an episode ends.
On the first step after a terminal step, the wrapper resets the environment and returns the initial observation with a reward of `0`, `terminated` and `truncated` both `false`, and the reset info.

### new Autoreset(env)

- `env: `[`<Env>`](#class-env) The environment to wrap.
- Returns: `<Autoreset>` the new wrapper.

### autoreset.reset([options])

- `options: <object>` Options passed to the inner environment. Optional.
- Returns: `[<ObsType>, <Record<string, any>>|<null>]` the reset result.

Clears the internal autoreset flag and delegates to the inner environment's `reset()`.

### autoreset.step(action)

- `action: <ActType>` The action to take.
- Returns: `Promise<[<ObsType>, <number>, <boolean>, <boolean>, <Record<string, any>>|<null>]>` the step result.

If the previous step ended the episode, the action is ignored, the environment is reset, and the initial observation is returned.
Otherwise the action is passed to the inner environment.

## class OrderEnforcing

A wrapper that enforces calling `reset()` before `step()` and `render()`.

### new OrderEnforcing(env[, disableRenderOrderEnforcing])

- `env: `[`<Env>`](#class-env) The environment to wrap.
- `disableRenderOrderEnforcing: <boolean>` Set to `true` to allow calling `render()` before `reset()`. Default: `false`
- Returns: `<OrderEnforcing>` the new wrapper.

### orderEnforcing.reset([options])

- `options: <object>` Options passed to the inner environment. Optional.
- Returns: `[<ObsType>, <Record<string, any>>|<null>]` the reset result.

Marks the environment as reset and delegates to the inner environment's `reset()`.

### orderEnforcing.step(action)

- `action: <ActType>` The action to take.
- Returns: `Promise<[<ObsType>, <number>, <boolean>, <boolean>, <Record<string, any>>|<null>]>` the step result.

Throws if `reset()` has not been called yet.
Otherwise delegates to the inner environment's `step()`.

### orderEnforcing.render()

- Returns: `Promise<void | tf.Tensor>`

Throws if `reset()` has not been called and render order enforcing is enabled.
Otherwise delegates to the inner environment's `render()`.

### orderEnforcing.hasReseted

- `<boolean>`

Whether `reset()` has been called on this wrapper.

## class RecordEpisodeStatistics

A wrapper that records episode statistics and adds them to the info object when an episode ends.

The statistics are stored in `info[statsKey]` as an object:

```ts
{
  rewards: number; // cumulative reward
  length: number; // number of steps
  time: number; // elapsed time in seconds
}
```

### new RecordEpisodeStatistics(env[, statsKey])

- `env: `[`<Env>`](#class-env) The environment to wrap.
- `statsKey: <string>` The key under which the statistics are stored in `info`. Default: `'episode'`
- Returns: `<RecordEpisodeStatistics>` the new wrapper.

### recordEpisodeStatistics.reset([options])

- `options: <object>` Options passed to the inner environment. Optional.
- Returns: `[<ObsType>, <Record<string, any>>|<null>]` the reset result.

Resets the accumulated return, episode length, and start time, then delegates to the inner environment's `reset()`.

### recordEpisodeStatistics.step(action)

- `action: <ActType>` The action to take.
- Returns: `Promise<[<ObsType>, <number>, <boolean>, <boolean>, <Record<string, any>>|<null>]>` the step result.

Accumulates the reward and step count.
When the episode is terminated or truncated, the statistics are added to `info` under `statsKey`.
Throws if `info` already contains a value for `statsKey`.

## class ClipReward

A [`RewardWrapper`](#class-rewardwrapper) that clips rewards to a range.

### new ClipReward(env, minReward, maxReward)

- `env: `[`<Env>`](#class-env) The environment to wrap.
- `minReward: <number>|<null>` The minimum reward, or `null` for no lower bound.
- `maxReward: <number>|<null>` The maximum reward, or `null` for no upper bound.
- Returns: `<ClipReward>` the new wrapper.

Throws if both bounds are `null`, or if `maxReward` is lower than `minReward`.

### clipReward.rewardTransform(reward)

- `reward: <number>` The original reward.
- Returns: `<number>` the clipped reward.

## class ClipAction

An [`ActionWrapper`](#class-actionwrapper) that clips actions to the inner environment's action space.
This wrapper only works with a [`Box`](#class-box) action space.

### new ClipAction(env)

- `env: `[`<Env>`](#class-env) The environment to wrap. Its action space must be a [`Box`](#class-box).
- Returns: `<ClipAction>` the new wrapper.

Throws if the inner environment's action space is not a `Box`.
The wrapper exposes a `Box` action space with the same shape and data type but with bounds `-Infinity` and `Infinity`.

### clipAction.actionTransform(action)

- `action: <tf.Tensor>` The action to clip.
- Returns: `<tf.Tensor>` the action with every element clipped to the inner action space's bounds.

## class RescaleAction

An [`ActionWrapper`](#class-actionwrapper) that linearly rescales actions from the inner environment's action space to a new range.
This wrapper only works with a [`Box`](#class-box) action space with finite bounds.

### new RescaleAction(env, minAction, maxAction)

- `env: `[`<Env>`](#class-env) The environment to wrap. Its action space must be a [`Box`](#class-box).
- `minAction: <number>|<tf.Tensor>` The minimum of the new action range. A number is broadcast to the action space's shape.
- `maxAction: <number>|<tf.Tensor>` The maximum of the new action range. A number is broadcast to the action space's shape.
- Returns: `<RescaleAction>` the new wrapper.

Throws if the inner action space is not a `Box`, if `minAction` or `maxAction` do not have the same shape as the action space, or if `maxAction` is lower than `minAction` (via [`Box`](#class-box) validation).
The inner action space must have finite bounds; infinite bounds are not currently supported.
The wrapper exposes a `Box` action space with bounds `[minAction, maxAction]`.

### rescaleAction.actionTransform(action)

- `action: <tf.Tensor>` The action in the inner environment's range.
- Returns: `<tf.Tensor>` the action mapped to `[minAction, maxAction]`.

The mapping is affine:

```text
newAction = action * (maxAction - minAction) / (high - low)
          + minAction - low * (maxAction - minAction) / (high - low)
```

where `low` and `high` are the inner action space's bounds.

## class StickyAction

An [`ActionWrapper`](#class-actionwrapper) that repeats the previous action with a given probability, for a random number of steps.

### new StickyAction(env, repeatActionProbability[, repeatActionDuration])

- `env: `[`<Env>`](#class-env) The environment to wrap.
- `repeatActionProbability: <number>` The probability of repeating the previous action. Must be in `[0, 1)`.
- `repeatActionDuration: <number>|<[number, number]>` How many times to repeat an action, or a lower and upper bound (inclusive) for a random duration. Must be integers greater than or equal to `1`. Default: `1`
- Returns: `<StickyAction>` the new wrapper.

Throws a `RangeError` if the probability is outside `[0, 1)` or the duration is out of range, and a `TypeError` if the duration is not an integer.

### stickyAction.actionTransform(action)

- `action: <ActType>` The action supplied by the caller.
- Returns: `<ActType>` either `action` or the previous action.

Once a repeat is triggered, the previous action is repeated for the sampled duration before new actions are accepted again.
The wrapper uses `Math.random()` because seeding is not implemented yet.

## class CartPoleEnv

An implementation of the classic cart-pole balancing problem, based on the version described by Barto, Sutton, and Anderson.

The agent must keep a pole upright by pushing a cart either left or right.

- Action space: [`Discrete(2)`](#class-discrete), where `0` pushes left and `1` pushes right.
- Observation space: [`Box`](#class-box) with shape `[4]` and data type `'float32'`, representing the cart position, cart velocity, pole angle, and pole angular velocity.
- Observations are `tf.Tensor`s.

### new CartPoleEnv(suttonBartoReward[, renderMode[, canvas]])

- `suttonBartoReward: <boolean>` Use the original Sutton-Barto reward function. Default: `false`
- `renderMode: <string>|<null>` `null` for no rendering, `'human'` for a window or canvas, or `'rgb_array'` for frames as tensors. Default: `null`
- `canvas: <HTMLCanvasElement>|<null>` The canvas to render on in browsers. Required when `renderMode` is not `null` in a browser. Default: `null`
- Returns: `<CartPoleEnv>` the new environment.

```ts
const env = new gym.envs.classic_control.CartPoleEnv();
```

With the default reward function, the environment returns `1` for every step that does not terminate and `0` on the terminating step.
With `suttonBartoReward` enabled, it returns `0` for non-terminal steps and `-1` on the terminating step.

### cartPoleEnv.reset()

- Returns: `[<tf.Tensor>, <null>]` an observation of shape `[4]` and `null` info.

Resets the cart position, cart velocity, pole angle, and pole angular velocity by sampling uniformly from `[-0.05, 0.05]`.

### cartPoleEnv.step(action)

- `action: <number>` `0` to push left, `1` to push right.
- Returns: `Promise<[<tf.Tensor>, <number>, <boolean>, <boolean>, <null>]>` a tuple of observation, reward, terminated, truncated, and `null` info.

Advances the simulation by `tau` seconds.
The episode terminates when the cart position leaves `[-2.4, 2.4]` or the pole angle leaves `[-12°, 12°]`.
`truncated` is always `false`.
Throws if the action is not in the action space or if [`reset()`](#cartpoleenvreset) has not been called.

In `'human'` mode, `step()` also renders the environment and waits for one frame (60 FPS).

### cartPoleEnv.render()

- Returns: `Promise<void | tf.Tensor>`

In `'human'` mode, draws the current state to the window or canvas.
In `'rgb_array'` mode, returns a `tf.Tensor` of shape `[400, 600, 4]` containing RGBA pixels.
Returns nothing when `renderMode` is `null`.

### cartPoleEnv.close()

Destroys the native window, if one was created.

### CartPoleEnv constants

| Constant                | Value       | Description                                 |
| ----------------------- | ----------- | ------------------------------------------- |
| `gravity`               | `9.8`       | Acceleration due to gravity.                |
| `massCart`              | `1.0`       | Mass of the cart.                           |
| `massPole`              | `0.1`       | Mass of the pole.                           |
| `totalMass`             | `1.1`       | Total mass of the cart and pole.            |
| `poleLength`            | `0.5`       | Half the length of the pole.                |
| `polemassLength`        | `0.05`      | `massPole * poleLength`.                    |
| `forceMag`              | `10.0`      | Magnitude of the force applied to the cart. |
| `tau`                   | `0.02`      | Simulation time step in seconds.            |
| `kinematicsIntegrator`  | `'euler'`   | Integration method.                         |
| `thetaThresholdRadians` | `0.2094...` | Maximum pole angle, equal to 12 degrees.    |
| `xThreshold`            | `2.4`       | Maximum cart position.                      |
| `screenWidth`           | `600`       | Render width in pixels.                     |
| `screenHeight`          | `400`       | Render height in pixels.                    |
| `frameRate`             | `60`        | Frame rate used for human rendering.        |

## class PendulumEnv

An implementation of the classic pendulum swing-up problem.
The agent must swing a pendulum upright and keep it balanced.

- Action space: [`Box`](#class-box) with shape `[1]`, data type `'float32'`, and bounds `[-2, 2]`. The action is the torque applied to the pendulum.
- Observation space: [`Box`](#class-box) with shape `[3]`, data type `'float32'`, and bounds `[-1, -1, -8]` to `[1, 1, 8]`, representing `cos(theta)`, `sin(theta)`, and the angular velocity.
- Observations are `tf.Tensor`s.

### new PendulumEnv(g[, renderMode[, canvas]])

- `g: <number>` The acceleration due to gravity. Default: `10.0`
- `renderMode: <string>|<null>` `null` for no rendering, `'human'` for a window or canvas, or `'rgb_array'` for frames as tensors. Default: `null`
- `canvas: <HTMLCanvasElement>|<null>` The canvas to render on in browsers. Required when `renderMode` is not `null` in a browser. Default: `null`
- Returns: `<PendulumEnv>` the new environment.

```ts
const env = new gym.envs.classic_control.PendulumEnv();
```

### pendulumEnv.reset()

- Returns: `[<tf.Tensor>, <null>]` an observation of shape `[3]` and `null` info.

Samples the angle uniformly from `[-π, π]` and the angular velocity uniformly from `[-1, 1]`, then returns `[cos(theta), sin(theta), thetaDot]`.

### pendulumEnv.step(action)

- `action: <tf.Tensor>` A tensor of shape `[1]` containing the torque.
- Returns: `Promise<[<tf.Tensor>, <number>, <boolean>, <boolean>, <null>]>` a tuple of observation, reward, terminated, truncated, and `null` info.

The action must be contained in the action space.
It is clipped to `[-2, 2]` before being applied.
The reward is the negative of the cost:

```text
cost = angleNormalize(theta)² + 0.1 * thetaDot² + 0.001 * torque²
```

`terminated` and `truncated` are always `false`.
Throws if the action is not in the action space or if [`reset()`](#pendulumenvreset) has not been called.

In `'human'` mode, `step()` also renders the environment and waits for one frame (30 FPS).

### pendulumEnv.render()

- Returns: `Promise<void | tf.Tensor>`

In `'human'` mode, draws the current state to the window or canvas.
In `'rgb_array'` mode, returns a `tf.Tensor` of shape `[500, 500, 4]` containing RGBA pixels.
Returns nothing when `renderMode` is `null`.

### pendulumEnv.close()

Destroys the native window, if one was created.

### PendulumEnv constants

| Constant    | Value  | Description                                |
| ----------- | ------ | ------------------------------------------ |
| `maxSpeed`  | `8`    | Maximum angular velocity.                  |
| `maxTorque` | `2.0`  | Maximum torque magnitude.                  |
| `dt`        | `0.05` | Simulation time step in seconds.           |
| `m`         | `1.0`  | Mass of the pendulum.                      |
| `l`         | `1.0`  | Length of the pendulum.                    |
| `screenDim` | `500`  | Render width and height in pixels.         |
| `defaultX`  | `π`    | Maximum absolute initial angle.            |
| `defaulty`  | `1.0`  | Maximum absolute initial angular velocity. |
| `frameRate` | `30`   | Frame rate used for human rendering.       |

## Rendering

### Render modes

| Value         | Description                                                                                               |
| ------------- | --------------------------------------------------------------------------------------------------------- |
| `null`        | No rendering. `render()` returns nothing.                                                                 |
| `'human'`     | Displays the environment in an SDL window on Node.js, or draws to the supplied canvas in a browser.       |
| `'rgb_array'` | `render()` returns the current frame as a `tf.Tensor` of shape `[height, width, 4]` with RGBA pixel data. |

### Canvas

On Node.js, the canvas used for drawing is created internally.
In browsers, you must pass an `HTMLCanvasElement` to the environment constructor when `renderMode` is not `null`; otherwise the constructor throws.

Human-window rendering is not supported on macOS; creating the window fails with an error.
Use `'rgb_array'` or `null` on macOS instead.

### Optional dependencies

Node.js rendering uses two optional dependencies:

- [`@kmamal/sdl`](https://www.npmjs.com/package/@kmamal/sdl) for window management
- [`@napi-rs/canvas`](https://www.npmjs.com/package/@napi-rs/canvas) for Canvas2D drawing

They are imported lazily, and only when a render mode is not `null`.
If they are not installed, environments can still be used without rendering.

### Frame timing

In `'human'` mode, the built-in environments render during `step()` and then wait for one frame:

- CartPole: 60 FPS
- Pendulum: 30 FPS

Calling `render()` manually draws immediately without waiting.
