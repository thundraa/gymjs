## Gymjs prototype

### Spaces

https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/spaces/space.py

Class space in general is like:

```py
class Space:
    def __init__(self, shape, dtype, seed):
        ...

    def sample() # Abstract

    # Alongside some methods which we don't need yet
```

- `shape` is the dimension of space, so like if the space is [[2, 3, 5], [1, 4, 0]] the shape is (2, 3)
- `dtype` is the type of the data that's in the space, like float32, int32 etc. We use types that are supported by tensorflow-js.
- `seed` is something related to randomization, we can ignore it for now. I don't even now if something like this is doable with JS as it relies on numpy.

- The method `sample` simply returns a sample of that space, would explain it more on observation space and action space.

Proposed Typescript equivalent:
gymnasium/spaces/space.ts

```ts
// Corresponding to types that tensowflow js supports

abstract class Space {
  public shape: number[]; // Array of integers
  public dtype: tf.DataType;

  constructor(shape: number[], dtype: tf.DataType) {
    this._shape = shape;
    this.dtype = dtype;
  }

  abstract sample(): tf.Tensor | number;
}
```

### Box

https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/spaces/box.py

Class box in general is like:

```py
class Box(Space):
    def __init__(self, low, high, shape, dtype, seed):
        ...

    def sample():
        ...

    # Alongside some methods which we don't need yet
```

- `low` is the lowest possible value
- `high` is the highest possible value
- `shape` and `type` are the same thing as space.

To make clear what `low` and `high` do, imagine that we declared an object of `Box(-1, 1, (2, 2), "int32")`.
Then an acceptable space is `[[-1, 0], [1, 1]]` but not `[[2, 5], [1, 9]]`. Basically all values should be between low and high.

What Box basically does is it conveys the notion of what the sizes and type of the observation is, so for example if we have a snake game we define observation as four numbers perhaps. Two for the position of the snake and another 2 for the position the apple. We know that the width don't exceed 100 for example. So the shape would be `[4]`, with type `int32` and an example of such observation return would be [50, 0, 34, 60]. And the method `sample` simply returns a sample of such observation that satisfies such constraints.

The reason we need this information is so that we know what type of data/dimensions we're dealing with which is necessary when we construct models to deal with the game data.

Proposed Typescript equivalent:
gymnasium/spaces/box.ts

```ts
class Box extends Space {
    private low: number;
    private high: number;

    constructor(low: number, high: number, shape: number[], dtype: tf.DataType) {
        super(shape, dtype)
        this.low = low;
        this.high = high;
    }

    sample(): tf.Tensor {
        ...
    }
}
```

### Discrete

https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/spaces/discrete.py

Class Discrete in general is like:

```py
class Space:
    def __init__(self, n, seed, start):
        ...

    def sample():
        ...

    # Alongside some methods which we don't need yet
```

- `n` is the size of discrete space
- `start` is where the numbering start

Basically it's an action space where your actions are mutually exclusive which we specify with numbers. For example in a snake game, we can say number 0 means going up, number 1 means going right and so on.

Proposed Typescript equivalent:

gymnasium/spaces/discrete

.ts

```ts
class Discrete extends Space {
    public n: number;
    public start: number;

    constructor(n: number, start: number = 0) {
        super(shape: [], dtype: "int32")
        this.n = n;
        this.start = start;
    }

    sample(): number {
        ...
    }
}
```

### ObsSpace and ActSpace

It's important to distinguish between spaces that are observations and spaces that are actions when defining an environment. We simply define a type for actions spaces and observations spaces:

```ts
type ActSpace = Discrete | ... // Other types that we might add later
type ObsSpace = Box | ... // Other types that we might add later
```

### Env

https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py:

```py
class Env: # Abstract class
    render_mode: str
    action_space: Discrete # just this for now
    observation_space: Box # just this for now
    def reset()
    def step(action)
    def render()
    def close()
```

- `render_mode` is how we choose to render or not render at all, commonly "human" and "rgb_array". "human" mean rendering the game and in 60 fps while "rgb_array" means not rendering at all. "human" is used in testing and "rgb_array" is used in training
- `action_space` is the type of action space
- `observation_space` is the type of observation space
- `reset` is a method that sets the game to the beginning, returns the observation and info.
- `step` is a method that advances the game, for example for one frame. Returns the observation, reward, if the game is terminated, if the game is truncated and info.
- `close` is a method that closes the game (deinitilizing things perhaps and stuff)

Proposed Typescript equivalent:

gymnasium/core.ts

```ts
abstract class Env {
  protected renderMode: str;
  public actionSpace: ActSpace;
  public observationSpace: ObsSpace;

  abstract reset(): [tf.Tensor, {}];
  abstract async step(
    action: number
  ): [tf.Tensor, number, boolean, boolean, {}]; // Action is number for now
  abstract async render(): void;
  abstract close(): void;
}
```

### CartPole Env

https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py

Class CartPoleEnv in general is like:

```py
class CartPoleEnv(gym.Env):
    def __init__(self, sutton_barto_reward, render_mode):
        ...

    def step():
        ...

    def reset():
        ...

    def render():
        ...
```

- `sutton_barto_reward` changes the way rewards work
- `render_mode` Chooses render mode between `human` and `rgb_array`

It basically an implementation of the abstract class `gym.Env`.

Proposed Typescript equivalent:

gymnasium/envs/classic_control/cartpole.ts

```ts
class CartPoleEnv extends Env {
    private suttonBartoReward: boolean;

    constructor(suttonBartoReward: boolean = false, renderMode: str = "rgb_array") {
        this.suttonBartoReward = suttonBartoReward;
        this.renderMode = renderMode;
        this.actionSpace = new Discrete(n=2, start=0);
        this.observationSpace = new Box(low=-Infinity, high=Infinity, shape=[4], dtype="float32"); // Not quite but it's alright for now

        ... // A bunch of constants and configurations
    }

    reset(): [tf.Tensor, {}] {
        ...
    }

    async step(action: number): [tf.Tensor, number, boolean, boolean, {}] {
        ...
    }

    async render(): void {
        ...
    }

    close(): void {
        // Nothing
    }
}
```

### References:

https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/types.ts (for tensorflow data types, inteface SingleValueMap)

https://gymnasium.farama.org

https://github.com/Farama-Foundation/Gymnasium

https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

https://github.com/zemlyansky/ppo-tfjs/
