import { ActionWrapper, Env } from '../core';

export class StickyAction<ObsType, ActType> extends ActionWrapper<
  ActType,
  ObsType,
  ActType
> {
  private readonly repeatActionProbability: number;
  private readonly repeatActionDuration: [number, number];

  private lastAction: ActType | null = null;
  private isStickyActions: boolean = false;
  private numRepeats: number = 0;
  private repeatsTaken: number = 0;

  constructor(
    env: Env<ObsType, ActType>,
    repeatActionProbability: number,
    repeatActionDuration: number | [number, number] = 1
  ) {
    super(env);

    if (repeatActionProbability < 0 || repeatActionProbability >= 1) {
      throw new RangeError(
        `StickyAction repeat probability must be in [0, 1). Received ${repeatActionProbability}.`
      );
    }

    const [lower, upper] = (() => {
      if (typeof repeatActionDuration === 'number') {
        return [repeatActionDuration, repeatActionDuration] as const;
      }

      return [repeatActionDuration[0], repeatActionDuration[1]] as const;
    })();

    if (!Number.isInteger(lower) || !Number.isInteger(upper)) {
      throw new TypeError(
        `StickyAction repeat action duration must be integer values. Received ${repeatActionDuration}.`
      );
    }

    if (lower < 1 || upper < 1 || lower > upper) {
      throw new RangeError(
        `StickyAction repeat action duration must be integers >= 1 and lower <= upper. Received ${repeatActionDuration}.`
      );
    }

    this.repeatActionProbability = repeatActionProbability;
    this.repeatActionDuration = [lower, upper];
  }

  reset(options?: Record<string, any>): [ObsType, Record<string, any> | null] {
    this.lastAction = null;
    this.isStickyActions = false;
    this.numRepeats = 0;
    this.repeatsTaken = 0;

    return super.reset(options);
  }

  actionTransform(action: ActType): ActType {
    let nextAction = action;
    // TODO: Use seeds when implemented
    if (
      this.isStickyActions ||
      (this.lastAction !== null && Math.random() < this.repeatActionProbability)
    ) {
      if (this.numRepeats === 0) {
        const [lower, upper] = this.repeatActionDuration;
        this.numRepeats =
          Math.floor(Math.random() * (upper - lower + 1)) + lower;
      }
      nextAction = this.lastAction as ActType;
      this.isStickyActions = true;
      this.repeatsTaken += 1;
    }

    if (this.isStickyActions && this.numRepeats === this.repeatsTaken) {
      this.isStickyActions = false;
      this.numRepeats = 0;
      this.repeatsTaken = 0;
    }

    this.lastAction = nextAction;
    return nextAction;
  }
}
