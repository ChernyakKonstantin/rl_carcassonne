from typing import Any

import numpy as np
from gymnasium import spaces


class DynamicDiscrete(spaces.Space):
    """Discrete integer action space without a fixed action count.

    This space is intended for environments where the real legal action set is
    carried by the observation and can change on every step. It validates only
    the stable part of the contract: an action is a non-negative integer index.
    The environment still has to reject indices outside the current legal
    action sequence.
    """

    def __init__(self):
        super().__init__(shape=(), dtype=np.int64)

    def contains(self, x: Any) -> bool:
        if isinstance(x, bool):
            return False
        if isinstance(x, np.integer):
            return int(x) >= 0
        if isinstance(x, int):
            return x >= 0
        return False

    def sample(self):
        raise ValueError("DynamicDiscrete cannot sample without a known action count.")

    def __repr__(self) -> str:
        return "DynamicDiscrete()"
