from typing import Optional

import torch as th
from torch.distributions import Categorical


class CategoricalActionDistribution:
    """Categorical distribution over the current legal action candidates."""

    def __init__(self) -> None:
        self.distribution: Optional[Categorical] = None

    def proba_distribution(self, action_logits: th.Tensor) -> "CategoricalActionDistribution":
        if action_logits.ndim != 1:
            raise ValueError(f"Expected 1D action logits, got shape {tuple(action_logits.shape)}.")
        if action_logits.numel() == 0:
            raise ValueError("Cannot build a policy distribution without legal action candidates.")
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self._distribution().log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self._distribution().entropy()

    def sample(self) -> th.Tensor:
        return self._distribution().sample()

    def mode(self) -> th.Tensor:
        return th.argmax(self._distribution().logits, dim=-1)

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()

    def _distribution(self) -> Categorical:
        if self.distribution is None:
            raise RuntimeError("Distribution parameters have not been set.")
        return self.distribution
