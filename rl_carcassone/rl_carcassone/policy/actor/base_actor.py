from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

import torch
import torch as th
import torch.nn as nn

from rl_carcassone.env.spaces import DynamicDiscrete

from .distribution import CategoricalActionDistribution


class BaseActor(nn.Module):
    def __init__(self, action_space: DynamicDiscrete):
        super().__init__()
        self.action_space: DynamicDiscrete = action_space

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def complete_setup(self, latent_dim_pi: int) -> None:
        self.action_dist = CategoricalActionDistribution()
        self.action_net = nn.Linear(latent_dim_pi, 1)

    def forward(
        self,
        observation: Mapping[str, Any],
        return_latent_pi: bool = False,
    ) -> Union[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        latent_pi = self.features_extractor(observation)
        action_logits = self.action_net(latent_pi).squeeze(-1)
        if return_latent_pi:
            return action_logits, latent_pi
        return action_logits

    def get_action(
        self,
        obs: Mapping[str, Any],
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor]:
        distribution = self.get_action_dist(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, log_prob

    def evaluate_action(
        self,
        obs: Mapping[str, Any],
        actions: th.Tensor | int,
    ) -> tuple[th.Tensor, th.Tensor, CategoricalActionDistribution]:
        distribution = self.get_action_dist(obs)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy, distribution

    def get_action_dist(self, observation: Mapping[str, Any]) -> CategoricalActionDistribution:
        action_logits = self.forward(observation)
        return self.action_dist.proba_distribution(action_logits)

    def load(self, load_dir: Path) -> None:
        self.load_state_dict(
            torch.load(load_dir.joinpath("actor.pt"), map_location=self.device, weights_only=True),
            strict=True,
        )

    def save(self, save_dir: Union[str, Path]) -> None:
        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)
        torch.save(self.state_dict(), save_dir.joinpath("actor.pt"))
