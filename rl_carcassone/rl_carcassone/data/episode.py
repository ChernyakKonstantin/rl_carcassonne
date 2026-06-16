from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch

Observation = Mapping[str, Any]


@dataclass
class Episode:
    states: List[Observation]
    actions: List[int]
    logprobs: List[float]
    rewards: List[float]
    infos: List[Dict[str, Any]]
    values: Optional[List[float]] = None
    advantages: Optional[List[float]] = None
    returns: Optional[List[float]] = None

    def __len__(self) -> int:
        return len(self.actions)

    def calculate_advantages(self, gamma: float, gae_lambda: float) -> None:
        if self.values is None:
            raise ValueError("Episode values must be set before calculating advantages.")
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        advantages = np.zeros(len(self), dtype=np.float32)
        last_gae_lam = 0.0
        for step in reversed(range(len(self))):
            next_value = 0.0 if step == len(self) - 1 else values[step + 1]
            delta = rewards[step] + gamma * next_value - values[step]
            last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
            advantages[step] = last_gae_lam
        self.advantages = advantages.tolist()

    def calculate_returns(self) -> None:
        if self.advantages is None or self.values is None:
            raise ValueError("Episode advantages and values must be set before calculating returns.")
        self.returns = (
            np.asarray(self.advantages, dtype=np.float32) + np.asarray(self.values, dtype=np.float32)
        ).tolist()

    @property
    def actions_as_torch(self) -> torch.Tensor:
        return torch.as_tensor(self.actions, dtype=torch.long)

    @property
    def logprobs_as_torch(self) -> torch.Tensor:
        return torch.as_tensor(self.logprobs, dtype=torch.float32)

    @property
    def rewards_as_torch(self) -> torch.Tensor:
        return torch.as_tensor(self.rewards, dtype=torch.float32)

    @property
    def values_as_torch(self) -> torch.Tensor:
        if self.values is None:
            raise ValueError("Episode values are not set.")
        return torch.as_tensor(self.values, dtype=torch.float32)

    @property
    def advantages_as_torch(self) -> torch.Tensor:
        if self.advantages is None:
            raise ValueError("Episode advantages are not set.")
        return torch.as_tensor(self.advantages, dtype=torch.float32)

    @property
    def returns_as_torch(self) -> torch.Tensor:
        if self.returns is None:
            raise ValueError("Episode returns are not set.")
        return torch.as_tensor(self.returns, dtype=torch.float32)


class Episodes:
    def __init__(self, episodes: Iterable[Episode]):
        self.episodes = list(episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> Episode:
        return self.episodes[index]

    def __iter__(self):
        return iter(self.episodes)

    @property
    def total_steps(self) -> int:
        return sum(len(episode) for episode in self.episodes)

    def calculate_advantages(self, gamma: float, gae_lambda: float) -> None:
        for episode in self.episodes:
            episode.calculate_advantages(gamma, gae_lambda)

    def calculate_returns(self) -> None:
        for episode in self.episodes:
            episode.calculate_returns()

    @property
    def states(self) -> List[Observation]:
        return [state for episode in self.episodes for state in episode.states]

    @property
    def actions(self) -> torch.Tensor:
        return torch.cat([episode.actions_as_torch for episode in self.episodes])

    @property
    def logprobs(self) -> torch.Tensor:
        return torch.cat([episode.logprobs_as_torch for episode in self.episodes])

    @property
    def rewards(self) -> torch.Tensor:
        return torch.cat([episode.rewards_as_torch for episode in self.episodes])

    @property
    def values(self) -> torch.Tensor:
        return torch.cat([episode.values_as_torch for episode in self.episodes])

    @property
    def advantages(self) -> torch.Tensor:
        return torch.cat([episode.advantages_as_torch for episode in self.episodes])

    @property
    def returns(self) -> torch.Tensor:
        return torch.cat([episode.returns_as_torch for episode in self.episodes])

    def get_statistics(self) -> Dict[str, float]:
        episode_lengths = [len(episode) for episode in self.episodes]
        episode_rewards = [float(np.sum(episode.rewards)) for episode in self.episodes]
        legal_action_counts = [
            info["legal_action_count"]
            for episode in self.episodes
            for info in episode.infos
            if "legal_action_count" in info
        ]
        return {
            "ep_len_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "ep_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "ep_reward_min": float(np.min(episode_rewards)) if episode_rewards else 0.0,
            "ep_reward_max": float(np.max(episode_rewards)) if episode_rewards else 0.0,
            "legal_action_count_mean": float(np.mean(legal_action_counts)) if legal_action_counts else 0.0,
            "total_steps": float(self.total_steps),
        }
