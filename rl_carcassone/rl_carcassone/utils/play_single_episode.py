from typing import Optional

import torch
from gymnasium import Env

from rl_carcassone.data import Episode
from rl_carcassone.policy.actor import BaseActor


def play_single_episode(
    env: Env,
    actor: BaseActor,
    deterministic: bool = False,
    seed: Optional[int] = None,
) -> Episode:
    """Collect one non-vectorized Carcassonne episode with the current actor."""

    episode = Episode(states=[], actions=[], logprobs=[], rewards=[], infos=[])
    observation, info = env.reset(seed=seed)

    while True:
        with torch.no_grad():
            action_tensor, logprob_tensor = actor.get_action(observation, deterministic=deterministic)

        action = int(action_tensor.detach().cpu().item())
        logprob = float(logprob_tensor.detach().cpu().item())
        next_observation, reward, terminated, truncated, info = env.step(action)

        episode.states.append(observation)
        episode.actions.append(action)
        episode.logprobs.append(logprob)
        episode.rewards.append(float(reward))
        episode.infos.append(dict(info))

        if terminated or truncated:
            episode.infos[-1]["terminal_observation"] = next_observation
            episode.infos[-1]["done_type"] = "terminated" if terminated else "truncated"
            episode.infos[-1]["deterministic"] = deterministic
            break

        observation = next_observation

    return episode
