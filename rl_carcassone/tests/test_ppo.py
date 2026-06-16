from functools import partial

import torch

from rl_carcassone.algorithm import train_ppo
from rl_carcassone.data import Episode, Episodes
from rl_carcassone.env import CarcassonneEnv
from rl_carcassone.policy import AsymmetricActorCriticPolicy
from rl_carcassone.policy.actor import Actor
from rl_carcassone.policy.critic import Critic


def test_train_ppo_smoke_updates_on_short_rollout():
    env = CarcassonneEnv(seed=67, n_opponents=1)
    policy = AsymmetricActorCriticPolicy(
        actor_builder=partial(
            Actor,
            env.observation_space,
            env.action_space,
            hidden_dim=16,
            num_layers=1,
            heads=2,
        ),
        critic_builder=partial(
            Critic,
            env.observation_space,
            hidden_dim=16,
            num_layers=1,
            heads=2,
        ),
        actor_optimizer_kwargs={"lr": 1e-3},
        critic_optimizer_kwargs={"lr": 1e-3},
    )
    actor = policy.actor

    observation, _ = env.reset()
    states = []
    actions = []
    logprobs = []
    rewards = []
    infos = []
    for _ in range(2):
        with torch.no_grad():
            action, logprob = actor.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(int(action))
        states.append(observation)
        actions.append(int(action))
        logprobs.append(float(logprob))
        rewards.append(float(reward))
        infos.append(info)
        observation = next_observation
        if terminated or truncated:
            break

    episodes = Episodes(
        [
            Episode(
                states=states,
                actions=actions,
                logprobs=logprobs,
                rewards=rewards,
                infos=infos,
            )
        ]
    )

    stats = train_ppo(policy, episodes, n_epochs=1, batch_size=2, to_train_actor=False)

    assert episodes.episodes[0].values is not None
    assert episodes.episodes[0].advantages is not None
    assert episodes.episodes[0].returns is not None
    assert "policy_loss" in stats
    assert "value_loss" in stats
    assert stats["actor_n_updates"] == 0
