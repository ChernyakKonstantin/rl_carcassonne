"""Small PPO trainer for Carcassonne's dynamic discrete action space."""

import logging
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_carcassone.data import Episodes
from rl_carcassone.policy.actor import Actor
from rl_carcassone.policy.assymetric_actor_critic import AsymmetricActorCriticPolicy
from rl_carcassone.policy.critic import Critic
from rl_carcassone.utils import explained_variance

ActorCriticPolicy = AsymmetricActorCriticPolicy


def estimate_values(critic: Critic, episodes: Episodes) -> None:
    critic.eval()
    with torch.no_grad():
        for episode in episodes:
            episode.values = [float(critic(state).detach().cpu().item()) for state in episode.states]
    critic.train()


def _critic_values(critic: Critic, states: Sequence[Dict[str, Any]]) -> torch.Tensor:
    return torch.stack([critic(state).reshape(()) for state in states])


def _actor_evaluation(
    actor: Actor,
    states: Sequence[Dict[str, Any]],
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logprobs = []
    entropies = []
    for state, action in zip(states, actions):
        logprob, entropy, _ = actor.evaluate_action(state, action)
        logprobs.append(logprob.reshape(()))
        entropies.append(entropy.reshape(()))
    return torch.stack(logprobs), torch.stack(entropies)


def _select(sequence: Sequence[Any], indices: np.ndarray) -> list[Any]:
    return [sequence[int(index)] for index in indices]


def train_critic(
    critic: Critic,
    optimizer: torch.optim.Optimizer,
    episodes: Episodes,
    batch_size: int,
    n_epochs: int,
    clip_range_vf: float | None,
    max_grad_norm: float,
) -> Dict[str, Any]:
    total_samples = episodes.total_steps
    if total_samples == 0:
        raise ValueError("Cannot train PPO on an empty rollout.")

    train_size = max(1, int(round(total_samples * 0.85)))
    sample_indices = np.arange(total_samples)
    np.random.shuffle(sample_indices)
    train_indices = sample_indices[:train_size]
    val_indices = sample_indices[train_size:]

    states = episodes.states
    old_values = episodes.values.to(critic.device)
    returns = episodes.returns.to(critic.device)
    expl_var_before_training = explained_variance(episodes.values, episodes.returns)

    per_epoch_train_loss = []
    per_epoch_val_loss = []
    per_epoch_val_explained_variance = []
    train_explained_variance = np.nan
    epoch = 0

    for epoch in range(n_epochs):
        train_losses = []
        np.random.shuffle(train_indices)
        for start in range(0, len(train_indices), batch_size):
            selected_indices = train_indices[start : start + batch_size]
            selected_states = _select(states, selected_indices)
            selected_old_values = old_values[selected_indices]
            selected_returns = returns[selected_indices]

            new_values = _critic_values(critic, selected_states)
            if clip_range_vf is not None:
                new_values = selected_old_values + torch.clamp(
                    new_values - selected_old_values,
                    -clip_range_vf,
                    clip_range_vf,
                )
            value_loss = F.mse_loss(new_values, selected_returns)

            optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            optimizer.step()

            train_losses.append(float(value_loss.detach().cpu().item()))
        per_epoch_train_loss.append(float(np.mean(train_losses)))

        with torch.no_grad():
            val_losses = []
            val_values = []
            val_returns = []
            for start in range(0, len(val_indices), batch_size):
                selected_indices = val_indices[start : start + batch_size]
                selected_states = _select(states, selected_indices)
                selected_returns = returns[selected_indices]
                new_values = _critic_values(critic, selected_states)
                value_loss = F.mse_loss(new_values, selected_returns)
                val_losses.append(float(value_loss.detach().cpu().item()))
                val_values.extend(new_values.detach().cpu())
                val_returns.extend(selected_returns.detach().cpu())

            if val_losses:
                per_epoch_val_loss.append(float(np.mean(val_losses)))
                per_epoch_val_explained_variance.append(
                    float(explained_variance(torch.stack(val_values), torch.stack(val_returns)))
                )
            else:
                per_epoch_val_loss.append(np.nan)
                per_epoch_val_explained_variance.append(np.nan)

            train_values = _critic_values(critic, _select(states, train_indices)).detach().cpu()
            train_returns = returns[train_indices].detach().cpu()
            train_explained_variance = float(explained_variance(train_values, train_returns))

    return {
        "critic_lr": optimizer.param_groups[0]["lr"],
        "value_loss": per_epoch_val_loss[-1],
        "explained_variance_before_training": float(expl_var_before_training),
        "explained_variance_train": train_explained_variance,
        "explained_variance_val": per_epoch_val_explained_variance[-1],
        "critic_n_updates": epoch + 1,
        "per_epoch_critic_data": {
            "per_epoch_train_loss": per_epoch_train_loss,
            "per_epoch_val_loss": per_epoch_val_loss,
            "per_epoch_val_explained_variance": per_epoch_val_explained_variance,
        },
    }


def train_actor(
    actor: Actor,
    optimizer: torch.optim.Optimizer,
    episodes: Episodes,
    ent_coef: float = 0.0,
    n_epochs: int = 10,
    batch_size: int = 64,
    clip_range: float = 0.2,
    normalize_advantage: bool = False,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
    **kwargs,
) -> Dict[str, Any]:
    total_samples = episodes.total_steps
    if total_samples == 0:
        raise ValueError("Cannot train PPO on an empty rollout.")

    states = episodes.states
    old_logprobs = episodes.logprobs.to(actor.device)
    actions = episodes.actions.to(actor.device)
    advantages = episodes.advantages.to(actor.device)
    indices = np.arange(total_samples)

    clipfracs = []
    pg_losses = []
    entropy_losses = []
    old_approx_kl = torch.tensor(0.0, device=actor.device)
    approx_kl = torch.tensor(0.0, device=actor.device)
    epoch = 0

    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        for start in range(0, total_samples, batch_size):
            selected_indices = indices[start : start + batch_size]
            if len(selected_indices) == 0:
                continue

            selected_states = _select(states, selected_indices)
            selected_actions = actions[selected_indices]
            selected_old_logprobs = old_logprobs[selected_indices]
            selected_advantages = advantages[selected_indices]

            new_logprobs, entropies = _actor_evaluation(actor, selected_states, selected_actions)
            logratio = new_logprobs - selected_old_logprobs
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(float(((ratio - 1.0).abs() > clip_range).float().mean().cpu().item()))

            if normalize_advantage and len(selected_advantages) > 1:
                selected_advantages = (selected_advantages - selected_advantages.mean()) / (
                    selected_advantages.std() + 1e-8
                )

            pg_loss1 = selected_advantages * ratio
            pg_loss2 = selected_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            pg_loss = -torch.min(pg_loss1, pg_loss2).mean()
            entropy_loss = -entropies.mean()
            loss = pg_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            optimizer.step()

            pg_losses.append(float(pg_loss.detach().cpu().item()))
            entropy_losses.append(float(entropy_loss.detach().cpu().item()))

        if target_kl is not None and approx_kl > target_kl:
            logging.getLogger(name="PPO_Trainer").info(
                f"approx_kl={float(approx_kl.detach().cpu().item())} > target_kl={target_kl}, "
                f"exit training step at epoch={epoch}"
            )
            break

    return {
        "actor_lr": optimizer.param_groups[0]["lr"],
        "policy_loss": float(np.mean(pg_losses)) if pg_losses else np.nan,
        "entropy_loss": float(np.mean(entropy_losses)) if entropy_losses else np.nan,
        "old_approx_kl": float(old_approx_kl.detach().cpu().item()),
        "approx_kl": float(approx_kl.detach().cpu().item()),
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else np.nan,
        "actor_n_updates": epoch + 1,
    }


def train_ppo(
    policy: ActorCriticPolicy,
    episodes: Episodes,
    to_train_actor: bool = True,
    gamma: float = 0.99,
    gae_lambda: float = 0.9,
    ent_coef: float = 0.0,
    n_epochs: int = 10,
    batch_size: int = 64,
    clip_range: float = 0.2,
    normalize_advantage: bool = False,
    clip_range_vf: float | None = 0.2,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
    **kwargs,
) -> Dict[str, Any]:
    estimate_values(policy.critic, episodes)
    episodes.calculate_advantages(gamma, gae_lambda)
    episodes.calculate_returns()

    critic_statistics = train_critic(
        critic=policy.critic,
        optimizer=policy.critic_optimizer,
        episodes=episodes,
        batch_size=batch_size,
        n_epochs=n_epochs,
        clip_range_vf=clip_range_vf,
        max_grad_norm=max_grad_norm,
    )
    if to_train_actor:
        actor_statistics = train_actor(
            actor=policy.actor,
            optimizer=policy.actor_optimizer,
            episodes=episodes,
            ent_coef=ent_coef,
            n_epochs=n_epochs,
            batch_size=batch_size,
            clip_range=clip_range,
            normalize_advantage=normalize_advantage,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
        )
    else:
        actor_statistics = {
            "actor_lr": None,
            "policy_loss": None,
            "entropy_loss": None,
            "old_approx_kl": None,
            "approx_kl": None,
            "clipfrac": None,
            "actor_n_updates": 0,
        }

    return {
        **actor_statistics,
        **critic_statistics,
    }
