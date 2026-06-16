from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
import torch as th
import torch.nn as nn


class AsymmetricActorCriticPolicy(nn.Module):
    """Actor-critic container with separate builders and optimizers.

    The current Carcassonne policy uses the same environment observation for
    actor and critic. The asymmetric wrapper is kept as an extension point: a
    future critic can be built with richer training-only state while the actor
    remains constrained to inference-time observations.
    """

    def __init__(
        self,
        actor_builder: Callable[[], nn.Module],
        critic_builder: Callable[[], nn.Module],
        ortho_init: bool = True,
        init_method: str = "orthogonal",
        actor_optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        critic_optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        actor_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        critic_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> None:
        """Create actor, critic, initializers, and optimizers.

        Args:
            actor_builder: Zero-argument callable that returns the actor module.
            critic_builder: Zero-argument callable that returns the critic module.
            ortho_init: Whether to initialize actor/critic modules after construction.
            init_method: Weight initialization method for linear/convolution layers;
                supported values are ``"orthogonal"`` and ``"kaiming"``.
            actor_optimizer_class: Optimizer class used for actor parameters.
            critic_optimizer_class: Optimizer class used for critic parameters.
            actor_optimizer_kwargs: Keyword arguments for the actor optimizer, e.g.
                ``{"lr": 3e-4, "weight_decay": 1e-5}``.
            critic_optimizer_kwargs: Keyword arguments for the critic optimizer.
            device: Device for both neural networks.
            **kwargs: Reserved for config compatibility; currently ignored.
        """
        super().__init__()
        actor_optimizer_kwargs = {} if actor_optimizer_kwargs is None else dict(actor_optimizer_kwargs)
        critic_optimizer_kwargs = {} if critic_optimizer_kwargs is None else dict(critic_optimizer_kwargs)

        self.actor = actor_builder()
        self.critic = critic_builder()

        if ortho_init:
            module_gains = {
                self.actor.features_extractor: np.sqrt(2),
                self.critic.features_extractor: np.sqrt(2),
                self.actor.action_net: 0.01,
                self.critic.value_net: 1.0,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, method=init_method, gain=gain))

        actor_weight_decay = actor_optimizer_kwargs.pop("weight_decay", 0.0)
        self.actor_optimizer = actor_optimizer_class(
            [
                {
                    "params": list(self.actor.parameters()),
                    "weight_decay": actor_weight_decay,
                }
            ],
            **actor_optimizer_kwargs,
        )
        self.critic_optimizer = critic_optimizer_class(
            self.critic.parameters(),
            **critic_optimizer_kwargs,
        )
        self.to(device)

    @staticmethod
    def init_weights(module: nn.Module, method: str = "orthogonal", gain: float = 1.0) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if method == "orthogonal":
                nn.init.orthogonal_(module.weight, gain=gain)
            elif method == "kaiming":
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                module.weight.data.mul_(gain / nn.init.calculate_gain("relu"))
            else:
                raise ValueError(f"Unknown initialization method: {method!r}")
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def load(self, load_dir: Union[Path, str]) -> None:
        load_dir = Path(load_dir)
        self.actor.load(load_dir)
        self.critic.load(load_dir)

    def save(self, save_dir: Union[Path, str]) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.actor.save(save_dir)
        self.critic.save(save_dir)
