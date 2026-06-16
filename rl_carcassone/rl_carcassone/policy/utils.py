from functools import partial
from importlib import import_module
from typing import Type

from gymnasium import Env

from rl_carcassone.policy.assymetric_actor_critic import AsymmetricActorCriticPolicy


def _load_class(module_path: str, class_name: str) -> Type:
    module = import_module(module_path)
    return getattr(module, class_name)


def get_actor_class(module_path: str, class_name: str) -> Type:
    """Return an actor class described by an experiment config.

    Args:
        module_path: Import path of a module that exposes the actor class, for
            example ``"rl_carcassone.policy.actor"``.
        class_name: Name of the actor class inside ``module_path``.
    """
    return _load_class(module_path, class_name)


def get_critic_class(module_path: str, class_name: str) -> Type:
    """Return a critic class described by an experiment config.

    Args:
        module_path: Import path of a module that exposes the critic class, for
            example ``"rl_carcassone.policy.critic"``.
        class_name: Name of the critic class inside ``module_path``.
    """
    return _load_class(module_path, class_name)


def build_policy(env: Env, algorithm_config: dict, device: str) -> AsymmetricActorCriticPolicy:
    """Build the configured actor-critic policy for an environment.

    Args:
        env: Environment that exposes the observation and action spaces used by
            the actor and critic constructors.
        algorithm_config: Experiment ``algorithm`` config section containing
            ``policy_kwargs``.
        device: Device for both neural networks.
    """
    policy_kwargs = algorithm_config["policy_kwargs"]

    actor_kwargs = dict(policy_kwargs["actor_kwargs"])
    actor_class = get_actor_class(
        module_path=actor_kwargs.pop("module_path"),
        class_name=actor_kwargs.pop("class_name"),
    )

    critic_kwargs = dict(policy_kwargs["critic_kwargs"])
    critic_class = get_critic_class(
        module_path=critic_kwargs.pop("module_path"),
        class_name=critic_kwargs.pop("class_name"),
    )

    return AsymmetricActorCriticPolicy(
        actor_builder=partial(
            actor_class,
            env.observation_space,
            env.action_space,
            **actor_kwargs,
        ),
        critic_builder=partial(
            critic_class,
            env.observation_space,
            **critic_kwargs,
        ),
        actor_optimizer_kwargs=policy_kwargs["actor_optimizer_kwargs"],
        critic_optimizer_kwargs=policy_kwargs["critic_optimizer_kwargs"],
        ortho_init=policy_kwargs["ortho_init"],
        init_method=policy_kwargs["init_method"],
        device=device,
    )
