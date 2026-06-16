from importlib import import_module
from typing import Type


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
