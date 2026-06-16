import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Union

from .episode import Episode, Episodes


def save_episode_atomic(
    episode: Episode,
    path: Union[str, Path],
    metadata: Dict[str, Any],
) -> Path:
    """Persist one episode with metadata using an atomic rename.

    The episode is first serialized to ``<path>.tmp`` and is then moved to
    ``path`` with ``os.replace``. This gives the storage layer a simple
    invariant: a visible ``.pt`` episode file is complete. The rollout trainer
    does not rely on this rename for synchronization with workers; it waits for
    explicit worker responses before reading any paths.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {"metadata": dict(metadata), "episode": episode}

    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    return path


def load_episode(path: Union[str, Path]) -> Episode:
    """Load one episode saved by ``save_episode_atomic``.

    This returns only the ``Episode`` payload. Metadata is currently stored for
    inspection/debugging and future dataset tooling, but is not used by PPO
    training.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["episode"]


def load_episodes(paths: Iterable[Union[str, Path]]) -> Episodes:
    """Load a sequence of episode files into an ``Episodes`` container."""
    return Episodes(load_episode(path) for path in paths)
