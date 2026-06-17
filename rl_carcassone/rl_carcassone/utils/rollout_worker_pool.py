import multiprocessing as mp
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rl_carcassone.data import save_episode_atomic
from rl_carcassone.env import CarcassonneEnv
from rl_carcassone.policy.utils import build_policy
from rl_carcassone.utils.logger import EventLogger

from .play_single_episode import play_single_episode


@dataclass(frozen=True)
class CollectedEpisodePaths:
    train_paths: List[Path]
    val_paths: List[Path]


class RolloutWorker:
    """Command handler executed inside one rollout subprocess.

    A worker owns one environment and one actor copy for its whole lifetime.
    The parent process communicates through pipes: ``update_weights`` reloads
    actor weights, ``collect_episodes`` writes complete episodes to disk and
    returns paths, and ``stop`` exits the subprocess.
    """

    def __init__(
        self,
        worker_id: int,
        config: Dict[str, Any],
        command_pipe: Connection,
        result_pipe: Connection,
    ) -> None:
        self.worker_id = worker_id
        self.config = config
        self.command_pipe = command_pipe
        self.result_pipe = result_pipe
        self.env = None
        self.actor = None

    @classmethod
    def run_process(
        cls,
        worker_id: int,
        config: Dict[str, Any],
        command_pipe: Connection,
        result_pipe: Connection,
    ) -> None:
        """Create and run a worker inside a spawned subprocess."""
        cls(
            worker_id=worker_id,
            config=config,
            command_pipe=command_pipe,
            result_pipe=result_pipe,
        ).run()

    def run(self) -> None:
        """Run the blocking worker command loop."""
        self._setup()
        try:
            while True:
                command = self.command_pipe.recv()
                command_type = command["type"]
                if command_type == "stop":
                    self.result_pipe.send({"type": "stopped", "worker_id": self.worker_id})
                    return
                if command_type == "update_weights":
                    self._update_weights(command)
                elif command_type == "collect_episodes":
                    self._collect_episodes(command)
                else:
                    raise ValueError(f"Unknown worker command type: {command_type!r}")
        except Exception as exc:
            self.result_pipe.send(
                {
                    "type": "error",
                    "worker_id": self.worker_id,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    def _setup(self) -> None:
        environment_config = self.config["environment"]
        experiment_config = self.config["experiment"]
        algorithm_config = self.config["algorithm"]

        self.env = CarcassonneEnv(
            seed=environment_config["seed"] + self.worker_id,
            n_opponents=environment_config["n_opponents"],
        )
        policy = build_policy(self.env, algorithm_config, device=experiment_config["worker_device"])
        self.actor = policy.actor
        self.actor.eval()

    def _update_weights(self, command: Dict[str, Any]) -> None:
        self.actor.load(Path(command["actor_dir"]))
        self.actor.eval()
        self.result_pipe.send(
            {
                "type": "weights_updated",
                "worker_id": self.worker_id,
                "policy_version": command["policy_version"],
            }
        )

    def _collect_episodes(self, command: Dict[str, Any]) -> None:
        train_paths = self._collect_split(
            policy_version=command["policy_version"],
            split="train",
            n_episodes=command["n_train_episodes"],
            deterministic=False,
            seed=command["train_seed"],
            storage_dir=Path(command["storage_dir"]),
        )
        val_paths = self._collect_split(
            policy_version=command["policy_version"],
            split="val",
            n_episodes=command["n_val_episodes"],
            deterministic=True,
            seed=command["val_seed"],
            storage_dir=Path(command["storage_dir"]),
        )
        self.result_pipe.send(
            {
                "type": "episodes_collected",
                "worker_id": self.worker_id,
                "policy_version": command["policy_version"],
                "train_paths": [str(path) for path in train_paths],
                "val_paths": [str(path) for path in val_paths],
            }
        )

    def _collect_split(
        self,
        policy_version: int,
        split: str,
        n_episodes: int,
        deterministic: bool,
        seed: int,
        storage_dir: Path,
    ) -> List[Path]:
        paths = []
        split_dir = storage_dir.joinpath(
            f"policy_{policy_version:06d}",
            split,
            f"worker_{self.worker_id:03d}",
        )
        for episode_index in range(n_episodes):
            episode_seed = seed + episode_index
            episode = play_single_episode(
                env=self.env,
                actor=self.actor,
                deterministic=deterministic,
                seed=episode_seed,
            )
            path = split_dir.joinpath(f"episode_{episode_index:06d}.pt")
            save_episode_atomic(
                episode=episode,
                path=path,
                metadata={
                    "worker_id": self.worker_id,
                    "policy_version": policy_version,
                    "split": split,
                    "episode_index": episode_index,
                    "seed": episode_seed,
                    "n_steps": len(episode),
                },
            )
            paths.append(path)
        return paths


class RolloutWorkerPool:
    """Persistent subprocess pool for full-episode rollout collection.

    Each worker is a long-lived subprocess with its own environment and actor
    copy. The pool API is synchronous: public calls return only after every
    worker has replied. Workers write complete episodes to disk and send only
    file paths back to the trainer, which then loads those episodes into RAM.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        run_dir: Union[str, Path],
        n_workers: int,
        storage_dirname: str,
        event_logger: Optional[EventLogger] = None,
    ) -> None:
        if n_workers < 1:
            raise ValueError("RolloutWorkerPool requires n_workers >= 1.")
        self.config = config
        self.run_dir = Path(run_dir)
        self.n_workers = n_workers
        self.event_logger = event_logger
        self.storage_dir = self.run_dir.joinpath(storage_dirname)
        self._context = mp.get_context("spawn")
        self._workers = []
        self._command_pipes: List[Connection] = []
        self._result_pipes: List[Connection] = []
        self._start_workers()

    def __enter__(self) -> "RolloutWorkerPool":
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        self.close()

    def update_weights(self, policy_version: int, actor_dir: Union[str, Path]) -> None:
        """Ask every worker to load actor weights and wait for confirmation.

        Args:
            policy_version: Version label attached to subsequent worker
                responses and stored episodes. It is used for consistency
                checks, not for loading weights.
            actor_dir: Directory containing ``actor.pt``. In the trainer this
                is currently the run's ``latest`` checkpoint directory.

        This method blocks until all workers send ``weights_updated`` for the
        requested policy version.
        """
        for pipe in self._command_pipes:
            pipe.send(
                {
                    "type": "update_weights",
                    "policy_version": policy_version,
                    "actor_dir": str(actor_dir),
                }
            )
        self._collect_responses("weights_updated", policy_version)

    def collect_episodes(
        self,
        policy_version: int,
        n_train_episodes: int,
        n_val_episodes: int,
        seed: int,
    ) -> CollectedEpisodePaths:
        """Collect complete train/validation episodes and return their paths.

        Episode counts are per-worker, following the stable-baselines3 style:
        with ``n_train_episodes=20`` and five workers, the pool collects 100
        training episodes in total. Each worker writes its assigned episodes
        under the pool's intermediate storage directory and returns paths only
        after writing has completed.

        This method blocks until every worker sends ``episodes_collected`` for
        the requested policy version. The trainer should read the returned
        paths only after this method returns; it does not need to watch the
        storage directory while workers are running.
        """
        for worker_id, pipe in enumerate(self._command_pipes):
            worker_seed = seed + worker_id * 100_000
            pipe.send(
                {
                    "type": "collect_episodes",
                    "policy_version": policy_version,
                    "storage_dir": str(self.storage_dir),
                    "n_train_episodes": n_train_episodes,
                    "n_val_episodes": n_val_episodes,
                    "train_seed": worker_seed,
                    "val_seed": worker_seed + 50_000,
                }
            )

        responses = self._collect_responses("episodes_collected", policy_version)
        train_paths = []
        val_paths = []
        for response in sorted(responses, key=lambda item: item["worker_id"]):
            train_paths.extend(Path(path) for path in response["train_paths"])
            val_paths.extend(Path(path) for path in response["val_paths"])
        return CollectedEpisodePaths(train_paths=train_paths, val_paths=val_paths)

    def close(self) -> None:
        """Stop workers and terminate any process that does not exit promptly."""
        for pipe in self._command_pipes:
            if not pipe.closed:
                try:
                    pipe.send({"type": "stop"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for pipe in self._result_pipes:
            if not pipe.closed:
                try:
                    if pipe.poll(timeout=1):
                        pipe.recv()
                except EOFError:
                    pass
        for process in self._workers:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    def _start_workers(self) -> None:
        for worker_id in range(self.n_workers):
            parent_command, child_command = self._context.Pipe()
            parent_result, child_result = self._context.Pipe(duplex=False)
            process = self._context.Process(
                target=RolloutWorker.run_process,
                args=(worker_id, self.config, child_command, child_result),
            )
            process.start()
            child_command.close()
            child_result.close()
            self._command_pipes.append(parent_command)
            self._result_pipes.append(parent_result)
            self._workers.append(process)

    def _collect_responses(self, response_type: str, policy_version: int) -> List[Dict[str, Any]]:
        responses = []
        for pipe in self._result_pipes:
            try:
                response = pipe.recv()
            except EOFError as exc:
                raise RuntimeError("Rollout worker exited without sending a response.") from exc
            if response["type"] == "error":
                raise RuntimeError(
                    f"Rollout worker {response['worker_id']} failed: {response['message']}\n" f"{response['traceback']}"
                )
            if response["type"] != response_type:
                raise RuntimeError(f"Unexpected worker response: {response!r}")
            if response.get("policy_version") != policy_version:
                raise RuntimeError(f"Unexpected policy version in worker response: {response!r}")
            if self.event_logger is not None:
                if response_type == "weights_updated":
                    self.event_logger.log("worker_weights_updated", worker_id=response["worker_id"])
                elif response_type == "episodes_collected":
                    self.event_logger.log("worker_episodes_collected", worker_id=response["worker_id"])
            responses.append(response)
        return responses
