from pathlib import Path

from rl_carcassone.env import CarcassonneEnv
from rl_carcassone.policy.utils import build_policy
from rl_carcassone.utils import RolloutWorkerPool
from rl_carcassone.utils.config import load_config


class FakeCommandPipe:
    def __init__(self):
        self.commands = []

    def send(self, command):
        self.commands.append(command)


class FakeResultPipe:
    def __init__(self, response):
        self.response = response

    def recv(self):
        return self.response


def test_rollout_worker_pool_updates_weights_and_collects_empty_batch(tmp_path):
    config = load_config(Path("config/ppo_smoke.yaml"))
    config["collection"]["n_workers"] = 1
    env = CarcassonneEnv(
        seed=config["environment"]["seed"],
        n_opponents=config["environment"]["n_opponents"],
    )
    policy = build_policy(env, config["algorithm"], device=config["experiment"]["trainer_device"])
    policy.save(tmp_path.joinpath("latest"))

    pool = RolloutWorkerPool(
        config=config,
        run_dir=tmp_path,
        n_workers=config["collection"]["n_workers"],
        storage_dirname=config["collection"]["storage_dirname"],
    )
    try:
        pool.update_weights(policy_version=0, actor_dir=tmp_path.joinpath("latest"))
        paths = pool.collect_episodes(
            policy_version=0,
            n_train_episodes=0,
            n_val_episodes=0,
            seed=config["environment"]["seed"],
        )
    finally:
        pool.close()

    assert paths.train_paths == []
    assert paths.val_paths == []


def test_rollout_worker_pool_episode_counts_are_per_worker(tmp_path):
    command_pipes = [FakeCommandPipe(), FakeCommandPipe(), FakeCommandPipe()]
    result_pipes = [
        FakeResultPipe(
            {
                "type": "episodes_collected",
                "worker_id": worker_id,
                "policy_version": 7,
                "train_paths": [],
                "val_paths": [],
            }
        )
        for worker_id in range(3)
    ]
    pool = RolloutWorkerPool.__new__(RolloutWorkerPool)
    pool.n_workers = 3
    pool.storage_dir = tmp_path
    pool._command_pipes = command_pipes
    pool._result_pipes = result_pipes

    pool.collect_episodes(
        policy_version=7,
        n_train_episodes=20,
        n_val_episodes=4,
        seed=100,
    )

    for command_pipe in command_pipes:
        command = command_pipe.commands[0]
        assert command["n_train_episodes"] == 20
        assert command["n_val_episodes"] == 4
