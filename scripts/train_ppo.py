import time
from datetime import datetime
from pathlib import Path

import click

from rl_carcassone.algorithm import train_ppo
from rl_carcassone.data import load_episodes
from rl_carcassone.env import CarcassonneEnv
from rl_carcassone.policy.utils import build_policy
from rl_carcassone.utils import EventLogger, ExperimentLogger, RolloutWorkerPool
from rl_carcassone.utils.config import load_config, save_config


def _load_starting_weights(policy, experiment_config: dict) -> None:
    actor_start_from = experiment_config.get("actor_start_from")
    if actor_start_from is not None:
        policy.actor.load(Path(actor_start_from))

    critic_start_from = experiment_config.get("critic_start_from")
    if critic_start_from is not None:
        policy.critic.load(Path(critic_start_from))


@click.command()
@click.option(
    "--config-path",
    type=click.Path(path_type=Path),
    default=Path("config/ppo_baseline.yaml"),
    show_default=True,
)
def main(config_path: Path) -> None:
    config = load_config(config_path)
    experiment_config = config["experiment"]
    environment_config = config["environment"]
    algorithm_config = config["algorithm"]
    collection_config = config["collection"]

    experiment_dir = Path(experiment_config["experiment_dir"])
    trainer_device = experiment_config["trainer_device"]
    run_dir = experiment_dir.joinpath(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir)
    event_logger = EventLogger(run_dir)
    event_logger.log("run_started")

    env = CarcassonneEnv(
        seed=environment_config["seed"],
        n_opponents=environment_config["n_opponents"],
    )
    policy = build_policy(env, algorithm_config, device=trainer_device)
    _load_starting_weights(policy, experiment_config)
    policy.save(run_dir.joinpath("latest"))
    event_logger.log("initial_checkpoint_saved")

    total_iterations = algorithm_config["total_iterations"]
    warmup_iterations = algorithm_config["warmup_iterations"]
    n_train_episodes = algorithm_config["n_train_episodes"]
    n_val_episodes = algorithm_config["n_val_episodes"]
    seed = environment_config["seed"]

    train_kwargs = {
        "gamma": algorithm_config["gamma"],
        "gae_lambda": algorithm_config["gae_lambda"],
        "ent_coef": algorithm_config["ent_coef"],
        "n_epochs": algorithm_config["n_epochs"],
        "batch_size": algorithm_config["batch_size"],
        "clip_range": algorithm_config["clip_range"],
        "normalize_advantage": algorithm_config["normalize_advantage"],
        "clip_range_vf": algorithm_config["clip_range_vf"],
        "max_grad_norm": algorithm_config["max_grad_norm"],
        "target_kl": algorithm_config["target_kl"],
    }

    start_time = time.time()
    logger = ExperimentLogger(run_dir)
    passed_train_episodes = 0
    n_workers = collection_config["n_workers"]
    if n_workers < 1:
        raise ValueError("collection.n_workers must be at least 1.")
    total_train_episodes = total_iterations * n_workers * n_train_episodes
    worker_pool = None
    completed = False
    try:
        worker_pool = RolloutWorkerPool(
            config=config,
            run_dir=run_dir,
            n_workers=n_workers,
            storage_dirname=collection_config["storage_dirname"],
            event_logger=event_logger,
        )
        event_logger.log("worker_pool_started")
        for iteration in range(total_iterations):
            event_logger.log("iteration_started", iteration=iteration)
            iteration_seed = seed + iteration * 10_000
            event_logger.log("update_weights_sent")
            worker_pool.update_weights(policy_version=iteration, actor_dir=run_dir.joinpath("latest"))
            event_logger.log("update_weights_completed")
            event_logger.log("collect_episodes_sent")
            collected_paths = worker_pool.collect_episodes(
                policy_version=iteration,
                n_train_episodes=n_train_episodes,
                n_val_episodes=n_val_episodes,
                seed=iteration_seed,
            )
            event_logger.log("collect_episodes_completed")
            event_logger.log("episodes_load_started")
            train_episodes = load_episodes(collected_paths.train_paths)
            val_episodes = load_episodes(collected_paths.val_paths)
            event_logger.log("episodes_load_completed")

            to_train_actor = iteration >= warmup_iterations
            event_logger.log("ppo_update_started", train_actor=str(to_train_actor).lower())
            train_stats = train_ppo(
                policy,
                train_episodes,
                to_train_actor=to_train_actor,
                **train_kwargs,
            )
            event_logger.log("ppo_update_completed")
            passed_train_episodes += len(train_episodes)
            row = {
                "iteration": iteration,
                "actor_training_enabled": to_train_actor,
                "elapsed_seconds": time.time() - start_time,
                "train_rollout": train_episodes.get_statistics(),
                "val_rollout": val_episodes.get_statistics(),
                "train": train_stats,
            }
            logger.log(
                iteration=iteration,
                val_rollout_statistics=row["val_rollout"],
                train_rollout_statistics=row["train_rollout"],
                train_statistics=row["train"],
                passed_episodes=passed_train_episodes,
                elapsed_time=row["elapsed_seconds"],
            )
            event_logger.log("metrics_logged")
            policy.save(run_dir.joinpath("checkpoints", f"after_{passed_train_episodes}_train_episodes"))
            event_logger.log("checkpoint_saved")
            policy.save(run_dir.joinpath("latest"))
            event_logger.log("latest_checkpoint_saved")
            event_logger.log(
                "iteration_completed",
                iteration=iteration,
                train_episodes_done=passed_train_episodes,
            )

            print(f"train episodes: {passed_train_episodes}/{total_train_episodes}")
        completed = True
    except Exception as exc:
        event_logger.log_exception(exc)
        raise
    finally:
        if worker_pool is not None:
            event_logger.log("worker_pool_closing")
            worker_pool.close()
            event_logger.log("worker_pool_closed")

    if completed:
        event_logger.log("run_completed")
    print("Completed")


if __name__ == "__main__":
    main()
