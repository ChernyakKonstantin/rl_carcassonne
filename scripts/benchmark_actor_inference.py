"""Benchmark actor inference on real CarcassonneEnv observations.

This script is intended for answering a narrow performance question: how the
configured actor's inference time changes as the candidate graph batch grows
during real game trajectories.

It does not measure full rollout collection. Observation construction happens
while samples are collected, before the timed region. The timed region is only
``actor.get_action(...)`` on already collected observations:

- ``raw`` mode mirrors rollout-time actor usage more closely: the actor receives
  the env observation and the feature extractor converts candidate graphs to a
  PyG batch inside the timed call.
- ``prebatched`` mode precomputes that PyG batch before timing, so it isolates
  the neural-network forward pass and action distribution work.

For end-to-end rollout timing use a dedicated ``play_single_episode(...)``
benchmark, because env stepping and candidate graph generation dominate a
different part of the pipeline.
"""

import csv
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import numpy as np
import torch

from rl_carcassone.env import CarcassonneEnv
from rl_carcassone.policy.actor import BaseActor
from rl_carcassone.policy.utils import get_actor_class
from rl_carcassone.utils.config import load_config


@dataclass(frozen=True)
class ObservationSample:
    """One collected actor-decision observation from a real environment run."""

    sample_id: int
    episode: int
    agent_step: int
    observation: Mapping[str, Any]


@dataclass(frozen=True)
class GraphStats:
    """Aggregate size metrics for one observation's action-candidate graphs."""

    candidate_count: int
    total_nodes: int
    total_edges: int
    mean_nodes: float
    mean_edges: float
    max_nodes: int
    max_edges: int


def _parse_csv_option(value: str) -> tuple[str, ...]:
    """Parse comma-separated CLI options while ignoring extra whitespace."""

    return tuple(item.strip() for item in value.split(",") if item.strip())


def _build_actor(env: CarcassonneEnv, algorithm_config: Mapping[str, Any], device: torch.device) -> BaseActor:
    """Build only the configured actor, without critic or optimizers."""

    policy_kwargs = algorithm_config["policy_kwargs"]
    actor_kwargs = dict(policy_kwargs["actor_kwargs"])
    actor_class = get_actor_class(
        module_path=actor_kwargs.pop("module_path"),
        class_name=actor_kwargs.pop("class_name"),
    )
    actor = actor_class(env.observation_space, env.action_space, **actor_kwargs)
    actor.to(device)
    actor.eval()
    return actor


def _available_devices(device_names: Sequence[str]) -> list[torch.device]:
    """Return requested torch devices, skipping unavailable CUDA devices."""

    devices = []
    for device_name in device_names:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"Skipping {device_name}: CUDA is not available.")
            continue
        devices.append(device)
    if not devices:
        raise click.ClickException("No requested benchmark devices are available.")
    return devices


def _collect_samples(
    *,
    seed: int,
    n_opponents: int,
    n_episodes: int,
    max_agent_steps: int | None,
    max_samples: int | None,
    action_selection: str,
) -> list[ObservationSample]:
    """Collect untimed observations by playing real env trajectories.

    The chosen actions are used only to move the environment forward and shape
    the sampled board sizes. They are not part of the timed benchmark. This is
    deliberately separate from timing so graph preview construction does not get
    mixed into actor inference measurements.
    """

    env = CarcassonneEnv(seed=seed, n_opponents=n_opponents)
    rng = np.random.default_rng(seed)
    samples: list[ObservationSample] = []

    for episode in range(n_episodes):
        if max_samples is not None and len(samples) >= max_samples:
            break
        observation, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        agent_step = 0
        while not (terminated or truncated):
            candidate_count = len(observation["action_candidate_graphs"])
            if candidate_count == 0:
                break
            samples.append(
                ObservationSample(
                    sample_id=len(samples),
                    episode=episode,
                    agent_step=agent_step,
                    observation=observation,
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                break
            if max_agent_steps is not None and agent_step + 1 >= max_agent_steps:
                break

            action = _select_collection_action(
                candidate_count=candidate_count,
                action_selection=action_selection,
                rng=rng,
            )
            observation, _, terminated, truncated, _ = env.step(action)
            agent_step += 1

    env.close()
    return samples


def _select_collection_action(
    *,
    candidate_count: int,
    action_selection: str,
    rng: np.random.Generator,
) -> int:
    """Choose the action used to advance the sample-collection trajectory."""

    if action_selection == "first":
        return 0
    if action_selection == "last":
        return candidate_count - 1
    if action_selection == "random":
        return int(rng.integers(candidate_count))
    raise ValueError(f"Unknown action selection: {action_selection}")


def _graph_stats(observation: Mapping[str, Any]) -> GraphStats:
    """Compute candidate-graph batch size metrics written to the CSV."""

    candidate_graphs = observation["action_candidate_graphs"]
    node_counts = [sum(values.shape[0] for values in graph.nodes.values()) for graph in candidate_graphs]
    edge_counts = [sum(values.shape[0] for values in graph.edge_links.values()) for graph in candidate_graphs]
    candidate_count = len(candidate_graphs)
    return GraphStats(
        candidate_count=candidate_count,
        total_nodes=sum(node_counts),
        total_edges=sum(edge_counts),
        mean_nodes=sum(node_counts) / candidate_count,
        mean_edges=sum(edge_counts) / candidate_count,
        max_nodes=max(node_counts),
        max_edges=max(edge_counts),
    )


def _sync_device(device: torch.device) -> None:
    """Synchronize asynchronous CUDA work before stopping a timer."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_observation_for_mode(
    actor: BaseActor,
    observation: Mapping[str, Any],
    mode: str,
    device: torch.device,
) -> Mapping[str, Any]:
    """Prepare an observation for one benchmark mode.

    ``raw`` returns the original env observation. ``prebatched`` adds the
    extractor's internal ``candidate_graphs`` key, which bypasses conversion
    from numpy graph instances to a PyG ``Batch`` during the timed call.
    """

    if mode == "raw":
        return observation
    if mode == "prebatched":
        prepared = dict(observation)
        prepared["candidate_graphs"] = actor.features_extractor.candidate_graphs_to_batch(
            observation["action_candidate_graphs"],
            device=device,
        )
        return prepared
    raise ValueError(f"Unknown benchmark mode: {mode}")


def _time_actor_call(
    actor: BaseActor,
    observation: Mapping[str, Any],
    *,
    deterministic: bool,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[float, float, float]:
    """Time repeated ``actor.get_action`` calls and return mean/stdev/min ms."""

    with torch.inference_mode():
        for _ in range(warmup):
            actor.get_action(observation, deterministic=deterministic)
        _sync_device(device)

        elapsed_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            actor.get_action(observation, deterministic=deterministic)
            _sync_device(device)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(elapsed_ms)
    stdev_ms = statistics.stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0
    return mean_ms, stdev_ms, min(elapsed_ms)


def _benchmark_samples(
    *,
    samples: Sequence[ObservationSample],
    algorithm_config: Mapping[str, Any],
    env_config: Mapping[str, Any],
    devices: Sequence[torch.device],
    modes: Sequence[str],
    actor_dir: Path | None,
    warmup: int,
    repeats: int,
    deterministic: bool,
) -> list[dict[str, Any]]:
    """Run actor timing for every requested sample, device, and mode."""

    rows: list[dict[str, Any]] = []
    env = CarcassonneEnv(seed=env_config["seed"], n_opponents=env_config["n_opponents"])

    for device in devices:
        print(f"Benchmarking device={device}...", flush=True)
        actor = _build_actor(env, algorithm_config, device)
        if actor_dir is not None:
            actor.load(actor_dir)

        for sample in samples:
            stats = _graph_stats(sample.observation)
            for mode in modes:
                print(
                    f"  sample={sample.sample_id} step={sample.agent_step} "
                    f"device={device} mode={mode} "
                    f"candidates={stats.candidate_count} "
                    f"mean_nodes={stats.mean_nodes:.1f}",
                    flush=True,
                )
                prepared_observation = _make_observation_for_mode(
                    actor,
                    sample.observation,
                    mode,
                    device,
                )
                mean_ms, stdev_ms, min_ms = _time_actor_call(
                    actor,
                    prepared_observation,
                    deterministic=deterministic,
                    warmup=warmup,
                    repeats=repeats,
                    device=device,
                )
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "episode": sample.episode,
                        "agent_step": sample.agent_step,
                        "device": str(device),
                        "mode": mode,
                        "deterministic": deterministic,
                        "candidate_count": stats.candidate_count,
                        "total_candidate_nodes": stats.total_nodes,
                        "total_candidate_edges": stats.total_edges,
                        "mean_candidate_nodes": stats.mean_nodes,
                        "mean_candidate_edges": stats.mean_edges,
                        "max_candidate_nodes": stats.max_nodes,
                        "max_candidate_edges": stats.max_edges,
                        "n_remaining_cards": int(sample.observation["n_remaining_cards"]),
                        "mean_ms": mean_ms,
                        "stdev_ms": stdev_ms,
                        "min_ms": min_ms,
                        "repeats": repeats,
                        "warmup": warmup,
                    }
                )

    env.close()
    return rows


def _select_evenly_spaced_samples(
    samples: Sequence[ObservationSample],
    max_benchmark_samples: int | None,
) -> list[ObservationSample]:
    """Pick evenly spaced observations from a collected trajectory.

    Full episodes can contain enough expensive states that benchmarking every
    state is impractical. Even spacing keeps early, middle, and late board sizes
    represented without biasing toward the start of the episode.
    """

    if max_benchmark_samples is None or max_benchmark_samples >= len(samples):
        return list(samples)
    if max_benchmark_samples < 1:
        raise click.ClickException("--max-benchmark-samples must be at least 1.")
    if max_benchmark_samples == 1:
        return [samples[0]]

    indices = np.linspace(0, len(samples) - 1, num=max_benchmark_samples)
    selected_indices = sorted({int(round(index)) for index in indices})
    return [samples[index] for index in selected_indices]


def _write_csv(rows: Sequence[Mapping[str, Any]], output_csv: Path) -> None:
    """Write benchmark result rows with a stable header order."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    """Print a compact per-device/per-mode summary for terminal use."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["device"]), str(row["mode"])), []).append(row)

    print("Summary:")
    for (device, mode), group in sorted(grouped.items()):
        mean_ms = statistics.fmean(float(row["mean_ms"]) for row in group)
        largest = max(group, key=lambda row: int(row["mean_candidate_nodes"]))
        print(
            f"  {device:8s} {mode:10s} "
            f"mean={mean_ms:.3f} ms "
            f"largest_graph={float(largest['mean_candidate_nodes']):.1f} nodes/candidate "
            f"largest_time={float(largest['mean_ms']):.3f} ms"
        )


@click.command()
@click.option(
    "--config-path",
    type=click.Path(path_type=Path),
    default=Path("config/ppo_baseline.yaml"),
    show_default=True,
)
@click.option("--devices", default="cpu,cuda", show_default=True, help="Comma-separated torch devices.")
@click.option("--modes", default="raw,prebatched", show_default=True, help="Comma-separated: raw,prebatched.")
@click.option("--actor-dir", type=click.Path(path_type=Path), default=None, help="Directory containing actor.pt.")
@click.option("--seed", type=int, default=None, help="Overrides environment.seed from config.")
@click.option("--n-opponents", type=int, default=None, help="Overrides environment.n_opponents from config.")
@click.option("--episodes", type=int, default=1, show_default=True)
@click.option("--max-agent-steps", type=int, default=None)
@click.option("--max-samples", type=int, default=None)
@click.option(
    "--max-benchmark-samples",
    type=int,
    default=None,
    help="Benchmark this many evenly spaced collected observations.",
)
@click.option(
    "--action-selection",
    type=click.Choice(["first", "last", "random"]),
    default="first",
    show_default=True,
    help="Actions used only while collecting benchmark observations.",
)
@click.option("--warmup", type=int, default=5, show_default=True)
@click.option("--repeats", type=int, default=20, show_default=True)
@click.option("--deterministic/--stochastic", default=True, show_default=True)
@click.option("--output-csv", type=click.Path(path_type=Path), default=None)
def main(
    config_path: Path,
    devices: str,
    modes: str,
    actor_dir: Path | None,
    seed: int | None,
    n_opponents: int | None,
    episodes: int,
    max_agent_steps: int | None,
    max_samples: int | None,
    max_benchmark_samples: int | None,
    action_selection: str,
    warmup: int,
    repeats: int,
    deterministic: bool,
    output_csv: Path | None,
) -> None:
    """Collect observation samples, benchmark actor calls, and write a CSV."""

    if episodes < 1:
        raise click.ClickException("--episodes must be at least 1.")
    if repeats < 1:
        raise click.ClickException("--repeats must be at least 1.")
    if warmup < 0:
        raise click.ClickException("--warmup cannot be negative.")

    config = load_config(config_path)
    env_config = dict(config["environment"])
    if seed is not None:
        env_config["seed"] = seed
    if n_opponents is not None:
        env_config["n_opponents"] = n_opponents

    mode_names = _parse_csv_option(modes)
    if not mode_names:
        raise click.ClickException("--modes must contain at least one mode.")
    invalid_modes = sorted(set(mode_names) - {"raw", "prebatched"})
    if invalid_modes:
        raise click.ClickException(f"Unsupported mode(s): {', '.join(invalid_modes)}.")

    benchmark_devices = _available_devices(_parse_csv_option(devices))

    print(
        f"Collecting observations: seed={env_config['seed']} "
        f"n_opponents={env_config['n_opponents']} episodes={episodes}"
    )
    samples = _collect_samples(
        seed=env_config["seed"],
        n_opponents=env_config["n_opponents"],
        n_episodes=episodes,
        max_agent_steps=max_agent_steps,
        max_samples=max_samples,
        action_selection=action_selection,
    )
    if not samples:
        raise click.ClickException("No observations with legal action candidates were collected.")
    benchmark_samples = _select_evenly_spaced_samples(samples, max_benchmark_samples)
    print(
        f"Collected {len(samples)} observations; "
        f"benchmarking {len(benchmark_samples)} observations.",
        flush=True,
    )

    rows = _benchmark_samples(
        samples=benchmark_samples,
        algorithm_config=config["algorithm"],
        env_config=env_config,
        devices=benchmark_devices,
        modes=mode_names,
        actor_dir=actor_dir,
        warmup=warmup,
        repeats=repeats,
        deterministic=deterministic,
    )

    if output_csv is None:
        output_csv = Path("logs").joinpath(
            f"actor_inference_benchmark_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        )
    _write_csv(rows, output_csv)
    _print_summary(rows)
    print(f"Wrote rows: {output_csv}")


if __name__ == "__main__":
    main()
