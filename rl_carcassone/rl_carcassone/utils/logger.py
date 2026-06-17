import csv
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union


class ExperimentLogger:
    SKIPPED_VAL_ROLLOUT_KEYS = {"legal_action_count_mean"}

    def __init__(self, directory: Union[str, Path]):
        if isinstance(directory, str):
            directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.progress_fname = directory.joinpath("progress.csv")
        self.per_epoch_fname = directory.joinpath("per_epoch_critic_data.jsonl")
        self.records: List[Dict[str, Any]] = []

    def log(
        self,
        iteration: int,
        val_rollout_statistics: Dict[str, Any],
        train_rollout_statistics: Dict[str, Any],
        train_statistics: Dict[str, Any],
        passed_episodes: int,
        elapsed_time: float,
    ) -> None:
        train_statistics = dict(train_statistics)
        per_epoch_critic_data = train_statistics.pop("per_epoch_critic_data")
        self._log_per_epoch(
            {
                "iteration": iteration,
                "passed_episodes": passed_episodes,
                "per_epoch_critic_data": per_epoch_critic_data,
            }
        )

        record = {"iteration": iteration}
        for k, v in val_rollout_statistics.items():
            if k in self.SKIPPED_VAL_ROLLOUT_KEYS:
                continue
            record[f"val_rollout/{k}"] = v
        for k, v in train_rollout_statistics.items():
            record[f"train_rollout/{k}"] = v
        for k, v in train_statistics.items():
            # NOTE: This is cumulative column.
            if k == "critic_n_updates" and self.records:
                v += self.records[-1]["train/critic_n_updates"]
            elif k == "actor_n_updates" and self.records:
                v += self.records[-1]["train/actor_n_updates"]
            record[f"train/{k}"] = v
        record["passed_episodes"] = passed_episodes
        record["elapsed_time"] = elapsed_time
        self.records.append(self._jsonable(record))
        self._write_progress()

    def _write_progress(self) -> None:
        fieldnames = []
        for record in self.records:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(self.progress_fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    @staticmethod
    def _jsonable(record: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in record.items():
            if hasattr(value, "item"):
                result[key] = value.item()
            else:
                result[key] = value
        return result

    def _log_per_epoch(self, per_epoch_critic_data: Dict[str, Union[int, Dict[str, List[float]]]]) -> None:
        json_str = json.dumps(per_epoch_critic_data) + "\n"
        with open(self.per_epoch_fname, "a") as f:
            f.write(json_str)


class EventLogger:
    """Append timestamped trainer events to a plain text log file."""

    def __init__(self, directory: Union[str, Path], filename: str = "events.log") -> None:
        if isinstance(directory, str):
            directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.fname = directory.joinpath(filename)

    def log(self, event: str, **fields: Any) -> None:
        line = f"{self._timestamp()} | {event}"
        if fields:
            details = " | ".join(f"{key}={value}" for key, value in fields.items())
            line = f"{line} | {details}"
        with open(self.fname, "a") as f:
            f.write(f"{line}\n")

    def log_exception(self, exc: BaseException) -> None:
        self.log("error", message=str(exc))
        formatted_traceback = traceback.format_exc()
        with open(self.fname, "a") as f:
            f.write(formatted_traceback)
            if not formatted_traceback.endswith("\n"):
                f.write("\n")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
