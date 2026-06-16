from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(config_path: Union[Path, str]) -> Dict[str, Any]:
    if not isinstance(config_path, Path):
        config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config is empty: {config_path}")
    return config


def save_config(config: Dict[str, Any], save_dir: Union[Path, str]) -> None:
    config = deepcopy(config)
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir.joinpath("config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
