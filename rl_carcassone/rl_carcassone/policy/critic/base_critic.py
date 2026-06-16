from pathlib import Path
from typing import Any, Mapping, Union

import torch
import torch as th
import torch.nn as nn


class BaseCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def complete_setup(self, latent_dim_vf: int) -> None:
        self.value_net = nn.Linear(latent_dim_vf, 1)

    def forward(self, observation: Mapping[str, Any]) -> th.Tensor:
        latent_vf = self.features_extractor(observation)
        return self.value_net(latent_vf).squeeze(-1)

    def load(self, load_dir: Path) -> None:
        self.load_state_dict(
            torch.load(load_dir.joinpath("critic.pt"), map_location=self.device, weights_only=True),
            strict=True,
        )

    def save(self, save_dir: Union[str, Path]) -> None:
        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)
        torch.save(self.state_dict(), save_dir.joinpath("critic.pt"))

    def apply_input_preprocessor(self, x: Mapping[str, Any]) -> Mapping[str, Any]:
        return x
