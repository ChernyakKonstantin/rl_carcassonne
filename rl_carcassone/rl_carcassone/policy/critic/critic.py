from gymnasium import spaces

from rl_carcassone.policy.features_extractor import CarcassonneGraphFeatureExtractor

from .base_critic import BaseCritic


class Critic(BaseCritic):
    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.features_extractor = CarcassonneGraphFeatureExtractor(
            observation_space=observation_space,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            pool_candidates=True,
        )
        self.complete_setup(self.features_extractor.out_dim)
