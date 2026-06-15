import random
from dataclasses import dataclass
from typing import Any, Sequence

from .card import Card
from .utils import Action


class PlayerState:
    """Mutable player state shared by manual players and autonomous policies."""

    def __init__(self):
        self.id = None
        self._max_meeples = 7
        self.remaining_meeples = self._max_meeples
        self.scores = 0

    def reset(self):
        self.remaining_meeples = self._max_meeples
        self.scores = 0

    def return_n_meeples(self, n_meeples: int):
        self.remaining_meeples += n_meeples
        if self.remaining_meeples > self._max_meeples:
            raise ValueError


@dataclass(frozen=True)
class ActionSelectionContext:
    """Read-only action-selection input for autonomous players."""

    board: Any
    player: PlayerState
    card: Card
    actions: Sequence[Action]


class RandomPlayer(PlayerState):
    """Autonomous player that chooses uniformly from legal actions."""

    def __init__(self, seed: int):
        super().__init__()
        self.rng = random.Random(seed)

    def select_action(self, context: ActionSelectionContext) -> Action:
        return self.rng.choice(list(context.actions))
