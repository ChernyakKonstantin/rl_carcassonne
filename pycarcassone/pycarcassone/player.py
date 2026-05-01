import random
from typing import List
from uuid import uuid4

from .card import Card
from .utils import Action


class BasePlayer:
    def __init__(self):
        self.id = uuid4().int
        self._max_meeples = 7
        self.remaining_meeples = 7
        self.scores = 0

    def select_action(self, current_board_state, current_card: Card, possible_actions: List[Action]) -> Action:
        raise NotImplementedError

    def return_n_meeples(self, n_meeples: int):
        self.remaining_meeples += n_meeples
        if self.remaining_meeples > self._max_meeples:
            raise ValueError


class RandomPlayer(BasePlayer):
    def __init__(self, seed: int):
        super().__init__()
        self.rng = random.Random(seed)

    def select_action(self, current_board_state, current_card: Card, possible_actions: List[Action]) -> Action:
        if self.remaining_meeples > 0:
            return self.rng.choice(possible_actions)
        else:
            return self.rng.choice(list(filter(lambda a: a.meeple_position is None, possible_actions)))
