import json
import random
from copy import deepcopy
from pathlib import Path
from typing import List

from .card import Card

PATH_TO_CARDS = Path(__file__).parent.joinpath("assets").joinpath("cards.json")


class Deck:
    def __init__(self, seed: int = 42):
        self._cards = self._load()
        self.rng = random.Random(seed)
        self.remaining_cards = None  # NOTE: Set on reset

    def _load(self) -> List[Card]:
        with open(PATH_TO_CARDS, "r") as f:
            cards_data = json.load(f)
        cards = []
        for entry in cards_data:
            count = entry.pop("count")
            cards.extend([Card(**entry)] * count)
        return cards

    def __len__(self) -> int:
        if self.remaining_cards is None:
            raise ValueError("Deck was not reseted!")
        else:
            return len(self.remaining_cards)

    def reset(self):
        self.remaining_cards = deepcopy(self._cards)
        self.rng.shuffle(self.remaining_cards)

    def get_card(self) -> Card:
        return self.remaining_cards.pop(0)

    def put_card_back(self, card: Card):
        """Put card back to the end of a deck."""
        self.remaining_cards.append(card)
