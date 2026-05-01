import random
from types import NoneType
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .board import Action, Board
from .deck import Card, Deck
from .player import BasePlayer
from .utils import PixelMeaning


# TODO: Add calculations for fields on the game end!
class Game:
    PLAYER_ID: int = 0  # NOTE: Main player (i.e. agent or user).

    def __init__(self, players: List[BasePlayer], seed: int = 42, enable_render: bool = False):
        if len(players) < 2:
            raise ValueError("At least 2 players are required.")
        if len(players) > 5:
            raise ValueError("Up to 5 players supported.")
        self.players = players
        self.id2player = {player.id: player for player in self.players}
        n_players = len(self.players)
        if n_players < 0:
            raise ValueError("Minimum number of players is 2")
        elif n_players > 5:
            raise ValueError("Maximum number of players is 5")
        self.board = Board()
        self.deck = Deck(seed=seed)
        self.rng = random.Random()
        self.enable_render = enable_render

    def reset(self):
        """Clear board, put initial card, randomize players order."""
        self.deck.reset()
        first_card = self.deck.get_card()
        self.board.reset(first_card)
        # NOTE: Randomly select the order of players.
        self.rng.shuffle(self.players)
        if self.enable_render:
            self.render()

    def get_board_view(self) -> np.ndarray:
        return self.board.get_view()

    def render(self):
        board_view = self.get_board_view()
        image = np.zeros((*board_view.shape, 3), dtype=np.uint8)
        image[board_view == PixelMeaning.ABBOT] = (255, 0, 0)
        image[board_view == PixelMeaning.FIELD] = (0, 255, 0)
        image[board_view == PixelMeaning.CITY] = (0, 0, 255)
        image[board_view == PixelMeaning.ROAD] = (146, 146, 146)
        image[board_view == 5] = (0, 190, 190)

        fig = plt.figure(dpi=200)
        ax = fig.add_subplot()
        ax.imshow(image, extent=[0, image.shape[1], 0, image.shape[0]])
        ax.set_xticks(np.arange(0, image.shape[1], 5), minor=False)
        ax.set_yticks(np.arange(0, image.shape[0], 5), minor=False)
        ax.grid(which="major", snap=True, lw=0.3, ls="-", color="grey")
        plt.show()

    def close(self):
        if self.enable_render:
            plt.close("all")

    def _get_current_player(self) -> BasePlayer:
        current_player = self.players.pop(0)
        self.players.append(current_player)
        return current_player

    def _get_card_and_possible_actions(self) -> Tuple[bool, Union[Card, NoneType], List[Action]]:
        card_is_found = False
        cards_to_put_back = []
        while len(self.deck) > 0:
            card = self.deck.get_card()
            possible_actions = self.board.get_possible_actions(card)
            if len(possible_actions) > 0:
                card_is_found = True
                break
            else:
                cards_to_put_back.append(card)
        for card_to_put_back in cards_to_put_back:
            self.deck.put_card_back(card_to_put_back)
        if not card_is_found:
            card = None
            possible_actions = []
        return card_is_found, card, possible_actions

    def _loop_step(self) -> bool:
        current_player = self._get_current_player()
        card_is_found, card, possible_actions = self._get_card_and_possible_actions()
        if not card_is_found:
            return False
        action = current_player.select_action(card, self.board.get_view(), possible_actions)
        if action.meeple_position is not None:
            current_player.remaining_meeples -= 1
        self.board.put_card_and_meeple(card, action, current_player.id)
        step_results = self.board.get_outcomes(complete_property_only=True, consider_fields=False)
        for player_id, result in step_results.items():
            self.id2player[player_id].scores += result.score
            self.id2player[player_id].return_n_meeples(result.returned_meeples)
        return True

    def _process_outcomes(self):
        final_results = self.board.get_outcomes(complete_property_only=False, consider_fields=True)
        for player_id, result in final_results.items():
            self.id2player[player_id].scores += result.score
            self.id2player[player_id].return_n_meeples(result.returned_meeples)

    def mainloop(self):
        """
        Play until there are no cards left.
        """
        while True:
            resume = self._loop_step()
            if not resume:
                break
            else:
                if self.enable_render:
                    self.render()
        self._process_outcomes()
        if self.enable_render:
            self.render()
