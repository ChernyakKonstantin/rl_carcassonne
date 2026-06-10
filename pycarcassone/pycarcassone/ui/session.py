from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..card import Card, CardOption
from ..game import Game
from ..player import BasePlayer, RandomPlayer
from ..utils import Action, Orientation, PixelMeaning


class HumanPlayer(BasePlayer):
    def select_action(self, current_board_state, current_card: Card, possible_actions: List[Action]) -> Action:
        raise RuntimeError("Human actions must be selected through HumanGameSession.apply_action")


@dataclass
class PendingTurn:
    player: BasePlayer
    card: Card
    actions: List[Action]


class HumanGameSession:
    def __init__(self, seed: int = 67, n_opponents: int = 2):
        if n_opponents < 1:
            raise ValueError("At least one opponent is required.")
        if n_opponents > 4:
            raise ValueError("At most four opponents are supported.")
        self.seed = seed
        self.n_opponents = n_opponents
        self.human = HumanPlayer()
        self.players: List[BasePlayer] = [self.human]
        self.players.extend(RandomPlayer(seed + 1000 + i) for i in range(n_opponents))
        self.game = Game(list(self.players), seed=seed, enable_render=False)
        self.pending_turn: Optional[PendingTurn] = None
        self.terminal = False
        self.message = ""
        self.game.reset()
        self._advance_to_human_turn()

    def _advance_to_human_turn(self):
        self.pending_turn = None
        while True:
            current_player = self.game._get_current_player()
            card_is_found, card, possible_actions = self.game._get_card_and_possible_actions()
            if not card_is_found:
                self.game._process_outcomes()
                self.terminal = True
                self.message = "Game over."
                return

            possible_actions = self.game.get_player_possible_actions(current_player, possible_actions)
            if current_player.id == self.human.id:
                self.pending_turn = PendingTurn(current_player, card, possible_actions)
                self.message = "Your turn."
                return

            action = current_player.select_action(self.game.board.get_view(), card, possible_actions)
            self.game.apply_player_action(current_player, card, action)

    def apply_action(self, action_index: int):
        if self.terminal:
            raise ValueError("Game is already over.")
        if self.pending_turn is None:
            raise RuntimeError("No pending human turn.")
        if action_index < 0 or action_index >= len(self.pending_turn.actions):
            raise ValueError(f"Unknown action index: {action_index}")
        action = self.pending_turn.actions[action_index]
        self.game.apply_player_action(self.pending_turn.player, self.pending_turn.card, action)
        self._advance_to_human_turn()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "terminal": self.terminal,
            "message": self.message,
            "seed": self.seed,
            "deck_remaining": len(self.game.deck),
            "players": self._serialize_players(),
            "human_player_id": str(self.human.id),
            "board": {
                "tiles": [self._serialize_tile(tile) for tile in self.game.board.get_tiles_snapshot()],
            },
            "current_turn": self._serialize_turn(),
        }

    def _serialize_players(self) -> List[Dict[str, Any]]:
        result = []
        for idx, player in enumerate(self.players):
            result.append(
                {
                    "id": str(player.id),
                    "label": "You" if player.id == self.human.id else f"P{idx + 1}",
                    "score": player.scores,
                    "remaining_meeples": player.remaining_meeples,
                    "human": player.id == self.human.id,
                    "color": self._player_color(idx),
                }
            )
        return result

    @staticmethod
    def _player_color(player_index: int) -> str:
        colors = ["#d43f3a", "#2364aa", "#279f6f", "#f2a900", "#7b4ab8"]
        return colors[player_index % len(colors)]

    @staticmethod
    def _serialize_tile(tile: Dict[str, Any]) -> Dict[str, Any]:
        y, x = tile["position"]
        property_data = []
        for property_index, data in sorted(tile["property_data"].items()):
            owner = data["owner"]
            property_data.append(
                {
                    "index": property_index,
                    "type": data["type"],
                    "type_value": data["type_value"],
                    "owner": str(owner) if owner is not None else None,
                    "owners": [str(owner) for owner in data["owners"]],
                    "ignored": data["ignored"],
                    "shield": data["shield"],
                }
            )
        return {
            "position": {"y": y, "x": x},
            "values": tile["values"],
            "properties": tile["properties"],
            "property_data": property_data,
        }

    def _serialize_turn(self) -> Optional[Dict[str, Any]]:
        if self.pending_turn is None:
            return None
        return {
            "card": self._serialize_card(self.pending_turn.card),
            "actions": [
                self._serialize_action(index, action) for index, action in enumerate(self.pending_turn.actions)
            ],
        }

    @classmethod
    def _serialize_card(cls, card: Card) -> Dict[str, Any]:
        options = []
        for orientation, option in sorted(card.options.items(), key=lambda item: item[0]):
            options.append(cls._serialize_card_option(orientation, option))
        return {
            "type": card.type,
            "shield": card.shield,
            "options": options,
        }

    @staticmethod
    def _serialize_card_option(orientation: Orientation, option: CardOption) -> Dict[str, Any]:
        return {
            "orientation": int(orientation),
            "angle": int(orientation) * 90,
            "values": option.values,
            "properties": option.properties,
            "property_types": [
                {
                    "index": index,
                    "type": property_meta.type.name,
                    "type_value": int(property_meta.type),
                    "shield": option.shield and property_meta.type == PixelMeaning.CITY,
                }
                for index, property_meta in enumerate(option.properties_metas)
            ],
        }

    @staticmethod
    def _serialize_action(index: int, action: Action) -> Dict[str, Any]:
        y, x = action.position
        return {
            "index": index,
            "position": {"y": y, "x": x},
            "orientation": int(action.orientation),
            "angle": int(action.orientation) * 90,
            "meeple_position": action.meeple_position,
        }
