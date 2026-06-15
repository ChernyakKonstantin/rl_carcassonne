"""Session adapter between the browser UI and the engine `GameEngine`.

The session owns UI-facing metadata and JSON serialization. It relies on
`GameEngine.current_turn` as the source of truth for the pending engine decision and
autoplays bot players until a manual player must act.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..card import Card, CardOption
from ..game import GameEngine
from ..player import ActionSelectionContext, PlayerState, RandomPlayer
from ..utils import Action, Orientation, PixelMeaning


@dataclass
class PlayerSetup:
    name: str
    kind: str
    onnx_path: Optional[str] = None


class GameSession:
    KIND_HUMAN = "human"
    KIND_RANDOM_BOT = "random_bot"
    KIND_ONNX_BOT = "onnx_bot"
    DEFAULT_PLAYERS = [
        {"name": "You", "kind": KIND_HUMAN},
        {"name": "Bot 1", "kind": KIND_RANDOM_BOT},
        {"name": "Bot 2", "kind": KIND_RANDOM_BOT},
    ]

    def __init__(
        self,
        seed: int = 67,
        players: Optional[List[Dict[str, Any]]] = None,
    ):
        self.seed = seed
        self.player_setups = self._normalize_player_setups(players)
        self.players: List[PlayerState] = []
        self.manual_player_ids = set()
        self.player_meta: Dict[int, Dict[str, Any]] = {}
        self._build_players()
        self.game = GameEngine(list(self.players), seed=seed)
        self.terminal = False
        self.message = ""
        self.game.reset()
        self._bind_player_metadata_after_game_id_assignment()
        self._advance_to_next_manual_turn()

    @classmethod
    def _normalize_player_setups(
        cls,
        players: Optional[List[Dict[str, Any]]],
    ) -> List[PlayerSetup]:
        if not players:
            players = cls.DEFAULT_PLAYERS
        if len(players) < 2:
            raise ValueError("At least two players are required.")
        if len(players) > 5:
            raise ValueError("At most five players are supported.")
        result = []
        for index, player in enumerate(players):
            name = str(player.get("name") or f"Player {index + 1}").strip() or f"Player {index + 1}"
            kind = str(player.get("kind") or cls.KIND_HUMAN)
            onnx_path = player.get("onnx_path")
            if kind not in {cls.KIND_HUMAN, cls.KIND_RANDOM_BOT, cls.KIND_ONNX_BOT}:
                raise ValueError(f"Unknown player kind: {kind}")
            result.append(PlayerSetup(name=name, kind=kind, onnx_path=onnx_path))
        return result

    def _build_players(self):
        for index, setup in enumerate(self.player_setups):
            if setup.kind == self.KIND_HUMAN:
                player = PlayerState()
            elif setup.kind == self.KIND_RANDOM_BOT:
                player = RandomPlayer(self.seed + 1000 + index)
            elif setup.kind == self.KIND_ONNX_BOT:
                raise ValueError("ONNX bot support is not implemented yet.")
            else:
                raise ValueError(f"Unknown player kind: {setup.kind}")
            self.players.append(player)

    def _bind_player_metadata_after_game_id_assignment(self):
        self.manual_player_ids = set()
        self.player_meta = {}
        for setup, player in zip(self.player_setups, self.players):
            is_manual = setup.kind == self.KIND_HUMAN
            if is_manual:
                self.manual_player_ids.add(player.id)
            self.player_meta[player.id] = {
                "name": setup.name,
                "kind": setup.kind,
                "manual": is_manual,
                "onnx_path": setup.onnx_path,
            }
        if len(self.manual_player_ids) == 0:
            raise ValueError("At least one human player is required.")

    def _advance_to_next_manual_turn(self):
        while True:
            turn = self.game.advance_to_next_turn()
            if turn is None:
                break
            if turn.player.id in self.manual_player_ids:
                break
            select_action = getattr(turn.player, "select_action", None)
            if not callable(select_action):
                raise RuntimeError(
                    f"Player {turn.player.id} cannot be autoplayed; use a manual player or a bot with select_action(context)."
                )
            action = select_action(
                ActionSelectionContext(
                    board=self.game.board,
                    player=turn.player,
                    card=turn.card,
                    actions=turn.actions,
                )
            )
            self.game.apply_turn_action(action)
        self.terminal = self.game.terminal
        self.message = self._message_for_state()

    def _message_for_state(self) -> str:
        if self.terminal:
            return "GameEngine over."
        turn = self.game.current_turn
        if turn is None:
            return ""
        player_name = self.player_meta[turn.player.id]["name"]
        if player_name == "You":
            return "Your turn."
        return f"{player_name}'s turn."

    def apply_action(self, action_index: int):
        if self.terminal:
            raise ValueError("GameEngine is already over.")
        turn = self.game.current_turn
        if turn is None or turn.player.id not in self.manual_player_ids:
            raise RuntimeError("No pending manual turn.")
        self.game.apply_turn_action_by_index(action_index)
        self._advance_to_next_manual_turn()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "terminal": self.terminal,
            "message": self.message,
            "seed": self.seed,
            "deck_remaining": len(self.game.deck),
            "players": self._serialize_players(),
            "manual_player_ids": [str(player_id) for player_id in sorted(self.manual_player_ids)],
            "board": {
                "tiles": [self._serialize_tile(tile) for tile in self.game.board.get_tiles_snapshot()],
            },
            "current_turn": self._serialize_turn(),
        }

    def _serialize_players(self) -> List[Dict[str, Any]]:
        result = []
        for idx, player in enumerate(self.players):
            meta = self.player_meta[player.id]
            result.append(
                {
                    "id": str(player.id),
                    "label": meta["name"],
                    "kind": meta["kind"],
                    "manual": meta["manual"],
                    "score": player.scores,
                    "remaining_meeples": player.remaining_meeples,
                    "human": meta["manual"],
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
        turn = self.game.current_turn
        if turn is None:
            return None
        return {
            "player": self._serialize_turn_player(turn.player),
            "card": self._serialize_card(turn.card),
            "actions": [self._serialize_action(index, action) for index, action in enumerate(turn.actions)],
        }

    def _serialize_turn_player(self, player: PlayerState) -> Dict[str, Any]:
        meta = self.player_meta[player.id]
        return {
            "id": str(player.id),
            "label": meta["name"],
            "kind": meta["kind"],
            "manual": meta["manual"],
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


HumanGameSession = GameSession
