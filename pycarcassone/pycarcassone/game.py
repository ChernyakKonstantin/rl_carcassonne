import random
from dataclasses import dataclass
from types import NoneType
from typing import Any, Dict, List, Optional, Tuple, Union

from .board import Action, Board
from .deck import Card, Deck
from .player import ActionSelectionContext, PlayerState


@dataclass
class GameTurn:
    player: PlayerState
    card: Card
    actions: List[Action]


@dataclass(frozen=True)
class GamePlayerSnapshot:
    id: int
    scores: int
    remaining_meeples: int


@dataclass(frozen=True)
class GameTurnSnapshot:
    player_id: int
    card: Card
    actions: Tuple[Action, ...]


@dataclass(frozen=True)
class GameStateSnapshot:
    terminal: bool
    seed: int
    deck_remaining: int
    players: Tuple[GamePlayerSnapshot, ...]
    player_order: Tuple[int, ...]
    current_turn: Optional[GameTurnSnapshot]
    tiles_snapshot: List[Dict[str, Any]]
    graph_snapshot: Dict[str, List]


class GameEngine:
    """
    Stateful turn controller for engine adapters.

    This class does not run a complete playable application by itself. Use an
    external adapter, such as the browser UI or the Gymnasium environment, to
    request turns, choose actions, and apply them through the turn-service API.

    The class owns the game state: board, deck, player order, player resources,
    scores, current pending turn, and terminal state. It can:

    - reset the whole game deterministically from a seed;
    - draw the next placeable card and produce legal actions for the current
      player;
    - filter legal actions by current-player resources, for example no meeples
      left;
    - apply a chosen action, update the board, score completed properties, and
      return meeples;
    - autoplay non-target players that implement `select_action(context)`.

    Typical adapter flow:

    1. Call `reset()`.
    2. Call `advance_to_next_turn()` or `advance_until_player_turn(...)`.
    3. Show/encode the returned `GameTurn.card` and `GameTurn.actions`.
    4. Choose one of those actions externally.
    5. Call `apply_turn_action(...)` or `apply_turn_action_by_index(...)`.
    6. Repeat until `terminal` is true or the advance method returns `None`.

    Use `advance_to_next_turn()` when the adapter wants to control every player
    decision itself. Use `advance_until_player_turn(player_id, autoplay=True)`
    when the adapter owns one target player and wants autonomous players to move
    automatically through `select_action(context)`.
    """

    MAX_PLAYERS = 5

    def __init__(self, players: List[PlayerState], seed: int = 42):
        if len(players) < 2:
            raise ValueError("At least 2 players are required.")
        if len(players) > self.MAX_PLAYERS:
            raise ValueError(f"Up to {self.MAX_PLAYERS} players supported.")
        self.seed = seed
        self._initial_players_order = list(players)
        self._assign_player_ids()
        self.players = list(self._initial_players_order)
        self.id2player = {player.id: player for player in self._initial_players_order}
        n_players = len(self.players)
        if n_players < 2:
            raise ValueError("Minimum number of players is 2")
        elif n_players > self.MAX_PLAYERS:
            raise ValueError(f"Maximum number of players is {self.MAX_PLAYERS}")
        self.board = Board()
        self.deck = Deck(seed=seed)
        self.rng = random.Random(seed)
        self.current_turn: Optional[GameTurn] = None
        self.terminal = False

    def _assign_player_ids(self):
        for player_id, player in enumerate(self._initial_players_order):
            player.id = player_id

    def reset(self, seed: int = None):
        """
        Start a fresh game.

        Resets board, deck, scores, meeples, pending turn, terminal flag, player
        order, and RNG state. Passing `seed` replaces the game seed before the
        reset, so repeated resets with the same seed are deterministic.
        """
        if seed is not None:
            self.seed = seed
        self.rng.seed(self.seed)
        self.current_turn = None
        self.terminal = False
        self.players = list(self._initial_players_order)
        for player in self._initial_players_order:
            player.reset()
        self.deck.reset(seed=self.seed)
        first_card = self.deck.get_card()
        self.board.reset(first_card)
        # NOTE: Randomly select the order of players.
        self.rng.shuffle(self.players)

    def get_state_snapshot(self) -> GameStateSnapshot:
        """
        Return a read-oriented snapshot of the current game state.

        This is the main engine-level state API for adapters. It collects the
        game controller state (terminal flag, deck size, players, turn order,
        and pending turn) plus board exports for UI/RL serialization. Expensive
        action-candidate graph previews are intentionally not included; build
        them from `snapshot.current_turn` and `board` only when an adapter needs
        them.
        """
        current_turn = None
        if self.current_turn is not None:
            current_turn = GameTurnSnapshot(
                player_id=self.current_turn.player.id,
                card=self.current_turn.card,
                actions=tuple(self.current_turn.actions),
            )
        return GameStateSnapshot(
            terminal=self.terminal,
            seed=self.seed,
            deck_remaining=len(self.deck),
            players=tuple(
                GamePlayerSnapshot(
                    id=player.id,
                    scores=player.scores,
                    remaining_meeples=player.remaining_meeples,
                )
                for player in sorted(self.id2player.values(), key=lambda player: player.id)
            ),
            player_order=tuple(player.id for player in self.players),
            current_turn=current_turn,
            tiles_snapshot=self.board.get_tiles_snapshot(),
            graph_snapshot=self.board.get_graph_snapshot(),
        )

    def _get_current_player(self) -> PlayerState:
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

    @staticmethod
    def get_player_possible_actions(player: PlayerState, possible_actions: List[Action]) -> List[Action]:
        if player.remaining_meeples > 0:
            return possible_actions
        return [action for action in possible_actions if action.meeple_position is None]

    def apply_player_action(self, current_player: PlayerState, card: Card, action: Action):
        if action.meeple_position is not None:
            current_player.remaining_meeples -= 1
        self.board.put_card_and_meeple(card, action, current_player.id)
        step_results = self.board.resolve_outcomes(complete_property_only=True, consider_fields=False)
        for player_id, result in step_results.items():
            self.id2player[player_id].scores += result.score
            self.id2player[player_id].return_n_meeples(result.returned_meeples)

    def advance_to_next_turn(self) -> Optional[GameTurn]:
        """
        Return the next unresolved player decision without choosing an action.

        If a turn is already pending, returns the same `GameTurn` again. This
        makes the method safe for polling adapters such as web sessions.

        If no turn is pending, rotates to the next player, draws the next card
        that has at least one legal placement, filters actions by that player's
        meeple resources, stores the result in `current_turn`, and returns it.

        Returns `None` only when the game is terminal or no placeable card
        remains. In that end-of-game case final scoring is applied and
        `terminal` becomes true.

        Use this method when your adapter wants to observe and decide for every
        player explicitly.
        """
        if self.terminal:
            return None
        if self.current_turn is not None:
            return self.current_turn
        current_player = self._get_current_player()
        card_is_found, card, possible_actions = self._get_card_and_possible_actions()
        if not card_is_found:
            self._process_outcomes()
            self.terminal = True
            return None
        possible_actions = self.get_player_possible_actions(current_player, possible_actions)
        self.current_turn = GameTurn(current_player, card, possible_actions)
        return self.current_turn

    def apply_turn_action(self, action: Action):
        """
        Apply one legal action for the currently pending turn.

        The action must be one of `current_turn.actions`; this method clears the
        pending turn, places the card and optional meeple, applies step scoring,
        and returns completed-property meeples to their owners.
        """
        if self.terminal:
            raise RuntimeError("GameEngine is already over.")
        if self.current_turn is None:
            raise RuntimeError("No pending turn. Call advance_to_next_turn first.")
        if action not in self.current_turn.actions:
            raise ValueError("Action is not legal for the current turn.")
        turn = self.current_turn
        self.current_turn = None
        self.apply_player_action(turn.player, turn.card, action)

    def apply_turn_action_by_index(self, action_index: int):
        """Apply `current_turn.actions[action_index]` for adapter-facing action ids."""
        if self.current_turn is None:
            raise RuntimeError("No pending turn. Call advance_to_next_turn first.")
        if action_index < 0 or action_index >= len(self.current_turn.actions):
            raise ValueError(f"Unknown action index: {action_index}")
        self.apply_turn_action(self.current_turn.actions[action_index])

    def advance_until_player_turn(self, player_id: int, autoplay: bool = True) -> Optional[GameTurn]:
        """
        Advance until a target player must act.

        This is the convenience method for single-owner adapters: RL envs where
        player `0` is the trainable agent, or a UI session waiting for the next
        manual player. It repeatedly calls `advance_to_next_turn()`.

        If the next turn belongs to `player_id`, returns it without applying any
        action. If another player is next and `autoplay` is true, that player
        must have a callable `select_action(context)` method. The context
        contains the board object, current player, card, and legal actions, so
        the player can build its own graph/image/JSON observation. The selected
        action is applied, then advancement continues. If `autoplay` is false,
        returns the other player's turn immediately.

        Returns `None` when the game reaches terminal state before `player_id`
        gets another turn.
        """
        while True:
            turn = self.advance_to_next_turn()
            if turn is None:
                return None
            if turn.player.id == player_id:
                return turn
            if not autoplay:
                return turn
            select_action = getattr(turn.player, "select_action", None)
            if not callable(select_action):
                raise RuntimeError(
                    f"Player {turn.player.id} cannot be autoplayed; handle this turn explicitly "
                    "or use a player object with select_action(context)."
                )
            action = select_action(
                ActionSelectionContext(
                    board=self.board,
                    player=turn.player,
                    card=turn.card,
                    actions=turn.actions,
                )
            )
            self.apply_turn_action(action)

    def _process_outcomes(self):
        final_results = self.board.resolve_outcomes(complete_property_only=False, consider_fields=True)
        for player_id, result in final_results.items():
            self.id2player[player_id].scores += result.score
            self.id2player[player_id].return_n_meeples(result.returned_meeples)
