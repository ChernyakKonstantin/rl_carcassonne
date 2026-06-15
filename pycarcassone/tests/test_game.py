import pytest

from pycarcassone.game import GameEngine
from pycarcassone.player import PlayerState, RandomPlayer


class FirstActionPlayer(PlayerState):
    def __init__(self):
        super().__init__()
        self.last_context = None

    def select_action(self, context):
        self.last_context = context
        return context.actions[0]


def _deck_prefix(game: GameEngine):
    return [card.type for card in game.deck.remaining_cards[:5]]


def test_game_assigns_dense_player_ids_and_resets_player_state():
    players = [RandomPlayer(10), RandomPlayer(20), RandomPlayer(30)]
    game = GameEngine(players, seed=123)

    assert [player.id for player in players] == [0, 1, 2]

    game.reset()
    first_order = [player.id for player in game.players]
    first_deck_prefix = _deck_prefix(game)

    players[0].scores = 11
    players[0].remaining_meeples = 3
    game.advance_to_next_turn()

    game.reset()

    assert [player.id for player in game.players] == first_order
    assert _deck_prefix(game) == first_deck_prefix
    assert all(player.scores == 0 for player in players)
    assert all(player.remaining_meeples == 7 for player in players)


def test_game_accepts_five_players():
    players = [RandomPlayer(seed) for seed in range(5)]
    game = GameEngine(players, seed=123)

    assert [player.id for player in players] == [0, 1, 2, 3, 4]
    game.reset()
    assert len(game.players) == 5


def test_game_reset_can_change_seed():
    players = [RandomPlayer(10), RandomPlayer(20)]
    game = GameEngine(players, seed=123)

    game.reset()
    first_deck_prefix = _deck_prefix(game)
    game.reset(seed=124)
    second_deck_prefix = _deck_prefix(game)

    assert second_deck_prefix != first_deck_prefix


def test_game_exposes_pending_turn_until_action_is_applied():
    players = [RandomPlayer(10), RandomPlayer(20)]
    game = GameEngine(players, seed=123)
    game.reset()

    turn = game.advance_to_next_turn()

    assert turn is not None
    assert turn.player in players
    assert len(turn.actions) > 0
    assert game.advance_to_next_turn() is turn

    game.apply_turn_action_by_index(0)

    assert game.current_turn is None
    assert not game.terminal


def test_game_state_snapshot_exposes_adapter_state_without_legacy_board_view():
    players = [RandomPlayer(10), RandomPlayer(20)]
    game = GameEngine(players, seed=123)
    game.reset()

    initial_snapshot = game.get_state_snapshot()

    assert not hasattr(game, "get_board_view")
    assert not hasattr(game.board, "get_view")
    assert not initial_snapshot.terminal
    assert initial_snapshot.seed == 123
    assert initial_snapshot.deck_remaining == len(game.deck)
    assert [player.id for player in initial_snapshot.players] == [0, 1]
    assert len(initial_snapshot.player_order) == 2
    assert initial_snapshot.current_turn is None
    assert len(initial_snapshot.tiles_snapshot) == 1
    assert len(initial_snapshot.graph_snapshot["nodes"]) > 0
    assert len(initial_snapshot.graph_snapshot["edges"]) > 0

    turn = game.advance_to_next_turn()
    turn_snapshot = game.get_state_snapshot()

    assert turn_snapshot.current_turn is not None
    assert turn_snapshot.current_turn.player_id == turn.player.id
    assert turn_snapshot.current_turn.card is turn.card
    assert turn_snapshot.current_turn.actions == tuple(turn.actions)


def test_game_advance_until_player_turn_autoplays_other_players():
    players = [RandomPlayer(10), RandomPlayer(20), RandomPlayer(30)]
    game = GameEngine(players, seed=67)
    game.reset()
    target_player_id = players[0].id

    turn = game.advance_until_player_turn(target_player_id, autoplay=True)

    assert turn is not None
    assert turn.player.id == target_player_id
    assert len(turn.actions) > 0


def test_game_rejects_autoplay_for_player_without_action_selector():
    target = RandomPlayer(10)
    manual = PlayerState()
    game = GameEngine([target, manual], seed=67)
    game.reset()
    game.players = [manual, target]

    with pytest.raises(RuntimeError, match="cannot be autoplayed"):
        game.advance_until_player_turn(target.id, autoplay=True)


def test_game_autoplay_passes_board_context_without_forcing_view():
    autoplayed = FirstActionPlayer()
    target = RandomPlayer(10)
    game = GameEngine([target, autoplayed], seed=67)
    game.reset()
    game.players = [autoplayed, target]

    turn = game.advance_until_player_turn(target.id, autoplay=True)

    assert turn.player is target
    assert autoplayed.last_context is not None
    assert autoplayed.last_context.board is game.board
    assert autoplayed.last_context.player is autoplayed
    assert autoplayed.last_context.card is not None
    assert len(autoplayed.last_context.actions) > 0
