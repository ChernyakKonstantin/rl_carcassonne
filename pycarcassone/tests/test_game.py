from pycarcassone.pycarcassone.game import Game
from pycarcassone.pycarcassone.player import RandomPlayer


def _deck_prefix(game: Game):
    return [card.type for card in game.deck.remaining_cards[:5]]


def test_game_assigns_dense_player_ids_and_resets_player_state():
    players = [RandomPlayer(10), RandomPlayer(20), RandomPlayer(30)]
    game = Game(players, seed=123)

    assert [player.id for player in players] == [0, 1, 2]
    assert players[0].id == Game.PLAYER_ID

    game.reset()
    first_order = [player.id for player in game.players]
    first_deck_prefix = _deck_prefix(game)

    players[0].scores = 11
    players[0].remaining_meeples = 3
    game._get_current_player()

    game.reset()

    assert [player.id for player in game.players] == first_order
    assert _deck_prefix(game) == first_deck_prefix
    assert all(player.scores == 0 for player in players)
    assert all(player.remaining_meeples == 7 for player in players)


def test_game_reset_can_change_seed():
    players = [RandomPlayer(10), RandomPlayer(20)]
    game = Game(players, seed=123)

    game.reset()
    first_deck_prefix = _deck_prefix(game)
    game.reset(seed=124)
    second_deck_prefix = _deck_prefix(game)

    assert second_deck_prefix != first_deck_prefix
