from pycarcassone.board import Board
from pycarcassone.deck import Deck
from pycarcassone.utils import Action, Orientation


def _cards_by_type():
    return {card.type: card for card in Deck()._load()}


def _city_ignored_values(board: Board):
    return [
        property_data["ignored"]
        for tile in board.get_tiles_snapshot()
        for property_data in tile["property_data"].values()
        if property_data["type"] == "CITY"
    ]


def test_resolve_outcomes_marks_scored_properties_ignored():
    cards = _cards_by_type()
    city_cap = cards[6]
    board = Board()
    board.reset(city_cap)
    board.put_card_and_meeple(
        city_cap,
        Action((-1, 0), Orientation.ROTATE_180),
        player_id=0,
    )

    board.resolve_outcomes(complete_property_only=True, consider_fields=False)

    assert all(_city_ignored_values(board))


def test_action_candidate_graph_preview_does_not_mutate_board():
    cards = _cards_by_type()
    board = Board()
    board.reset(cards[10])
    card = cards[0]
    action = board.get_possible_actions(card)[0]
    before_tiles = board.get_tiles_snapshot()
    before_graph = board.get_graph_snapshot()

    preview = board.preview_action_graph_snapshot(card, action, player_id=0)

    assert board.get_tiles_snapshot() == before_tiles
    assert board.get_graph_snapshot() == before_graph
    assert len(preview["nodes"]) > len(before_graph["nodes"])


def test_action_candidate_graph_previews_match_actions():
    cards = _cards_by_type()
    board = Board()
    board.reset(cards[10])
    card = cards[0]
    actions = board.get_possible_actions(card)

    previews = board.get_action_candidate_graph_snapshots(card, actions, player_id=0)

    assert len(previews) == len(actions)
    assert all(len(preview["nodes"]) > len(board.get_graph_snapshot()["nodes"]) for preview in previews)
