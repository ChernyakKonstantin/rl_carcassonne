from pycarcassone.pycarcassone.board import Board
from pycarcassone.pycarcassone.deck import Deck
from pycarcassone.pycarcassone.utils import Action, Orientation


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
