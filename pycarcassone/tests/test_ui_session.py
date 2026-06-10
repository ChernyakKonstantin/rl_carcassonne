import json

from pycarcassone.pycarcassone.board import Board
from pycarcassone.pycarcassone.deck import Deck
from pycarcassone.pycarcassone.ui import HumanGameSession


def _cards_by_type():
    return {card.type: card for card in Deck()._load()}


def test_tiles_snapshot_exposes_property_owner():
    cards = _cards_by_type()
    board = Board()
    board.reset(cards[10])
    card = cards[0]
    action = next(action for action in board.get_possible_actions(card) if action.meeple_position is not None)

    board.put_card_and_meeple(card, action, player_id=7)

    tile = next(tile for tile in board.get_tiles_snapshot() if tile["position"] == action.position)
    property_data = tile["property_data"][action.meeple_position]

    assert tile["properties"] == card.get_option(action.orientation).properties
    assert property_data["owner"] == 7
    assert property_data["owners"] == [7]


def test_tiles_snapshot_exposes_component_owners_on_connected_properties():
    cards = _cards_by_type()
    board = Board()
    board.reset(cards[10])
    card = cards[0]
    action = next(action for action in board.get_possible_actions(card) if action.meeple_position is not None)

    board.put_card_and_meeple(card, action, player_id=7)

    initial_tile = next(tile for tile in board.get_tiles_snapshot() if tile["position"] == (0, 0))

    assert initial_tile["property_data"][1]["owner"] is None
    assert initial_tile["property_data"][1]["owners"] == [7]


def test_tiles_snapshot_exposes_city_shield():
    cards = _cards_by_type()
    board = Board()
    board.reset(cards[21])

    tile = board.get_tiles_snapshot()[0]
    city_properties = [
        property_data for property_data in tile["property_data"].values() if property_data["type"] == "CITY"
    ]

    assert any(property_data["shield"] for property_data in city_properties)


def test_human_session_state_is_json_serializable_and_has_legal_actions():
    session = HumanGameSession(seed=67, n_opponents=2)

    state = session.to_dict()

    json.dumps(state)
    assert state["message"] == "Your turn."
    assert state["current_turn"] is not None
    assert len(state["current_turn"]["actions"]) > 0
    assert len(state["board"]["tiles"]) > 0
    assert "owners" in state["board"]["tiles"][0]["property_data"][0]


def test_human_session_serializes_current_card_shields():
    session = HumanGameSession(seed=67, n_opponents=2)

    state = session.to_dict()
    shielded_city_properties = [
        property_data
        for option in state["current_turn"]["card"]["options"]
        for property_data in option["property_types"]
        if property_data["type"] == "CITY" and property_data["shield"]
    ]

    assert len(shielded_city_properties) > 0


def test_human_session_applies_human_action_and_advances_back_to_human():
    session = HumanGameSession(seed=67, n_opponents=2)
    state = session.to_dict()
    action_index = state["current_turn"]["actions"][0]["index"]

    session.apply_action(action_index)
    next_state = session.to_dict()

    assert next_state["terminal"] or next_state["message"] == "Your turn."
    if not next_state["terminal"]:
        assert next_state["current_turn"] is not None
        assert len(next_state["current_turn"]["actions"]) > 0
