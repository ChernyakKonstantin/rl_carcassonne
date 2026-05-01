from pycarcassone.pycarcassone.deck import Deck
from pycarcassone.pycarcassone.graph import Graph
from pycarcassone.pycarcassone.utils import ConnectorType, Orientation, PixelMeaning


def _cards_by_type():
    return {card.type: card for card in Deck()._load()}


def test_field_meeple_is_blocked_by_neighbor_field_owner():
    cards = _cards_by_type()
    neighbor_card = cards[10].get_option(Orientation.ROTATE_0)
    field_card = cards[0].get_option(Orientation.ROTATE_0)

    graph = Graph()
    graph.reset()
    graph.locate_card_and_meeple(
        player_id=0,
        card=neighbor_card,
        card_position=(0, 0),
        meeple_position=1,
    )

    possible_meeples = graph.get_possible_meeple_positions((1, 0), field_card)

    assert 0 not in possible_meeples


def test_field_is_linked_to_neighbor_field_on_same_connector_name():
    cards = _cards_by_type()
    neighbor_card = cards[10].get_option(Orientation.ROTATE_0)
    field_card = cards[0].get_option(Orientation.ROTATE_0)

    graph = Graph()
    graph.reset()
    graph.locate_card_and_meeple(
        player_id=None,
        card=neighbor_card,
        card_position=(0, 0),
        meeple_position=None,
    )
    graph.locate_card_and_meeple(
        player_id=0,
        card=field_card,
        card_position=(1, 0),
        meeple_position=None,
    )

    field_position_node_name = graph._find_position_node_name((1, 0))
    neighbor_position_node_name = graph._find_position_node_name((0, 0))
    field_connector_node_name = graph._get_position_connector_nodes_names(field_position_node_name)[ConnectorType.N]
    neighbor_connector_node_name = graph._get_position_connector_nodes_names(neighbor_position_node_name)[
        ConnectorType.S
    ]

    assert graph._graph.has_edge(field_connector_node_name, neighbor_connector_node_name)

    field_property_node_name = graph._get_attached_property_node_name(field_connector_node_name)
    neighbor_property_node_name = graph._get_attached_property_node_name(neighbor_connector_node_name)
    assert graph._graph.nodes[field_property_node_name]["property"] == PixelMeaning.FIELD
    assert graph._graph.nodes[neighbor_property_node_name]["property"] == PixelMeaning.FIELD


def test_graph_uses_position_based_node_names():
    graph = Graph()
    graph.reset()

    assert graph._find_position_node_name((0, 0)) == ("position", 0, 0)
    assert graph._get_position_connector_nodes_names(("position", 0, 0))[ConnectorType.N] == (
        "connector",
        0,
        0,
        ConnectorType.N,
    )


def test_completed_city_scores_adjacent_field_once():
    cards = _cards_by_type()
    city_cap = cards[6]

    graph = Graph()
    graph.reset()
    graph.locate_card_and_meeple(
        player_id=0,
        card=city_cap.get_option(Orientation.ROTATE_0),
        card_position=(0, 0),
        meeple_position=1,
    )
    graph.locate_card_and_meeple(
        player_id=1,
        card=city_cap.get_option(Orientation.ROTATE_180),
        card_position=(-1, 0),
        meeple_position=None,
    )

    owned_field_components = [
        component for component in graph.iter_property_components(PixelMeaning.FIELD) if component.owners == [0]
    ]

    assert len(owned_field_components) == 1
    assert graph.get_scores_for_field_component(owned_field_components[0]) == 3
