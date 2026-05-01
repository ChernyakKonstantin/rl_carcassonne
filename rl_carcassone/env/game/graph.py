from types import NoneType
from typing import Any, Dict, Hashable, List, Set, Tuple, Union

import networkx as nx

from .card import CardOption, PossibleNeighbor, PropertyMeta
from .utils import ConnectorType, Orientation, PixelMeaning


class Graph:
    def __init__(self):
        self._graph = None
        self._n_calls = None

    @staticmethod
    def _position_node_name(position: Tuple[int, int]) -> Tuple[str, int, int]:
        y, x = position
        return ("position", y, x)

    @staticmethod
    def _connector_node_name(position: Tuple[int, int], connector: ConnectorType) -> Tuple[str, int, int, ConnectorType]:
        y, x = position
        return ("connector", y, x, connector)

    @staticmethod
    def _property_node_name(position: Tuple[int, int], property_index: int) -> Tuple[str, int, int, int]:
        y, x = position
        return ("property", y, x, property_index)

    def _find_empty_nodes(self) -> List[Dict[str, Any]]:
        return [self._graph.nodes[n] for n, d in self._graph.nodes(data=True) if d.get("empty", None)]

    def _find_nodes_name_by_attribute_value(self, attribute_name: str, attribute_value: Any) -> List[Hashable]:
        """Find the first node that satisfy the criterion."""
        found = []
        for node_name, node_data in self._graph.nodes(data=True):
            if attribute_name in node_data:
                if node_data[attribute_name] == attribute_value:
                    found.append(node_name)
        return found

    def _find_position_node_name(self, position: Tuple[int, int]) -> Union[Hashable, NoneType]:
        node_name = self._position_node_name(position)
        if node_name not in self._graph:
            return None
        return node_name

    def _get_connector_nodes_names(self, position_node_name: Hashable) -> Dict[ConnectorType, Hashable]:
        position = self._graph.nodes[position_node_name]["position"]
        return {connector: self._connector_node_name(position, connector) for connector in ConnectorType}

    def _add_empty_node(self, position: Tuple[int, int]):
        position_node_name = self._position_node_name(position)
        self._graph.add_node(
            position_node_name,
            position=position,
            empty=True,
            view=None,
            possible_values=None,
        )
        for connector in ConnectorType:
            connector_node_name = self._connector_node_name(position, connector)
            self._graph.add_node(connector_node_name, connector=connector)
            self._graph.add_edge(
                position_node_name,
                connector_node_name,
                # path_for=PixelMeaning.ANY_GROWING,
            )

    def _try_add_empty_nodes_around(self, position: Tuple[int, int]):
        at_north = (position[0] - 1, position[1])
        at_south = (position[0] + 1, position[1])
        at_west = (position[0], position[1] - 1)
        at_east = (position[0], position[1] + 1)
        for neighbor_position in (at_north, at_south, at_east, at_west):
            neighbor_position_node_name = self._find_position_node_name(neighbor_position)
            if neighbor_position_node_name is None:
                self._add_empty_node(neighbor_position)

    def _try_to_link_property(
        self,
        property_position: Tuple[int, int],
        position_connector_nodes_names: Dict[ConnectorType, Hashable],
        property: PropertyMeta,
    ):
        for connector in property.connectors:
            if ConnectorType.is_north(connector):
                position_to_test = (property_position[0] - 1, property_position[1])
            elif ConnectorType.is_south(connector):
                position_to_test = (property_position[0] + 1, property_position[1])
            elif ConnectorType.is_east(connector):
                position_to_test = (property_position[0], property_position[1] + 1)
            elif ConnectorType.is_west(connector):
                position_to_test = (property_position[0], property_position[1] - 1)
            else:
                raise ValueError(f"Unknown {connector=}")
            connector_to_test = ConnectorType.inverse(connector)
            to_test_position_node_name = self._find_position_node_name(position_to_test)
            if self._graph.nodes[to_test_position_node_name]["empty"]:
                # NOTE: There is no card at the neighbor position, so we cannot link property.
                continue
            # fmt: off
            connector_to_test_node_name = self._get_connector_nodes_names(to_test_position_node_name)[connector_to_test]  # noqa: E501
            # fmt: on
            property_to_test_node_name = self._get_attached_property_node_name(connector_to_test_node_name)
            property_to_test_type = self._graph.nodes[property_to_test_node_name]["property"]
            if property_to_test_type != property.type:
                continue
            # NOTE: Because of `options` all adjacent cards are linkable.
            self._graph.add_edge(
                position_connector_nodes_names[connector],
                connector_to_test_node_name,
                path_for=property.type,
            )

    def _update_neighbors_possible_values(
        self,
        position: Tuple[int, int],
        possible_neighbors_per_side: Dict[str, Set[PossibleNeighbor]],
    ):
        for side_name, possible_neighbors in possible_neighbors_per_side.items():
            if side_name == ConnectorType.N:
                neighbor_position = (position[0] - 1, position[1])
            elif side_name == ConnectorType.S:
                neighbor_position = (position[0] + 1, position[1])
            elif side_name == ConnectorType.W:
                neighbor_position = (position[0], position[1] - 1)
            elif side_name == ConnectorType.E:
                neighbor_position = (position[0], position[1] + 1)
            else:
                raise KeyError(f"Unknown side name: {side_name}")

            position_node_name = self._find_position_node_name(neighbor_position)
            if self._graph.nodes[position_node_name]["possible_values"] is None:
                self._graph.nodes[position_node_name]["possible_values"] = possible_neighbors
            elif isinstance(self._graph.nodes[position_node_name]["possible_values"], set):
                self._graph.nodes[position_node_name]["possible_values"].intersection_update(possible_neighbors)
            else:
                possible_values_type = type(self._graph.nodes[position_node_name]["possible_values"])
                raise TypeError(f"Unexpected type of `possible_values`: {possible_values_type} at side {side_name}")

    def _get_attached_property_node_name(self, connector_node_name: Hashable) -> Hashable:
        neighbors = [n for n in self._graph.neighbors(connector_node_name) if "property" in self._graph.nodes[n]]
        if len(neighbors) != 1:
            # NOTE: A connector is attached to a single property.
            raise RuntimeError(f"Node has {len(neighbors)} neighbors instead of 1 at node {connector_node_name}")
        return neighbors[0]

    def _traverse_property(self, start_property_node_name: Hashable) -> Set[Hashable]:
        property_type = self._graph.nodes[start_property_node_name]["property"]
        visited = set()
        queue = [start_property_node_name]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            for _, neighbor, data in self._graph.edges(node, data=True):
                # if "path_for" in data and data["path_for"] in {property_type, PixelMeaning.ANY_GROWING}:
                if "path_for" in data and data["path_for"] == property_type:
                    queue.append(neighbor)
        return visited

    def get_property_owners(self, start_property_node_name: Hashable, real_only: bool) -> List[int]:
        property_type = self._graph.nodes[start_property_node_name]["property"]
        if property_type == PixelMeaning.ABBOT:
            return [self._graph.nodes[start_property_node_name]["owner"]]
        else:
            visited = self._traverse_property(start_property_node_name)
            owners = []
            for node_name in visited:
                if "owner" in self._graph.nodes[node_name]:
                    owners.append(self._graph.nodes[node_name]["owner"])
            if real_only:
                owners = list(filter(lambda x: x is not None, owners))
            return owners

    def reset(self):
        self._graph = nx.Graph()
        self._add_empty_node(position=(0, 0))
        self._n_calls = 0

    def locate_card_and_meeple(
        self,
        player_id: Union[NoneType, int],  # NOTE: `None` is for the first call only.
        card: CardOption,
        card_position: Tuple[int, int],
        meeple_position: Union[int, NoneType],
    ):
        """
        This method assumes that card, its position and meeple positions are valid.
        WARNING: No internal validation!
        """
        # NOTE: ----- State validation. -----
        if self._n_calls is None:
            raise RuntimeError("You have to call `reset` before using this graph.")
        if self._n_calls > 0 and player_id is None:
            raise ValueError("player_id is None")
        self._n_calls += 1
        # NOTE: ----- Complete state validation. -----

        # NOTE: ----- Handle card data. -----
        position_node_name = self._find_position_node_name(card_position)
        self._graph.nodes[position_node_name]["empty"] = False
        self._graph.nodes[position_node_name]["view"] = card.values
        self._try_add_empty_nodes_around(card_position)
        self._update_neighbors_possible_values(card_position, card.possible_neighbors)
        # NOTE: ----- Complete handling card data. -----

        # NOTE: ----- Handle properties. -----
        connector_nodes_names = self._get_connector_nodes_names(position_node_name)
        for property_index, property in enumerate(card.properties_metas):
            property_node_name = self._property_node_name(card_position, property_index)
            if meeple_position == property_index:
                owner = player_id
            else:
                owner = None
            if property.type == PixelMeaning.ABBOT:
                # NOTE: Abbot is a special case, since it has no connectors.

                # NOTE: We do not care about the abbot if it has no owner.
                #        If `False`, this flag is raised after players got reward for this property.
                ignore_abbot = owner is None
                self._graph.add_node(
                    property_node_name,
                    property=PixelMeaning.ABBOT,
                    owner=owner,
                    ignore=ignore_abbot,
                )
                self._graph.add_edge(property_node_name, position_node_name)
            elif property.type == PixelMeaning.EMPTY:
                pass  # TODO: I don't remember if it can happen.
            elif property.type in (PixelMeaning.ROAD, PixelMeaning.CITY, PixelMeaning.FIELD):
                extra_kwargs = dict()
                if property.type == PixelMeaning.CITY:
                    extra_kwargs["shield"] = card.shield  # TODO: This attribute must be in property.
                self._graph.add_node(
                    property_node_name,
                    property=property.type,
                    owner=owner,
                    ignore=False,  # NOTE: This flag is raised after players got reward for this property.
                    **extra_kwargs,
                )
                # NOTE: Growing property attaches to a connector to enable linking with neighbors.
                for connector in property.connectors:
                    self._graph.add_edge(
                        property_node_name,
                        connector_nodes_names[connector],
                        path_for=property.type,
                    )
                # NOTE: Try to link new property with neighbors.
                self._try_to_link_property(
                    property_position=card_position,
                    position_connector_nodes_names=connector_nodes_names,
                    property=property,
                )
            else:
                raise TypeError(f"Unknown {property.type}")
        # NOTE: ----- Complete handling properties. -----

    def get_possible_card_positions(self, card_type: int, orientation: Orientation) -> List[Tuple[int, int]]:
        to_test = PossibleNeighbor(card_type, orientation)
        empty_nodes = self._find_empty_nodes()
        possible_positions = []
        for node in empty_nodes:
            if to_test in node["possible_values"]:
                possible_positions.append(node["position"])
        return possible_positions

    def get_possible_meeple_positions(self, card_position: Tuple[int, int], card: CardOption) -> List[int]:
        possible_meeple_positions = []
        for property_index, property in enumerate(card.properties_metas):
            meeple_is_allowed = True
            if property.type == PixelMeaning.ABBOT:
                # NOTE: Abbot can be placed anywhere.
                possible_meeple_positions.append(property_index)
                continue
            for connector in property.connectors:
                if ConnectorType.is_north(connector):
                    position_to_test = (card_position[0] - 1, card_position[1])
                elif ConnectorType.is_south(connector):
                    position_to_test = (card_position[0] + 1, card_position[1])
                elif ConnectorType.is_east(connector):
                    position_to_test = (card_position[0], card_position[1] + 1)
                elif ConnectorType.is_west(connector):
                    position_to_test = (card_position[0], card_position[1] - 1)
                else:
                    raise ValueError(f"Unknown {connector=}")
                connector_to_test = ConnectorType.inverse(connector)
                to_test_position_node_name = self._find_position_node_name(position_to_test)
                if to_test_position_node_name is None or self._graph.nodes[to_test_position_node_name]["empty"]:
                    # NOTE: There is no card at the neighbor position, so we cannot check.
                    continue
                connector_node_name = self._get_connector_nodes_names(to_test_position_node_name)[connector_to_test]
                property_node_name = self._get_attached_property_node_name(connector_node_name)
                property_type = self._graph.nodes[property_node_name]["property"]
                if property_type != property.type:
                    continue
                property_owners = self.get_property_owners(
                    start_property_node_name=property_node_name,
                    real_only=True,
                )
                if len(property_owners) > 0:
                    meeple_is_allowed = False
                    break
            if meeple_is_allowed:
                possible_meeple_positions.append(property_index)
        return possible_meeple_positions

    def get_view(self) -> Dict[Tuple[int, int], str]:
        view = dict()
        for node_name, node_data in self._graph.nodes(data=True):
            if "empty" not in node_data:
                continue  # NOTE: Wrong node type.
            elif node_data["empty"]:
                continue  # NOTE: Nothing to show
            else:
                view[node_data["position"]] = node_data["view"]
        return view

    def find_owned_abbot_nodes_names(self) -> List[Hashable]:
        abbot_nodes_names = self._find_nodes_name_by_attribute_value(
            attribute_name="property",
            attribute_value=PixelMeaning.ABBOT,
        )
        non_ignored = [name for name in abbot_nodes_names if not self._graph.nodes[name]["ignore"]]
        return [name for name in non_ignored if self._graph.nodes[name]["owner"] is not None]

    def find_owned_road_nodes_names(self) -> List[Hashable]:
        road_nodes_names = self._find_nodes_name_by_attribute_value(
            attribute_name="property",
            attribute_value=PixelMeaning.ROAD,
        )
        non_ignored = set([name for name in road_nodes_names if not self._graph.nodes[name]["ignore"]])
        owned_road_nodes_names = []
        while len(non_ignored) > 0:
            node_name = non_ignored.pop()
            owned_road_nodes_names.append(node_name)
            visited = self._traverse_property(node_name)
            non_ignored = non_ignored - visited
        return owned_road_nodes_names

    def find_owned_city_nodes_names(self) -> List[Hashable]:
        city_nodes_names = self._find_nodes_name_by_attribute_value(
            attribute_name="property",
            attribute_value=PixelMeaning.CITY,
        )
        non_ignored = set([name for name in city_nodes_names if not self._graph.nodes[name]["ignore"]])
        owned_city_nodes_names = []
        while len(non_ignored) > 0:
            node_name = non_ignored.pop()
            owned_city_nodes_names.append(node_name)
            visited = self._traverse_property(node_name)
            non_ignored = non_ignored - visited
        return owned_city_nodes_names

    def ignore(self, property_node_name: Hashable):
        node_type = self._graph.nodes[property_node_name]["property"]
        if node_type == PixelMeaning.ABBOT:
            self._graph.nodes[property_node_name]["ignore"] = True
        elif node_type in {PixelMeaning.ROAD, PixelMeaning.CITY}:
            for node_name in self._traverse_property(property_node_name):
                if "ignore" in self._graph.nodes[node_name]:
                    self._graph.nodes[node_name]["ignore"] = True
        else:
            raise TypeError(f"Unexpected {node_type}")

    def _is_abbot_complete(self, abbot_node_name: Hashable) -> bool:
        neighbors = list(self._graph.neighbors(abbot_node_name))
        if len(neighbors) != 1:
            raise RuntimeError(f"Abbot has {len(neighbors)} neighbors instead of 1")
        position = self._graph.nodes[neighbors[0]]["position"]
        to_test_positions = [
            (position[0] - 1, position[1] - 1),
            (position[0] - 1, position[1]),
            (position[0] - 1, position[1] + 1),
            (position[0], position[1] - 1),
            (position[0], position[1] + 1),
            (position[0] + 1, position[1] - 1),
            (position[0] + 1, position[1]),
            (position[0] + 1, position[1] + 1),
        ]
        complete = True
        for to_test_position in to_test_positions:
            to_test_position_node_name = self._find_position_node_name(to_test_position)
            if to_test_position_node_name is None:
                complete = False
                break
            if self._graph.nodes[to_test_position_node_name]["empty"]:
                complete = False
                break
        return complete

    def _is_road_complete(self, road_node_name: Hashable) -> bool:
        visited = self._traverse_property(road_node_name)
        connector_nodes_names = []
        for node_name in visited:
            if "connector" in self._graph.nodes[node_name]:
                connector_nodes_names.append(node_name)
        if len(connector_nodes_names) == 0:
            raise RuntimeError("No connectors")
        for node_name in connector_nodes_names:
            paths_for_road = []
            for _, _, edge_data in self._graph.edges(node_name, data=True):
                if "path_for" in edge_data and edge_data["path_for"] == PixelMeaning.ROAD:
                    paths_for_road.append(edge_data)
            if len(paths_for_road) > 2:
                raise RuntimeError(f"Connector have {len(paths_for_road)} edges but expected less than 3")
            elif len(paths_for_road) == 0:
                raise ValueError("Connector has no edges with path_for=ROAD")
            elif len(paths_for_road) < 2:
                return False
        return True

    def _is_city_complete(self, city_node_name: Hashable) -> bool:
        visited = self._traverse_property(city_node_name)
        connector_nodes_names = []
        for node_name in visited:
            if "connector" in self._graph.nodes[node_name]:
                connector_nodes_names.append(node_name)
        if len(connector_nodes_names) == 0:
            raise RuntimeError("No connectors")
        for node_name in connector_nodes_names:
            paths_for_city = []
            for _, _, edge_data in self._graph.edges(node_name, data=True):
                if "path_for" in edge_data and edge_data["path_for"] == PixelMeaning.CITY:
                    paths_for_city.append(edge_data)
            if len(paths_for_city) > 2:
                raise RuntimeError(f"Connector have {len(paths_for_city)} edges but expected less than 3")
            if len(paths_for_city) == 0:
                raise ValueError("Connector has no edges with path_for=CITY")
            elif len(paths_for_city) < 2:
                return False
        return True

    def is_property_complete(self, property_node_name: Hashable) -> bool:
        node_type = self._graph.nodes[property_node_name]["property"]
        if node_type == PixelMeaning.ABBOT:
            return self._is_abbot_complete(property_node_name)
        elif node_type == PixelMeaning.ROAD:
            return self._is_road_complete(property_node_name)
        elif node_type == PixelMeaning.CITY:
            return self._is_city_complete(property_node_name)
        else:
            raise TypeError(f"Unexpected {node_type}")

    def get_scores_for_abbot(self, abbot_node_name: Hashable) -> int:
        result = []
        for n in self._graph.neighbors(abbot_node_name):
            if "position" in self._graph.nodes[n] and not self._graph.nodes[n]["empty"]:
                result.append(n)
        return len(result) + 1

    def get_scores_for_city(self, city_node_name: Hashable, is_complete: bool) -> int:
        visited = self._traverse_property(city_node_name)
        scores = 0
        for node_name in visited:
            if "property" in self._graph.nodes[node_name]:
                scores += 1
                if self._graph.nodes[node_name]["shield"]:
                    scores += 1
        if is_complete:
            scores *= 2
        return scores
