from dataclasses import dataclass
from types import NoneType
from typing import Any, Dict, Hashable, Iterator, List, Set, Tuple, Union

import networkx as nx

from .card import CardOption, PossibleNeighbor, PropertyMeta
from .utils import ConnectorType, Orientation, PixelMeaning


@dataclass(frozen=True)
class PropertyComponent:
    type: PixelMeaning
    representative_node_name: Hashable
    node_names: Set[Hashable]
    property_node_names: List[Hashable]
    connector_node_names: List[Hashable]
    owners: List[Union[NoneType, int]]


class Graph:
    def __init__(self):
        self._graph = None
        self._n_calls = None

    @staticmethod
    def _position_node_name(position: Tuple[int, int]) -> Tuple[str, int, int]:
        y, x = position
        return ("position", y, x)

    @staticmethod
    def _connector_node_name(
        position: Tuple[int, int],
        connector: ConnectorType,
    ) -> Tuple[str, int, int, ConnectorType]:
        y, x = position
        return ("connector", y, x, connector)

    @staticmethod
    def _property_node_name(position: Tuple[int, int], property_index: int) -> Tuple[str, int, int, int]:
        y, x = position
        return ("property", y, x, property_index)

    @staticmethod
    def _get_neighbor_position(position: Tuple[int, int], connector: ConnectorType) -> Tuple[int, int]:
        """Return the single cardinal neighbor reached by a connector on one side of a tile."""
        if ConnectorType.is_north(connector):
            return (position[0] - 1, position[1])
        elif ConnectorType.is_south(connector):
            return (position[0] + 1, position[1])
        elif ConnectorType.is_east(connector):
            return (position[0], position[1] + 1)
        elif ConnectorType.is_west(connector):
            return (position[0], position[1] - 1)
        else:
            raise ValueError(f"Unknown {connector=}")

    @staticmethod
    def _get_neighbor_card_positions(position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return the four side-adjacent card positions; diagonals are not valid card neighbors."""
        return [
            (position[0] - 1, position[1]),
            (position[0] + 1, position[1]),
            (position[0], position[1] - 1),
            (position[0], position[1] + 1),
        ]

    @staticmethod
    def _get_surrounding_positions(position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return all eight positions around a tile, including diagonals; used for abbot completion."""
        return [
            (position[0] + y_delta, position[1] + x_delta)
            for y_delta in (-1, 0, 1)
            for x_delta in (-1, 0, 1)
            if y_delta != 0 or x_delta != 0
        ]

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

    def _get_position_connector_nodes_names(self, position_node_name: Hashable) -> Dict[ConnectorType, Hashable]:
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
        for neighbor_position in self._get_neighbor_card_positions(position):
            neighbor_position_node_name = self._find_position_node_name(neighbor_position)
            if neighbor_position_node_name is None:
                self._add_empty_node(neighbor_position)

    def _find_neighbor_connector_node_name(
        self,
        property_position: Tuple[int, int],
        connector: ConnectorType,
    ) -> Union[Hashable, NoneType]:
        neighbor_position = self._get_neighbor_position(property_position, connector)
        neighbor_position_node_name = self._find_position_node_name(neighbor_position)
        if neighbor_position_node_name is None or self._graph.nodes[neighbor_position_node_name]["empty"]:
            return None
        neighbor_connector = ConnectorType.inverse(connector)
        return self._get_position_connector_nodes_names(neighbor_position_node_name)[neighbor_connector]

    def _try_to_link_property(
        self,
        property_position: Tuple[int, int],
        position_connector_nodes_names: Dict[ConnectorType, Hashable],
        property: PropertyMeta,
    ):
        for connector in property.connectors:
            neighbor_connector_node_name = self._find_neighbor_connector_node_name(
                property_position=property_position,
                connector=connector,
            )
            if neighbor_connector_node_name is None:
                # NOTE: There is no card at the neighbor position, so we cannot link property.
                continue
            neighbor_property_node_name = self._get_attached_property_node_name(neighbor_connector_node_name)
            neighbor_property_type = self._graph.nodes[neighbor_property_node_name]["property"]
            if neighbor_property_type != property.type:
                raise ValueError(f"Unexpected property type mismatch: {neighbor_property_type=} != {property.type=}")
            # NOTE: Because of `options` all adjacent cards are linkable.
            self._graph.add_edge(
                position_connector_nodes_names[connector],
                neighbor_connector_node_name,
                path_for=property.type,
            )

    def _update_neighbors_possible_values(
        self,
        position: Tuple[int, int],
        possible_neighbors_per_side: Dict[str, Set[PossibleNeighbor]],
    ):
        for side_name, possible_neighbors in possible_neighbors_per_side.items():
            neighbor_position = self._get_neighbor_position(position, side_name)
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

    def _build_property_component(
        self,
        representative_node_name: Hashable,
        node_names: Set[Hashable],
    ) -> PropertyComponent:
        property_type = self._graph.nodes[representative_node_name]["property"]
        property_node_names = []
        connector_node_names = []
        owners = []
        for node_name in node_names:
            if "property" in self._graph.nodes[node_name]:
                property_node_names.append(node_name)
                owners.append(self._graph.nodes[node_name]["owner"])
            elif "connector" in self._graph.nodes[node_name]:
                connector_node_names.append(node_name)
        return PropertyComponent(
            type=property_type,
            representative_node_name=representative_node_name,
            node_names=node_names,
            property_node_names=property_node_names,
            connector_node_names=connector_node_names,
            owners=owners,
        )

    def iter_property_components(self, property_type: PixelMeaning) -> Iterator[PropertyComponent]:
        property_node_names = self._find_nodes_name_by_attribute_value(
            attribute_name="property",
            attribute_value=property_type,
        )
        non_ignored = set([name for name in property_node_names if not self._graph.nodes[name]["ignore"]])
        while len(non_ignored) > 0:
            representative_node_name = non_ignored.pop()
            node_names = self._traverse_property(representative_node_name)
            component = self._build_property_component(representative_node_name, node_names)
            non_ignored = non_ignored - component.node_names
            yield component

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
        position_connector_nodes_names = self._get_position_connector_nodes_names(position_node_name)
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
                        position_connector_nodes_names[connector],
                        path_for=property.type,
                    )
                # NOTE: Try to link new property with neighbors.
                self._try_to_link_property(
                    property_position=card_position,
                    position_connector_nodes_names=position_connector_nodes_names,
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

    @staticmethod
    def _is_abbot_property(property: PropertyMeta) -> bool:
        return property.type == PixelMeaning.ABBOT

    def _neighbor_property_has_owner(
        self,
        property_node_name: Hashable,
        cache: Dict[Hashable, bool],
    ) -> bool:
        if property_node_name not in cache:
            property_owners = self.get_property_owners(
                start_property_node_name=property_node_name,
                real_only=True,
            )
            cache[property_node_name] = len(property_owners) > 0
        return cache[property_node_name]

    def _neighbor_properties_allows_meeple(
        self,
        card_position: Tuple[int, int],
        property: PropertyMeta,
        owner_cache: Dict[Hashable, bool],
    ) -> bool:
        """
        `owner_cache` stores whether an already placed neighboring property node reaches a real owner.

        A candidate card can have separate properties that touch the same neighboring property node through different
        connectors, e.g. a FIELD/ROAD/FIELD side facing a neighboring tile where both field edge segments belong to
        one field. Caching avoids repeated traversals in `get_property_owners`.
        """
        for connector in property.connectors:
            position_to_test = self._get_neighbor_position(card_position, connector)
            connector_to_test = ConnectorType.inverse(connector)
            to_test_position_node_name = self._find_position_node_name(position_to_test)
            if to_test_position_node_name is None or self._graph.nodes[to_test_position_node_name]["empty"]:
                # NOTE: There is no card at the neighbor position, so we cannot check.
                continue
            # fmt: off
            connector_node_name = self._get_position_connector_nodes_names(to_test_position_node_name)[connector_to_test]
            # fmt: on
            property_node_name = self._get_attached_property_node_name(connector_node_name)
            property_type = self._graph.nodes[property_node_name]["property"]
            if property_type != property.type:
                raise ValueError(f"Unexpected property type mismatch: {property_type=} != {property.type=}")
            if self._neighbor_property_has_owner(property_node_name, owner_cache):
                return False
        return True

    def get_possible_meeple_positions(self, card_position: Tuple[int, int], card: CardOption) -> List[int]:
        possible_meeple_positions = []
        # NOTE: Shared across candidate properties on this card; keys are neighboring property node names.
        owner_cache = dict()
        for property_index, property in enumerate(card.properties_metas):
            if self._is_abbot_property(property):
                possible_meeple_positions.append(property_index)
            elif self._neighbor_properties_allows_meeple(card_position, property, owner_cache):
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

    def ignore_abbot(self, property_node_name: Hashable):
        self._graph.nodes[property_node_name]["ignore"] = True

    def ignore_property_component(self, component: PropertyComponent):
        if component.type in {PixelMeaning.ROAD, PixelMeaning.CITY}:
            for node_name in component.node_names:
                if "ignore" in self._graph.nodes[node_name]:
                    self._graph.nodes[node_name]["ignore"] = True
        else:
            raise TypeError(f"Unexpected {component.type}")

    def is_abbot_complete(self, abbot_node_name: Hashable) -> bool:
        neighbors = list(self._graph.neighbors(abbot_node_name))
        if len(neighbors) != 1:
            raise RuntimeError(f"Abbot has {len(neighbors)} neighbors instead of 1")
        position = self._graph.nodes[neighbors[0]]["position"]
        complete = True
        for to_test_position in self._get_surrounding_positions(position):
            to_test_position_node_name = self._find_position_node_name(to_test_position)
            if to_test_position_node_name is None:
                complete = False
                break
            if self._graph.nodes[to_test_position_node_name]["empty"]:
                complete = False
                break
        return complete

    def is_growing_property_component_complete(self, component: PropertyComponent) -> bool:
        if len(component.connector_node_names) == 0:
            raise RuntimeError("No connectors")
        for node_name in component.connector_node_names:
            paths_for_property = []
            for _, _, edge_data in self._graph.edges(node_name, data=True):
                if "path_for" in edge_data and edge_data["path_for"] == component.type:
                    paths_for_property.append(edge_data)
            if len(paths_for_property) > 2:
                raise RuntimeError(f"Connector have {len(paths_for_property)} edges but expected less than 3")
            elif len(paths_for_property) == 0:
                raise ValueError(f"Connector has no edges with path_for={component.type.name}")
            elif len(paths_for_property) < 2:
                return False
        return True

    def get_scores_for_abbot(self, abbot_node_name: Hashable) -> int:
        result = []
        for n in self._graph.neighbors(abbot_node_name):
            if "position" in self._graph.nodes[n] and not self._graph.nodes[n]["empty"]:
                result.append(n)
        return len(result) + 1

    def get_scores_for_road_component(self, component: PropertyComponent) -> int:
        return len(component.owners)

    def get_scores_for_city_component(self, component: PropertyComponent, is_complete: bool) -> int:
        scores = 0
        for node_name in component.property_node_names:
            scores += 1
            if self._graph.nodes[node_name]["shield"]:
                scores += 1
        if is_complete:
            scores *= 2
        return scores
