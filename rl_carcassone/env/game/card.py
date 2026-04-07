from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Set, Union

from typing_extensions import Self

from .utils import ConnectorType, Orientation, PixelMeaning


@dataclass(frozen=True)
class PropertyMeta:
    type: PixelMeaning
    connectors: Set[ConnectorType]

    @classmethod
    def from_dict(cls, d: Dict[str, Union[str, List[str]]]) -> Self:
        return cls(
            type=PixelMeaning.from_symbol(d["type"]),
            connectors={ConnectorType.from_name(name) for name in d["connectors"]},
        )


@dataclass(frozen=True)
class PossibleNeighbor:
    type: int
    orientation: Orientation


@dataclass(frozen=True)
class CardOption:
    possible_neighbors: Dict[str, Set[PossibleNeighbor]]  # TODO: Use `ConnectorType` instead of `str`.
    values: str
    properties: List[int]
    properties_metas: List[PropertyMeta]
    shield: bool  # TODO: This attribute must be in city property.


class Card:
    def __init__(self, type: int, shield: bool, options: Dict[int, Dict]):
        self.type = type
        self.shield = shield
        # NOTE: Make copy to not alter underlying data for other cards.
        self.options = self._convert_options(deepcopy(options))

    def _convert_options(self, options: Dict[Union[str, int], Dict]) -> Dict[Orientation, CardOption]:
        converted_options = dict()
        for option_orientation, option in options.items():
            possible_neighbors = dict()
            for side, neighbor_list in option["possible_neighbors"].items():
                side = ConnectorType.from_name(side)
                possible_neighbors[side] = set()
                for neighbor_type, neighbor_orientation in neighbor_list:
                    possible_neighbor = PossibleNeighbor(neighbor_type, Orientation.from_angle(neighbor_orientation))
                    possible_neighbors[side].add(possible_neighbor)

            converted_options[Orientation.from_angle(option_orientation)] = CardOption(
                possible_neighbors=possible_neighbors,
                values=option["values"],
                properties=option["properties"],
                shield=self.shield,
                properties_metas=[PropertyMeta.from_dict(e) for e in option["properties_metas"]],
            )
        return converted_options

    def get_option(self, orientation: Orientation):
        return self.options[orientation]
