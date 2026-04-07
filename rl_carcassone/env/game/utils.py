from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Union

from matplotlib.colors import LinearSegmentedColormap
from typing_extensions import Self

property_cmap = LinearSegmentedColormap.from_list(
    "property_cmap",
    [
        "black",  # Empty
        "green",  # Field
        "gray",  # Road
        "blue",  # City
        "red",  # Abbot
    ],
)


class ConnectorType(IntEnum):
    N = 0
    E = 1
    S = 2
    W = 3
    NW = 4
    NE = 5
    EN = 6
    ES = 7
    SW = 8
    SE = 9
    WN = 10
    WS = 11

    @staticmethod
    def from_name(name: str) -> Self:
        return ConnectorType[name]

    @staticmethod
    def is_north(connector: Self) -> bool:
        return connector in (ConnectorType.N, ConnectorType.NE, ConnectorType.NW)

    @staticmethod
    def is_south(connector: Self) -> bool:
        return connector in (ConnectorType.S, ConnectorType.SE, ConnectorType.SW)

    @staticmethod
    def is_east(connector: Self) -> bool:
        return connector in (ConnectorType.E, ConnectorType.EN, ConnectorType.ES)

    @staticmethod
    def is_west(connector: Self) -> bool:
        return connector in (ConnectorType.W, ConnectorType.WN, ConnectorType.WS)

    @staticmethod
    def inverse(connector: Self) -> Self:
        if connector == ConnectorType.N:
            return ConnectorType.S
        elif connector == ConnectorType.E:
            return ConnectorType.W
        elif connector == ConnectorType.S:
            return ConnectorType.N
        elif connector == ConnectorType.W:
            return ConnectorType.E
        elif connector == ConnectorType.NW:
            return ConnectorType.SW
        elif connector == ConnectorType.NE:
            return ConnectorType.SE
        elif connector == ConnectorType.SW:
            return ConnectorType.NW
        elif connector == ConnectorType.SE:
            return ConnectorType.NE
        elif connector == ConnectorType.EN:
            return ConnectorType.WN
        elif connector == ConnectorType.ES:
            return ConnectorType.WS
        elif connector == ConnectorType.WN:
            return ConnectorType.EN
        elif connector == ConnectorType.WS:
            return ConnectorType.ES
        else:
            raise ValueError(f"Unknown {connector=}")


class PixelMeaning(IntEnum):
    EMPTY = 0
    FIELD = 1
    ROAD = 2
    CITY = 3
    ABBOT = 4
    ANY_GROWING = 5

    @staticmethod
    def from_symbol(s: str) -> Self:
        if s == "E":
            return PixelMeaning.EMPTY
        elif s == "F":
            return PixelMeaning.FIELD
        elif s == "R":
            return PixelMeaning.ROAD
        elif s == "C":
            return PixelMeaning.CITY
        elif s == "A":
            return PixelMeaning.ABBOT
        else:
            raise ValueError(f"Unknown symbol {s}")


class Orientation(IntEnum):
    ROTATE_0 = 0
    ROTATE_90 = 1
    ROTATE_180 = 2
    ROTATE_270 = 3

    @staticmethod
    def from_angle(a: Union[str, int]) -> Self:
        if isinstance(a, str):
            a = int(a)
        if a == 0:
            return Orientation.ROTATE_0
        elif a == 90:
            return Orientation.ROTATE_90
        elif a == 180:
            return Orientation.ROTATE_180
        elif a == 270:
            return Orientation.ROTATE_270
        else:
            raise ValueError(f"Unknown angle {a}")


@dataclass
class Action:
    position: Tuple[int, int]  # NOTE: As (y, x).
    orientation: Orientation
    meeple_position: int = None
