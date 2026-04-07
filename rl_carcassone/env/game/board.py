from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Union

import networkx as nx
import numpy as np

from .card import Card, CardOption
from .graph import Graph
from .utils import Action, ConnectorType, Orientation, PixelMeaning


@dataclass
class StepResult:
    score: int = 0
    returned_meeples: int = 0


class Board:
    def __init__(self):
        self._graph = Graph()

    def reset(self, card: Card):
        """This method clears all graphs and puts initial card."""
        self._graph.reset()
        action = Action((0, 0), Orientation.ROTATE_0, meeple_position=None)
        self.put_card_and_meeple(card, action, player_id=None)

    def get_outcomes(self, complete_property_only: bool, consider_fields: bool) -> Dict[int, StepResult]:
        step_results = defaultdict(StepResult)

        for node_name in self._graph.find_owned_abbot_nodes_names():
            if complete_property_only:
                to_process = self._graph.is_property_complete(node_name)
            else:
                to_process = True
            if to_process:
                owner_id = self._graph.get_property_owners(node_name, real_only=True)[0]  # NOTE: Abbot has a single owner.
                n_scores = self._graph.get_scores_for_abbot(node_name)
                if complete_property_only and n_scores != 9:
                    raise ValueError(f"Complete abbot must have 9 scores, while {n_scores} are obtained")
                step_results[owner_id].score += n_scores
                step_results[owner_id].returned_meeples += 1
                self._graph.ignore(node_name)

        for node_name in self._graph.find_owned_road_nodes_names():
            if complete_property_only:
                to_process = self._graph.is_property_complete(node_name)
            else:
                to_process = True
            if to_process:
                owners = self._graph.get_property_owners(node_name, real_only=False)
                n_scores = len(owners)
                real_owners = list(filter(lambda x: x is not None, owners))
                if len(real_owners) > 0:
                    owner_ids, meeples_per_owner = np.unique(real_owners, return_counts=True)
                    max_meeples = meeples_per_owner.max()
                    for owner_id, n_meeples in zip(owner_ids, meeples_per_owner):
                        n_meeples = int(n_meeples)
                        if n_meeples == max_meeples:
                            step_results[owner_id].score += n_scores
                            step_results[owner_id].returned_meeples += n_meeples
                        else:
                            step_results[owner_id].score += 0
                            step_results[owner_id].returned_meeples += n_meeples
                self._graph.ignore(node_name)

        for node_name in self._graph.find_owned_city_nodes_names():
            is_property_complete = self._graph.is_property_complete(node_name)
            if complete_property_only:
                to_process = is_property_complete
            else:
                to_process = True
            if to_process:
                owners = self._graph.get_property_owners(node_name, real_only=False)
                n_scores = self._graph.get_scores_for_city(node_name, is_property_complete)
                real_owners = list(filter(lambda x: x is not None, owners))
                if len(real_owners) > 0:
                    owner_ids, meeples_per_owner = np.unique(real_owners, return_counts=True)
                    max_meeples = meeples_per_owner.max()
                    for owner_id, n_meeples in zip(owner_ids, meeples_per_owner):
                        n_meeples = int(n_meeples)
                        if n_meeples == max_meeples:
                            step_results[owner_id].score += n_scores
                            step_results[owner_id].returned_meeples += n_meeples
                        else:
                            step_results[owner_id].score += 0
                            step_results[owner_id].returned_meeples += n_meeples
                self._graph.ignore(node_name)

        return step_results

    def put_card_and_meeple(self, card: Card, action: Action, player_id: int) -> Dict[int, StepResult]:
        """It is expected that `actions` comes from set of valid actions (see `get_possible_actions`)."""
        selected_option = card.get_option(action.orientation)
        self._graph.locate_card_and_meeple(
            player_id=player_id,
            card=selected_option,
            card_position=action.position,
            meeple_position=action.meeple_position,
        )

    def get_view(self) -> np.ndarray:
        # TODO: I need to return meeples positions somehow!
        CELL_SIZE = 5
        position_to_values = self._graph.get_view()
        ys, xs = [], []
        for y, x in position_to_values.keys():
            xs.append(x)
            ys.append(y)
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        width = (max_x - min_x + 1) * CELL_SIZE
        height = (max_y - min_y + 1) * CELL_SIZE
        view = np.zeros((height, width), dtype=np.int32)

        for (y, x), value in position_to_values.items():
            x -= min_x
            y -= min_y
            start_x = x * CELL_SIZE
            end_x = start_x + CELL_SIZE
            start_y = y * CELL_SIZE
            end_y = start_y + CELL_SIZE
            card_view = np.reshape(
                [PixelMeaning.from_symbol(v).value for v in value],
                (CELL_SIZE, CELL_SIZE),
            )
            view[start_y:end_y, start_x:end_x] = card_view
        return view

    def get_possible_actions(self, card: Card) -> List[Action]:
        possible_actions = []
        for orientation in card.options.keys():
            possible_card_positions = self._graph.get_possible_card_positions(card.type, orientation)
            for card_position in possible_card_positions:
                # NOTE: If option fits, it can be put with no meeples.
                possible_actions.append(Action(card_position, orientation))
                possible_meeple_positions = self._graph.get_possible_meeple_positions(
                    card_position,
                    card.get_option(orientation),
                )
                for meeple_position in possible_meeple_positions:
                    possible_actions.append(Action(card_position, orientation, meeple_position))
        return possible_actions
