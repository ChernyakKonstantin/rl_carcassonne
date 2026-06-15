from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .card import Card
from .graph import Graph
from .utils import Action, Orientation, PixelMeaning


@dataclass
class StepResult:
    score: int = 0
    returned_meeples: int = 0


class Board:
    def __init__(self):
        self._graph = Graph()

    def clone(self) -> "Board":
        """Return an independent board copy for previews and adapter-side branching."""
        board = Board()
        board._graph = self._graph.clone()
        return board

    def _apply_majority_outcome(
        self,
        step_results: Dict[int, StepResult],
        owners: List[int],
        n_scores: int,
    ):
        """In-place filling of `step_results` using Carcassonne majority rule."""
        real_owners = list(filter(lambda x: x is not None, owners))
        if len(real_owners) == 0:
            return
        owner_ids, meeples_per_owner = np.unique(real_owners, return_counts=True)
        max_meeples = meeples_per_owner.max()
        for owner_id, n_meeples in zip(owner_ids, meeples_per_owner):
            n_meeples = int(n_meeples)
            if n_meeples == max_meeples:
                step_results[owner_id].score += n_scores
            step_results[owner_id].returned_meeples += n_meeples

    def _get_abbot_outcomes(self, step_results: Dict[int, StepResult], complete_property_only: bool):
        """In-place filling of `step_results`."""
        for node_name in self._graph.find_owned_abbot_nodes_names():
            to_process = True
            if complete_property_only:
                to_process = self._graph.is_abbot_complete(node_name)
            if not to_process:
                continue
            owner_id = self._graph.get_property_owners(node_name, real_only=True)[0]  # NOTE: Abbot has a single owner.
            n_scores = self._graph.get_scores_for_abbot(node_name)
            if complete_property_only and n_scores != 9:
                raise ValueError(f"Complete abbot must have 9 scores, while {n_scores} are obtained")
            step_results[owner_id].score += n_scores
            step_results[owner_id].returned_meeples += 1
            self._graph.ignore_abbot(node_name)

    def _get_road_outcomes(self, step_results: Dict[int, StepResult], complete_property_only: bool):
        """In-place filling of `step_results`."""
        for component in self._graph.iter_property_components(PixelMeaning.ROAD):
            to_process = True
            if complete_property_only:
                to_process = self._graph.is_growing_property_component_complete(component)
            if not to_process:
                continue
            n_scores = self._graph.get_scores_for_road_component(component)
            owners = component.owners
            self._apply_majority_outcome(step_results, owners, n_scores)
            self._graph.ignore_property_component(component)

    def _get_city_outcomes(self, step_results: Dict[int, StepResult], complete_property_only: bool):
        for component in self._graph.iter_property_components(PixelMeaning.CITY):
            is_property_complete = self._graph.is_growing_property_component_complete(component)
            if complete_property_only and not is_property_complete:
                continue
            n_scores = self._graph.get_scores_for_city_component(component, is_property_complete)
            owners = component.owners
            self._apply_majority_outcome(step_results, owners, n_scores)
            self._graph.ignore_property_component(component)

    def _get_field_outcomes(self, step_results: Dict[int, StepResult]):
        for component in self._graph.iter_property_components(PixelMeaning.FIELD):
            n_scores = self._graph.get_scores_for_field_component(component)
            owners = component.owners
            self._apply_majority_outcome(step_results, owners, n_scores)
            self._graph.ignore_property_component(component)

    def reset(self, card: Card):
        """This method clears all graphs and puts initial card."""
        self._graph.reset()
        action = Action((0, 0), Orientation.ROTATE_0, meeple_position=None)
        self.put_card_and_meeple(card, action, player_id=None)

    def resolve_outcomes(self, complete_property_only: bool, consider_fields: bool) -> Dict[int, StepResult]:
        """Resolve scoring outcomes and mark scored graph properties as ignored."""
        step_results = defaultdict(StepResult)
        self._get_abbot_outcomes(step_results, complete_property_only)
        self._get_road_outcomes(step_results, complete_property_only)
        self._get_city_outcomes(step_results, complete_property_only)
        if consider_fields:
            self._get_field_outcomes(step_results)
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

    def get_tiles_snapshot(self) -> List[Dict]:
        return self._graph.get_tiles_snapshot()

    def get_graph_snapshot(self) -> Dict[str, List]:
        return self._graph.get_graph_snapshot()

    def preview_action_graph_snapshot(self, card: Card, action: Action, player_id: int) -> Dict[str, List]:
        """Return the graph snapshot after applying an action to a board copy.

        The real board is not mutated. This preview applies tile placement and
        optional meeple placement only; it does not resolve scoring outcomes.
        """
        board = self.clone()
        board.put_card_and_meeple(card, action, player_id)
        return board.get_graph_snapshot()

    def get_action_candidate_graph_snapshots(
        self,
        card: Card,
        actions: List[Action],
        player_id: int,
    ) -> List[Dict[str, List]]:
        """Return one preview graph snapshot per legal action candidate."""
        return [self.preview_action_graph_snapshot(card, action, player_id) for action in actions]

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
