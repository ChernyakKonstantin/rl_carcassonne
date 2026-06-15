from typing import Any, Dict, Hashable, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pycarcassone.game import GameEngine
from pycarcassone.player import PlayerState, RandomPlayer
from pycarcassone.utils import ConnectorType, PixelMeaning

from .spaces import DynamicDiscrete, HeterogeneousGraph, HeterogeneousGraphInstance


class CarcassonneEnv(gym.Env):
    """
    Gymnasium wrapper with a dynamic legal action set.

    The action passed to ``step`` is an integer index into the current
    ``observation["action_candidate_graphs"]`` sequence.

    Action space:
        ``DynamicDiscrete()``. It validates only that an action is a
        non-negative integer. ``step`` still rejects indices outside the
        current candidate graph sequence.

    Observation space:
        ``spaces.Dict`` with the following fields:

        - ``action_candidate_graphs``:
          ``spaces.Sequence(HeterogeneousGraph(...), stack=False)``. Candidate
          graph ``i`` is produced by applying live engine action
          ``current_turn.actions[i]`` to a board copy; ``step(i)`` applies that
          same action to the real game. The current board graph, current tile,
          and raw legal-action metadata are intentionally not separate
          top-level observation fields while candidate graphs are returned
          directly.
        - ``players``:
          ``spaces.Box(shape=(n_players, 3), dtype=np.int32)`` sorted by engine
          player id. Each row is ``[player_id, score, remaining_meeples]``.
          In this environment, the trainable agent is always player ``0``.
        - ``player_order``:
          ``spaces.Box(shape=(n_players,), dtype=np.int32)`` with the current
          engine turn order as player ids.
        - ``n_remaining_cards``:
          ``spaces.Discrete(73)``. The game has 72 cards remaining after the
          initial board tile, so valid observation values are ``0..72``.

    Candidate graph schema:
        Position, connector, and property nodes are separate node types.
        Position node features are ``[y, x, empty]``. Connector node features
        are ``[connector_type]``. Property node features are
        ``[property_type, owner_id, ignored, shield, property_index]``, where
        ``owner_id == -1`` means no meeple owner. Edge relation types carry
        graph semantics; there are currently no per-edge feature values.

    Randomness:
        ``reset(seed=...)`` uses local RNG owners in the project rather than
        Gymnasium's root ``Env.np_random``. The environment rebuilds the
        ``GameEngine``, ``Deck``, and ``RandomPlayer`` opponents from the
        episode seed.
    """

    metadata = {"render_modes": []}

    AGENT_PLAYER_ID = 0
    MAX_PLAYERS = 5
    MAX_PROPERTY_INDEX = 7
    POSITION_FEATURE_SIZE = 3
    CONNECTOR_FEATURE_SIZE = 1
    PROPERTY_FEATURE_SIZE = 5
    COORD_LOW = -10_000
    COORD_HIGH = 10_000

    GRAPH_EDGE_TYPES = (
        ("position", "has_connector", "connector"),
        ("property", "touches_connector", "connector"),
        ("connector", "continues", "connector"),
        ("property", "placed_on", "position"),
        ("property", "field_city_border", "property"),
    )

    def __init__(self, seed: int = 42, n_opponents: int = 1):
        if n_opponents < 1:
            raise ValueError("At least one opponent is required.")
        if n_opponents > self.MAX_PLAYERS - 1:
            raise ValueError(f"At most {self.MAX_PLAYERS - 1} opponents are supported.")
        super().__init__()
        self.n_opponents = n_opponents
        self.n_players = n_opponents + 1
        self.seed_value = seed

        self.action_space = DynamicDiscrete()
        self.observation_space = self._make_observation_space()

        self._build_game(seed=seed)
        self.terminal = False

    def _build_game(self, seed: int):
        self.agent = PlayerState()
        opponents = [RandomPlayer(seed + i) for i in range(self.n_opponents)]
        self.game_engine = GameEngine(
            players=[self.agent, *opponents],
            seed=seed,
        )

    def _make_observation_space(self) -> spaces.Dict:
        graph_space = self._make_graph_space()
        return spaces.Dict(
            {
                # Variable-length legal action set. Candidate i is the graph after applying
                # current_turn.actions[i], and env.step(i) applies that same engine action.
                "action_candidate_graphs": spaces.Sequence(graph_space, stack=False),
                # Dense table sorted by engine player id: [player_id, score, remaining_meeples].
                # The trainable agent is always player 0 in this environment.
                "players": spaces.Box(
                    low=np.tile(np.array([0, 0, 0], dtype=np.int32), (self.n_players, 1)),
                    high=np.tile(
                        np.array([self.MAX_PLAYERS - 1, np.iinfo(np.int32).max, 7], dtype=np.int32),
                        (self.n_players, 1),
                    ),
                    dtype=np.int32,
                ),
                # Current engine turn order as player ids. This preserves who acts before the
                # agent's next decision after the chosen candidate is applied.
                "player_order": spaces.Box(
                    low=0,
                    high=self.MAX_PLAYERS - 1,
                    shape=(self.n_players,),
                    dtype=np.int32,
                ),
                # Game's deck has 73 cards, but 1 is on the table at the beginning.
                # Gymnasium Discrete(73) accepts remaining counts 0..72.
                "n_remaining_cards": spaces.Discrete(73),
            }
        )

    def _make_graph_space(self) -> HeterogeneousGraph:
        return HeterogeneousGraph(
            node_spaces={
                # [y, x, empty]. Empty frontier position nodes are kept because the engine graph
                # carries legal-placement frontier state through them.
                "position": spaces.Box(
                    low=np.array([self.COORD_LOW, self.COORD_LOW, 0], dtype=np.int32),
                    high=np.array([self.COORD_HIGH, self.COORD_HIGH, 1], dtype=np.int32),
                    shape=(self.POSITION_FEATURE_SIZE,),
                    dtype=np.int32,
                ),
                # [connector_type], where connector_type is ConnectorType's integer value.
                "connector": spaces.Box(
                    low=0,
                    high=max(int(connector) for connector in ConnectorType),
                    shape=(self.CONNECTOR_FEATURE_SIZE,),
                    dtype=np.int32,
                ),
                # [property_type, owner_id, ignored, shield, property_index].
                # owner_id == -1 means no meeple owner.
                "property": spaces.Box(
                    low=np.array([-1, -1, 0, 0, 0], dtype=np.int32),
                    high=np.array(
                        [
                            int(PixelMeaning.ANY_GROWING),
                            self.MAX_PLAYERS - 1,
                            1,
                            1,
                            self.MAX_PROPERTY_INDEX,
                        ],
                        dtype=np.int32,
                    ),
                    shape=(self.PROPERTY_FEATURE_SIZE,),
                    dtype=np.int32,
                ),
            },
            # Edge type carries relation semantics; there are currently no per-edge feature values.
            edge_spaces={edge_type: None for edge_type in self.GRAPH_EDGE_TYPES},
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.seed_value = seed
        self._build_game(seed=self.seed_value)
        self.game_engine.reset(seed=seed)
        self.terminal = False
        self._advance_to_agent_turn()
        observation = self._make_observation()
        info = self._make_info()
        return observation, info

    def step(self, action: Union[int, np.integer]):
        if self.terminal:
            raise RuntimeError("Cannot call step after the environment is terminated. Call reset first.")
        turn = self._current_agent_turn()
        if turn is None:
            raise RuntimeError("No pending agent turn.")
        action_index = int(action)
        if action_index < 0 or action_index >= len(turn.actions):
            raise ValueError(f"Invalid action index {action_index}; current legal action count is {len(turn.actions)}.")

        previous_agent_score = self.agent.scores
        self.game_engine.apply_turn_action_by_index(action_index)
        self._advance_to_agent_turn()
        self.terminal = self.game_engine.terminal

        observation = self._make_observation()
        reward = float(self.agent.scores - previous_agent_score)
        truncated = False
        return observation, reward, self.terminal, truncated, self._make_info()

    def _advance_to_agent_turn(self):
        self.game_engine.advance_until_player_turn(self.agent.id, autoplay=True)

    def _current_agent_turn(self):
        if self.game_engine is None:
            raise ValueError("Game is not created yet.")
        turn = self.game_engine.current_turn
        if turn is None or self.agent is None or turn.player.id != self.agent.id:
            return None
        return turn

    def _make_info(self) -> Dict[str, Any]:
        turn = self._current_agent_turn()
        return {
            "seed": self.seed_value,
            "agent_player_id": self.agent.id,
            "legal_action_count": 0 if turn is None else len(turn.actions),
            "scores": {player.id: player.scores for player in self.game_engine.id2player.values()},
            "remaining_meeples": {
                player.id: player.remaining_meeples for player in self.game_engine.id2player.values()
            },
        }

    def _make_observation(self) -> Dict[str, Any]:
        snapshot = self.game_engine.get_state_snapshot()
        turn = self._current_agent_turn()
        if turn is None:
            action_candidate_graphs = ()
        else:
            action_candidate_graphs = tuple(
                self._encode_graph_snapshot(graph_snapshot)
                for graph_snapshot in self.game_engine.board.get_action_candidate_graph_snapshots(
                    turn.card,
                    turn.actions,
                    player_id=self.agent.id,
                )
            )

        return {
            "action_candidate_graphs": action_candidate_graphs,
            "players": self._encode_players(),
            "player_order": np.array(snapshot.player_order, dtype=np.int32),
            "n_remaining_cards": int(snapshot.deck_remaining),
        }

    def _encode_players(self) -> np.ndarray:
        return np.array(
            [
                [player.id, player.scores, player.remaining_meeples]
                for player in sorted(self.game_engine.id2player.values(), key=lambda player: player.id)
            ],
            dtype=np.int32,
        )

    def _encode_graph_snapshot(self, graph_snapshot: Dict[str, List]) -> HeterogeneousGraphInstance:
        nodes: Dict[str, List[np.ndarray]] = {
            "position": [],
            "connector": [],
            "property": [],
        }
        node_indices: Dict[Hashable, Tuple[str, int]] = {}

        for node_name, node_data in graph_snapshot["nodes"]:
            node_type = self._node_type(node_name)
            feature = self._encode_node(node_name, node_data)
            node_indices[node_name] = (node_type, len(nodes[node_type]))
            nodes[node_type].append(feature)

        edge_links = {edge_type: [] for edge_type in self.GRAPH_EDGE_TYPES}
        for source, target, edge_data in graph_snapshot["edges"]:
            edge_type, link = self._encode_edge(source, target, edge_data, node_indices)
            edge_links[edge_type].append(link)

        return HeterogeneousGraphInstance(
            nodes={
                node_type: self._feature_array(features, self._node_feature_size(node_type))
                for node_type, features in nodes.items()
            },
            edge_links={
                edge_type: np.array(links, dtype=np.int64).reshape((len(links), 2))
                for edge_type, links in edge_links.items()
            },
            edges=None,
        )

    @staticmethod
    def _node_type(node_name: Hashable) -> str:
        if not isinstance(node_name, tuple) or len(node_name) < 1:
            raise ValueError(f"Unexpected graph node name: {node_name!r}")
        node_type = node_name[0]
        if node_type not in {"position", "connector", "property"}:
            raise ValueError(f"Unexpected graph node type: {node_type!r}")
        return node_type

    def _node_feature_size(self, node_type: str) -> int:
        return {
            "position": self.POSITION_FEATURE_SIZE,
            "connector": self.CONNECTOR_FEATURE_SIZE,
            "property": self.PROPERTY_FEATURE_SIZE,
        }[node_type]

    @staticmethod
    def _feature_array(features: List[np.ndarray], feature_size: int) -> np.ndarray:
        if not features:
            return np.empty((0, feature_size), dtype=np.int32)
        return np.stack(features).astype(np.int32, copy=False)

    def _encode_node(self, node_name: Hashable, node_data: Dict[str, Any]) -> np.ndarray:
        node_type = self._node_type(node_name)
        if node_type == "position":
            y, x = node_data["position"]
            return np.array([y, x, int(bool(node_data.get("empty", False)))], dtype=np.int32)
        if node_type == "connector":
            return np.array([int(node_data["connector"])], dtype=np.int32)
        if node_type == "property":
            property_index = node_name[3]
            owner = node_data.get("owner")
            property_type = node_data.get("property")
            return np.array(
                [
                    -1 if property_type is None else int(property_type),
                    -1 if owner is None else int(owner),
                    int(bool(node_data.get("ignore", False))),
                    int(bool(node_data.get("shield", False))),
                    int(property_index),
                ],
                dtype=np.int32,
            )
        raise ValueError(f"Unexpected graph node type: {node_type!r}")

    def _encode_edge(
        self,
        source: Hashable,
        target: Hashable,
        edge_data: Dict[str, Any],
        node_indices: Dict[Hashable, Tuple[str, int]],
    ) -> Tuple[Tuple[str, str, str], Tuple[int, int]]:
        source_type, source_index = node_indices[source]
        target_type, target_index = node_indices[target]

        if {source_type, target_type} == {"position", "connector"}:
            return self._oriented_edge(
                "position",
                "has_connector",
                "connector",
                source_type,
                source_index,
                target_type,
                target_index,
            )
        if {source_type, target_type} == {"property", "connector"}:
            return self._oriented_edge(
                "property",
                "touches_connector",
                "connector",
                source_type,
                source_index,
                target_type,
                target_index,
            )
        if source_type == "connector" and target_type == "connector":
            return ("connector", "continues", "connector"), (source_index, target_index)
        if {source_type, target_type} == {"property", "position"}:
            return self._oriented_edge(
                "property",
                "placed_on",
                "position",
                source_type,
                source_index,
                target_type,
                target_index,
            )
        if source_type == "property" and target_type == "property":
            if edge_data.get("relation") != "field_city_border":
                raise ValueError(f"Unexpected property-property edge data: {edge_data!r}")
            return ("property", "field_city_border", "property"), (source_index, target_index)
        raise ValueError(f"Unexpected graph edge: {source!r} -- {target!r} with {edge_data!r}")

    @staticmethod
    def _oriented_edge(
        expected_source_type: str,
        relation_type: str,
        expected_target_type: str,
        actual_source_type: str,
        actual_source_index: int,
        actual_target_type: str,
        actual_target_index: int,
    ) -> Tuple[Tuple[str, str, str], Tuple[int, int]]:
        edge_type = (expected_source_type, relation_type, expected_target_type)
        if actual_source_type == expected_source_type and actual_target_type == expected_target_type:
            return edge_type, (actual_source_index, actual_target_index)
        return edge_type, (actual_target_index, actual_source_index)
