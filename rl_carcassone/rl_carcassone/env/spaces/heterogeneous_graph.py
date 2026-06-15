from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
from gymnasium import spaces

NodeType = Hashable
RelationType = Hashable
EdgeType = Tuple[NodeType, RelationType, NodeType]


@dataclass(frozen=True)
class HeterogeneousGraphInstance:
    """Concrete value for ``HeterogeneousGraph`` observations.

    ``nodes`` maps a hashable node type to a batch of node feature rows.
    ``edge_links`` maps ``(source_node_type, relation_type, target_node_type)``
    to integer local node-index pairs. ``edges`` optionally stores per-edge
    features for the same edge types.
    """

    nodes: Mapping[NodeType, np.ndarray]
    edge_links: Mapping[EdgeType, np.ndarray] = field(default_factory=dict)
    edges: Optional[Mapping[EdgeType, np.ndarray]] = None


class HeterogeneousGraph(spaces.Space):
    """Gymnasium space for typed graph observations.

    Each node type has its own feature space, so graph observations do not need
    to be flattened into one wide node vector with placeholder values. Edge
    types follow the common heterogeneous-graph convention
    ``(source_node_type, relation_type, target_node_type)`` and use node indices
    local to their source and target node arrays. Node and relation types may
    be any hashable values, not only strings.
    """

    def __init__(
        self,
        node_spaces: Mapping[NodeType, spaces.Space],
        edge_spaces: Optional[Mapping[EdgeType, Optional[spaces.Space]]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(shape=None, dtype=None, seed=seed)
        self.node_spaces = dict(node_spaces)
        self.edge_spaces = {} if edge_spaces is None else dict(edge_spaces)
        self._validate_schema()

    def _validate_schema(self):
        if not self.node_spaces:
            raise ValueError("HeterogeneousGraph requires at least one node space.")
        for node_type, node_space in self.node_spaces.items():
            if not isinstance(node_type, Hashable):
                raise ValueError(f"Node type must be hashable: {node_type!r}")
            if not isinstance(node_space, spaces.Space):
                raise TypeError(f"Node space for {node_type!r} must be a Gymnasium Space.")
        for edge_type, edge_space in self.edge_spaces.items():
            self._validate_edge_type(edge_type)
            if edge_space is not None and not isinstance(edge_space, spaces.Space):
                raise TypeError(f"Edge space for {edge_type!r} must be a Gymnasium Space or None.")

    def _validate_edge_type(self, edge_type: EdgeType):
        if not isinstance(edge_type, tuple) or len(edge_type) != 3:
            raise ValueError(f"Edge type must be a (source_type, relation_type, target_type) tuple: {edge_type!r}")
        source_type, relation_type, target_type = edge_type
        if not isinstance(source_type, Hashable):
            raise ValueError(f"Source node type must be hashable in edge type {edge_type!r}.")
        if not isinstance(relation_type, Hashable):
            raise ValueError(f"Relation type must be hashable in edge type {edge_type!r}.")
        if not isinstance(target_type, Hashable):
            raise ValueError(f"Target node type must be hashable in edge type {edge_type!r}.")
        if source_type not in self.node_spaces:
            raise ValueError(f"Unknown source node type in edge type {edge_type!r}.")
        if target_type not in self.node_spaces:
            raise ValueError(f"Unknown target node type in edge type {edge_type!r}.")

    def sample(self, mask: Any = None, probability: Any = None) -> HeterogeneousGraphInstance:
        """Return an empty graph matching the declared schema."""
        if mask is not None or probability is not None:
            raise ValueError("HeterogeneousGraph does not support masked or probability-based sampling.")
        nodes = {
            node_type: np.empty((0, *node_space.shape), dtype=getattr(node_space, "dtype", np.float32))
            for node_type, node_space in self.node_spaces.items()
        }
        edge_links = {edge_type: np.empty((0, 2), dtype=np.int64) for edge_type in self.edge_spaces}
        edges = {
            edge_type: self._empty_feature_batch(edge_space)
            for edge_type, edge_space in self.edge_spaces.items()
            if edge_space is not None
        }
        return HeterogeneousGraphInstance(nodes=nodes, edge_links=edge_links, edges=edges)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, HeterogeneousGraphInstance):
            return False
        if not self._contains_nodes(x.nodes):
            return False
        if not self._contains_edge_links(x.nodes, x.edge_links):
            return False
        return self._contains_edges(x.edge_links, x.edges)

    def _contains_nodes(self, nodes: Mapping[NodeType, np.ndarray]) -> bool:
        if not isinstance(nodes, Mapping):
            return False
        unknown_types = set(nodes) - set(self.node_spaces)
        if unknown_types:
            return False
        for node_type, node_values in nodes.items():
            node_space = self.node_spaces[node_type]
            if not self._contains_feature_batch(node_space, node_values):
                return False
        return True

    def _contains_edge_links(
        self,
        nodes: Mapping[NodeType, np.ndarray],
        edge_links: Mapping[EdgeType, np.ndarray],
    ) -> bool:
        if not isinstance(edge_links, Mapping):
            return False
        unknown_types = set(edge_links) - set(self.edge_spaces)
        if unknown_types:
            return False
        node_counts = {node_type: len(node_values) for node_type, node_values in nodes.items()}
        for edge_type, links in edge_links.items():
            source_type, _, target_type = edge_type
            if not isinstance(links, np.ndarray):
                return False
            if not np.issubdtype(links.dtype, np.integer):
                return False
            if links.shape != (len(links), 2):
                return False
            source_count = node_counts.get(source_type, 0)
            target_count = node_counts.get(target_type, 0)
            if len(links) == 0:
                continue
            if np.any(links[:, 0] < 0) or np.any(links[:, 0] >= source_count):
                return False
            if np.any(links[:, 1] < 0) or np.any(links[:, 1] >= target_count):
                return False
        return True

    def _contains_edges(
        self,
        edge_links: Mapping[EdgeType, np.ndarray],
        edges: Optional[Mapping[EdgeType, np.ndarray]],
    ) -> bool:
        edges = {} if edges is None else edges
        if not isinstance(edges, Mapping):
            return False
        unknown_types = set(edges) - set(self.edge_spaces)
        if unknown_types:
            return False
        for edge_type, edge_values in edges.items():
            edge_space = self.edge_spaces[edge_type]
            if edge_space is None:
                return False
            if edge_type not in edge_links:
                return False
            if len(edge_values) != len(edge_links[edge_type]):
                return False
            if not self._contains_feature_batch(edge_space, edge_values):
                return False
        for edge_type, links in edge_links.items():
            edge_space = self.edge_spaces[edge_type]
            if edge_space is not None and edge_type not in edges and len(links) > 0:
                return False
        return True

    @staticmethod
    def _contains_feature_batch(space: spaces.Space, values: np.ndarray) -> bool:
        if not isinstance(values, np.ndarray):
            return False
        if values.shape[1:] != space.shape:
            return False
        return all(value in space for value in values)

    @staticmethod
    def _empty_feature_batch(space: spaces.Space) -> np.ndarray:
        return np.empty((0, *space.shape), dtype=getattr(space, "dtype", np.float32))

    def __repr__(self) -> str:
        node_types = ", ".join(repr(node_type) for node_type in sorted(self.node_spaces, key=repr))
        edge_types = ", ".join(repr(edge_type) for edge_type in sorted(self.edge_spaces, key=repr))
        return f"HeterogeneousGraph(node_types=[{node_types}], edge_types=[{edge_types}])"
