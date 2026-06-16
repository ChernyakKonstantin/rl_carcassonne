from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch import Tensor
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import HGTConv, global_mean_pool

from rl_carcassone.env.spaces import EdgeType, HeterogeneousGraph, HeterogeneousGraphInstance


class CarcassonneGraphFeatureExtractor(nn.Module):
    """Encode action-candidate heterogeneous graph observations.

    The environment exposes one graph per currently legal action. For actors,
    this extractor returns one embedding per candidate action. For critics, it
    can pool those candidate embeddings into a single state embedding.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        pool_candidates: bool = False,
        use_reverse_edges: bool = True,
        include_position_coordinates: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool_candidates = pool_candidates
        self.use_reverse_edges = use_reverse_edges
        self.include_position_coordinates = include_position_coordinates

        graph_space = self._graph_space(observation_space)
        self.base_edge_types = tuple(graph_space.edge_spaces)
        self.edge_types = self._edge_types_with_reverse(self.base_edge_types)
        self.node_types = tuple(graph_space.node_spaces)

        self.node_encoders = nn.ModuleDict(
            {
                "position": PositionNodeEncoder(
                    hidden_dim=hidden_dim,
                    include_coordinates=include_position_coordinates,
                ),
                "connector": ConnectorNodeEncoder(
                    hidden_dim=hidden_dim,
                    n_connector_types=self._n_categories(graph_space.node_spaces["connector"]),
                ),
                "property": PropertyNodeEncoder(
                    hidden_dim=hidden_dim,
                    n_property_types=self._n_categories(graph_space.node_spaces["property"], column=0, has_none=True),
                    n_owner_ids=self._n_categories(graph_space.node_spaces["property"], column=1, has_none=True),
                ),
            }
        )
        metadata = (list(self.node_types), list(self.edge_types))
        self.convs = nn.ModuleList(
            [
                HGTConv(
                    in_channels={node_type: hidden_dim for node_type in self.node_types},
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=heads,
                )
                for _ in range(num_layers)
            ]
        )

        self.context_dim = self._context_dim(observation_space)
        self.context_net = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.out_dim = hidden_dim * 2

    @staticmethod
    def _graph_space(observation_space: spaces.Dict) -> HeterogeneousGraph:
        graph_sequence_space = observation_space["action_candidate_graphs"]
        graph_space = graph_sequence_space.feature_space
        if not isinstance(graph_space, HeterogeneousGraph):
            raise TypeError("action_candidate_graphs must be a Sequence(HeterogeneousGraph)")
        return graph_space

    @staticmethod
    def _context_dim(observation_space: spaces.Dict) -> int:
        players_shape = observation_space["players"].shape
        player_order_shape = observation_space["player_order"].shape
        return int(np.prod(players_shape) + np.prod(player_order_shape) + 1)

    @staticmethod
    def _n_categories(space: spaces.Box, column: int = 0, has_none: bool = False) -> int:
        max_value = int(np.asarray(space.high).reshape(-1)[column])
        return max_value + 1 + int(has_none)

    def _edge_types_with_reverse(self, base_edge_types: Sequence[EdgeType]) -> tuple[EdgeType, ...]:
        edge_types = []
        for source_type, relation_type, target_type in base_edge_types:
            edge_types.append((source_type, relation_type, target_type))
            if self.use_reverse_edges:
                edge_types.append((target_type, f"rev_{relation_type}", source_type))
        return tuple(edge_types)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, observation: Mapping[str, Any]) -> Tensor:
        candidate_batch = observation.get("candidate_graphs")
        if candidate_batch is None and len(observation["action_candidate_graphs"]) == 0:
            candidate_embeddings = torch.empty((0, self.hidden_dim), device=self.device)
        else:
            if candidate_batch is None:
                candidate_batch = self.candidate_graphs_to_batch(
                    observation["action_candidate_graphs"],
                    device=self.device,
                )
            candidate_embeddings = self.encode_candidate_batch(candidate_batch)
        context = self.encode_context(observation)

        if self.pool_candidates:
            if candidate_embeddings.numel() == 0:
                graph_embedding = torch.zeros(self.hidden_dim, device=self.device)
            else:
                graph_embedding = candidate_embeddings.mean(dim=0)
            return torch.cat([graph_embedding, context], dim=-1)

        if candidate_embeddings.numel() == 0:
            return candidate_embeddings.new_empty((0, self.out_dim))
        context = context.expand(candidate_embeddings.shape[0], -1)
        return torch.cat([candidate_embeddings, context], dim=-1)

    def encode_candidate_batch(self, batch: Batch) -> Tensor:
        num_graphs = batch.num_graphs
        if num_graphs == 0:
            return torch.empty((0, self.hidden_dim), device=self.device)

        x_dict = {}
        for node_type in self.node_types:
            x = batch[node_type].x.to(self.device, dtype=torch.float32)
            x_dict[node_type] = self.node_encoders[str(node_type)](x)

        edge_index_dict = {
            edge_type: edge_index.to(self.device) for edge_type, edge_index in batch.edge_index_dict.items()
        }
        for conv in self.convs:
            next_x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                node_type: torch.relu(next_x_dict.get(node_type, x_dict[node_type])) for node_type in self.node_types
            }

        pooled_by_type = [
            self._pool_node_type(x_dict[node_type], batch[node_type].batch.to(self.device), num_graphs)
            for node_type in self.node_types
        ]
        return torch.stack(pooled_by_type, dim=0).mean(dim=0)

    def _pool_node_type(self, x: Tensor, graph_index: Tensor, num_graphs: int) -> Tensor:
        if x.shape[0] == 0:
            return torch.zeros((num_graphs, self.hidden_dim), device=self.device)
        return global_mean_pool(x, graph_index, size=num_graphs)

    def encode_context(self, observation: Mapping[str, Any]) -> Tensor:
        players = self._float_tensor(observation["players"])
        player_order = self._float_tensor(observation["player_order"])
        n_remaining_cards = self._float_tensor(observation["n_remaining_cards"]).reshape(1)

        players = players.clone()
        players[..., 0] = players[..., 0] / 4.0
        players[..., 1] = players[..., 1] / 100.0
        players[..., 2] = players[..., 2] / 7.0
        player_order = player_order / 4.0
        n_remaining_cards = n_remaining_cards / 72.0

        context = torch.cat([players.reshape(-1), player_order.reshape(-1), n_remaining_cards], dim=0)
        return self.context_net(context)

    def _float_tensor(self, value: Any) -> Tensor:
        if isinstance(value, Tensor):
            return value.to(self.device, dtype=torch.float32)
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def candidate_graphs_to_batch(
        self,
        candidate_graphs: Sequence[HeterogeneousGraphInstance | HeteroData],
        device: Optional[torch.device] = None,
    ) -> Batch:
        data_list = [
            graph if isinstance(graph, HeteroData) else self.graph_to_heterodata(graph, device=device)
            for graph in candidate_graphs
        ]
        if not data_list:
            raise ValueError("Cannot create a PyG batch from an empty candidate graph sequence.")
        return Batch.from_data_list(data_list)

    def graph_to_heterodata(
        self,
        graph: HeterogeneousGraphInstance,
        device: Optional[torch.device] = None,
    ) -> HeteroData:
        data = HeteroData()
        for node_type in self.node_types:
            values = graph.nodes[node_type]
            data[node_type].x = torch.as_tensor(values, device=device, dtype=torch.float32)

        for source_type, relation_type, target_type in self.base_edge_types:
            links = graph.edge_links[(source_type, relation_type, target_type)]
            edge_index = torch.as_tensor(links, device=device, dtype=torch.long).t().contiguous()
            data[(source_type, relation_type, target_type)].edge_index = edge_index
            if self.use_reverse_edges:
                data[(target_type, f"rev_{relation_type}", source_type)].edge_index = edge_index.flip(0)
        return data


class PositionNodeEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        include_coordinates: bool = False,
        coordinate_scale: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.include_coordinates = include_coordinates
        self.coordinate_scale = coordinate_scale
        input_dim = 3 if include_coordinates else 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.include_coordinates:
            yx = x[:, :2] / self.coordinate_scale
            empty = x[:, 2:3]
            features = torch.cat([yx, empty], dim=-1)
        else:
            features = x[:, 2:3]
        return self.net(features)


class ConnectorNodeEncoder(nn.Module):
    def __init__(self, hidden_dim: int, n_connector_types: int) -> None:
        super().__init__()
        self.connector_embedding = nn.Embedding(n_connector_types, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        connector_type = x[:, 0].long().clamp_min(0)
        return self.connector_embedding(connector_type)


class PropertyNodeEncoder(nn.Module):
    PROPERTY_TYPE_COLUMN = 0
    OWNER_ID_COLUMN = 1
    IGNORED_COLUMN = 2
    SHIELD_COLUMN = 3

    def __init__(
        self,
        hidden_dim: int,
        n_property_types: int,
        n_owner_ids: int,
    ) -> None:
        super().__init__()
        self.property_type_embedding = nn.Embedding(n_property_types, hidden_dim)
        self.owner_embedding = nn.Embedding(n_owner_ids, hidden_dim)
        self.flag_encoder = nn.Linear(2, hidden_dim)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        property_type = self._shift_optional_category(x[:, self.PROPERTY_TYPE_COLUMN])
        owner_id = self._shift_optional_category(x[:, self.OWNER_ID_COLUMN])
        flags = x[:, [self.IGNORED_COLUMN, self.SHIELD_COLUMN]]

        features = (
            self.property_type_embedding(property_type) + self.owner_embedding(owner_id) + self.flag_encoder(flags)
        )
        return self.output(features)

    @staticmethod
    def _shift_optional_category(values: Tensor) -> Tensor:
        return (values.long() + 1).clamp_min(0)
