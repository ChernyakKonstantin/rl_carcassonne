import numpy as np
from gymnasium import spaces

from rl_carcassone.env.spaces import HeterogeneousGraph, HeterogeneousGraphInstance


def test_heterogeneous_graph_contains_typed_nodes_and_edges():
    graph_space = HeterogeneousGraph(
        node_spaces={
            "position": spaces.Box(low=-256, high=256, shape=(3,), dtype=np.int32),
            "connector": spaces.Box(low=0, high=11, shape=(1,), dtype=np.int32),
            "property": spaces.Box(low=-1, high=5, shape=(5,), dtype=np.int32),
        },
        edge_spaces={
            ("position", "has_connector", "connector"): None,
            ("property", "touches_connector", "connector"): spaces.Discrete(3),
        },
    )
    graph = HeterogeneousGraphInstance(
        nodes={
            "position": np.array([[0, 0, 0], [1, 0, 1]], dtype=np.int32),
            "connector": np.array([[0], [1]], dtype=np.int32),
            "property": np.array([[0, 1, -1, 0, 0]], dtype=np.int32),
        },
        edge_links={
            ("position", "has_connector", "connector"): np.array([[0, 0], [1, 1]], dtype=np.int64),
            ("property", "touches_connector", "connector"): np.array([[0, 1]], dtype=np.int64),
        },
        edges={
            ("property", "touches_connector", "connector"): np.array([2], dtype=np.int64),
        },
    )

    assert graph in graph_space


def test_heterogeneous_graph_rejects_unknown_node_type():
    graph_space = HeterogeneousGraph(
        node_spaces={"position": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int32)},
    )
    graph = HeterogeneousGraphInstance(
        nodes={
            "position": np.array([[0]], dtype=np.int32),
            "unknown": np.array([[0]], dtype=np.int32),
        },
    )

    assert graph not in graph_space


def test_heterogeneous_graph_rejects_out_of_range_edge_index():
    edge_type = ("position", "has_connector", "connector")
    graph_space = HeterogeneousGraph(
        node_spaces={
            "position": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int32),
            "connector": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int32),
        },
        edge_spaces={edge_type: None},
    )
    graph = HeterogeneousGraphInstance(
        nodes={
            "position": np.array([[0]], dtype=np.int32),
            "connector": np.array([[0]], dtype=np.int32),
        },
        edge_links={edge_type: np.array([[0, 1]], dtype=np.int64)},
    )

    assert graph not in graph_space


def test_heterogeneous_graph_accepts_non_string_hashable_types():
    position_type = ("node", 0)
    connector_type = 1
    edge_type = (position_type, ("relation", "has_connector"), connector_type)
    graph_space = HeterogeneousGraph(
        node_spaces={
            position_type: spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int32),
            connector_type: spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int32),
        },
        edge_spaces={edge_type: None},
    )
    graph = HeterogeneousGraphInstance(
        nodes={
            position_type: np.array([[0]], dtype=np.int32),
            connector_type: np.array([[0]], dtype=np.int32),
        },
        edge_links={edge_type: np.array([[0, 0]], dtype=np.int64)},
    )

    assert graph in graph_space


def test_heterogeneous_graph_sample_returns_empty_graph_inside_space():
    graph_space = HeterogeneousGraph(
        node_spaces={
            "position": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "connector": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        },
        edge_spaces={("position", "has_connector", "connector"): spaces.Discrete(4)},
    )

    sample = graph_space.sample()

    assert sample in graph_space
    assert sample.nodes["position"].shape == (0, 1)
    assert sample.nodes["connector"].shape == (0, 2)
    assert sample.edge_links[("position", "has_connector", "connector")].shape == (0, 2)
