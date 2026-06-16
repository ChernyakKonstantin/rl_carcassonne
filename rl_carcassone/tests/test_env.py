import pytest

from rl_carcassone.env import CarcassonneEnv


def test_env_reset_returns_agent_turn_observation_inside_space():
    env = CarcassonneEnv(seed=67, n_opponents=2)

    observation, info = env.reset()

    assert info["agent_player_id"] == 0
    assert info["legal_action_count"] == len(observation["action_candidate_graphs"])
    assert info["legal_action_count"] > 0
    assert observation["players"].shape == (3, 3)
    assert observation["action_candidate_graphs"][0].nodes["position"].shape[1] == env.POSITION_FEATURE_SIZE
    assert observation["action_candidate_graphs"][0].nodes["connector"].shape[1] == env.CONNECTOR_FEATURE_SIZE
    assert observation["action_candidate_graphs"][0].nodes["property"].shape[1] == env.PROPERTY_FEATURE_SIZE
    assert ("position", "has_connector", "connector") in observation["action_candidate_graphs"][0].edge_links
    assert env.observation_space.contains(observation)


def test_env_accepts_four_opponents():
    env = CarcassonneEnv(seed=67, n_opponents=4)

    observation, info = env.reset()

    assert info["agent_player_id"] == 0
    assert observation["players"].shape == (5, 3)
    assert env.observation_space.contains(observation)


def test_env_step_uses_action_index_from_current_candidates():
    env = CarcassonneEnv(seed=67, n_opponents=2)
    observation, _ = env.reset()
    action_index = 0

    next_observation, reward, terminated, truncated, info = env.step(action_index)

    assert isinstance(reward, float)
    assert not truncated
    assert terminated is False or len(next_observation["action_candidate_graphs"]) == 0
    assert info["legal_action_count"] == len(next_observation["action_candidate_graphs"])
    assert env.observation_space.contains(next_observation)


def test_env_rejects_action_outside_current_legal_candidates():
    env = CarcassonneEnv(seed=67, n_opponents=2)
    observation, _ = env.reset()

    with pytest.raises(ValueError, match="Invalid action index"):
        env.step(len(observation["action_candidate_graphs"]))


def test_env_reset_is_deterministic_for_same_seed():
    env = CarcassonneEnv(seed=67, n_opponents=2)

    first_observation, first_info = env.reset()
    second_observation, second_info = env.reset()

    assert first_info["legal_action_count"] == second_info["legal_action_count"]
    assert (
        first_observation["action_candidate_graphs"][0].nodes["position"].tolist()
        == second_observation["action_candidate_graphs"][0].nodes["position"].tolist()
    )
    assert (
        first_observation["action_candidate_graphs"][0].nodes["property"].tolist()
        == second_observation["action_candidate_graphs"][0].nodes["property"].tolist()
    )
