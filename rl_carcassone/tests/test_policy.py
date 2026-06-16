import torch

from rl_carcassone.env import CarcassonneEnv
from rl_carcassone.env.spaces import HeterogeneousGraphInstance
from rl_carcassone.policy.actor import Actor
from rl_carcassone.policy.critic import Critic


def test_actor_outputs_one_logit_per_candidate_action():
    env = CarcassonneEnv(seed=67, n_opponents=2)
    observation, info = env.reset()
    actor = Actor(env.observation_space, env.action_space, hidden_dim=16, num_layers=1, heads=2)

    logits = actor(observation)

    assert logits.shape == (info["legal_action_count"],)
    assert torch.isfinite(logits).all()


def test_actor_samples_dynamic_discrete_action_and_evaluates_it():
    env = CarcassonneEnv(seed=67, n_opponents=2)
    observation, info = env.reset()
    actor = Actor(env.observation_space, env.action_space, hidden_dim=16, num_layers=1, heads=2)

    action, log_prob = actor.get_action(observation)
    evaluated_log_prob, entropy, _ = actor.evaluate_action(observation, action)

    assert env.action_space.contains(int(action))
    assert int(action) < info["legal_action_count"]
    assert log_prob.shape == ()
    assert evaluated_log_prob.shape == ()
    assert entropy.shape == ()
    assert torch.isfinite(log_prob)
    assert torch.isfinite(evaluated_log_prob)
    assert torch.isfinite(entropy)


def test_critic_outputs_scalar_value():
    env = CarcassonneEnv(seed=67, n_opponents=2)
    observation, _ = env.reset()
    critic = Critic(env.observation_space, hidden_dim=16, num_layers=1, heads=2)

    value = critic(observation)

    assert value.shape == ()
    assert torch.isfinite(value)


def test_critic_handles_observation_without_candidate_actions():
    env = CarcassonneEnv(seed=67, n_opponents=2)
    observation, _ = env.reset()
    observation = dict(observation)
    observation["action_candidate_graphs"] = ()
    critic = Critic(env.observation_space, hidden_dim=16, num_layers=1, heads=2)

    value = critic(observation)

    assert value.shape == ()
    assert torch.isfinite(value)


def test_actor_ignores_property_index_feature():
    env = CarcassonneEnv(seed=67, n_opponents=2)
    observation, _ = env.reset()
    actor = Actor(env.observation_space, env.action_space, hidden_dim=16, num_layers=1, heads=2)

    modified_observation = dict(observation)
    modified_graphs = []
    for graph in observation["action_candidate_graphs"]:
        nodes = dict(graph.nodes)
        property_nodes = nodes["property"].copy()
        property_nodes[:, 4] = 7 - property_nodes[:, 4]
        nodes["property"] = property_nodes
        modified_graphs.append(
            HeterogeneousGraphInstance(
                nodes=nodes,
                edge_links=graph.edge_links,
                edges=graph.edges,
            )
        )
    modified_observation["action_candidate_graphs"] = tuple(modified_graphs)

    with torch.no_grad():
        logits = actor(observation)
        modified_logits = actor(modified_observation)

    assert torch.allclose(logits, modified_logits)
