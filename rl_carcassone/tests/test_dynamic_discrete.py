import numpy as np
import pytest

from rl_carcassone.rl_carcassone.env.spaces import DynamicDiscrete


def test_dynamic_discrete_contains_non_negative_integer_by_default():
    action_space = DynamicDiscrete()

    assert 0 in action_space
    assert 123 in action_space
    assert np.int64(3) in action_space
    assert -1 not in action_space
    assert 1.5 not in action_space
    assert True not in action_space


def test_dynamic_discrete_does_not_sample_without_known_action_count():
    action_space = DynamicDiscrete()

    with pytest.raises(ValueError, match="known action count"):
        action_space.sample()
