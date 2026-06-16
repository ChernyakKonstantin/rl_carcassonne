from rl_carcassone.data import Episode, load_episode, load_episodes, save_episode_atomic


def test_save_episode_atomic_roundtrip(tmp_path):
    episode = Episode(
        states=[],
        actions=[1, 2],
        logprobs=[-0.1, -0.2],
        rewards=[1.0, 0.0],
        infos=[{"step": 0}, {"step": 1}],
    )
    path = tmp_path.joinpath("episode.pt")

    saved_path = save_episode_atomic(
        episode=episode,
        path=path,
        metadata={"policy_version": 3, "split": "train"},
    )
    loaded_episode = load_episode(saved_path)
    loaded_episodes = load_episodes([saved_path])

    assert saved_path == path
    assert not path.with_suffix(".pt.tmp").exists()
    assert loaded_episode.actions == [1, 2]
    assert loaded_episode.rewards == [1.0, 0.0]
    assert len(loaded_episodes) == 1
