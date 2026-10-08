import numpy as np

from jepa_asteroids.cooperative_asteroids import (
    DuoAsteroids, DuoConfig, TwoRocketTeam, perceive, run_audit, run_episode,
)


def test_two_rockets_receive_private_rgb_views_and_hidden_controls():
    game = DuoAsteroids(DuoConfig(max_steps=20), seed=41)
    frames = game.reset(41)
    assert len(frames) == 2
    assert frames[0].shape == (180, 320, 3)
    assert frames[0].dtype == np.uint8
    assert perceive(frames[0])["self"] is not None
    assert perceive(frames[1])["self"] is not None
    assert game.semantic_by_action[0] != game.semantic_by_action[1]


def test_both_control_maps_are_grounded_from_visual_effects():
    game = DuoAsteroids(DuoConfig(max_steps=30), seed=1_440_001); team = TwoRocketTeam()
    frames = game.reset(1_440_001); team.begin(frames)
    for _ in range(10):
        actions = team.act(frames); later, reward, terminated, truncated, _ = game.step(actions)
        team.observe(frames, actions, later, reward, terminated or truncated); frames = later
    status = team.status()
    assert status["both_controls_grounded"]
    for agent in range(2):
        expected = {semantic: action for action, semantic in enumerate(game.semantic_by_action[agent])}
        assert status["controls_grounded"][agent] == expected


def test_coordinated_pair_scores_with_both_rockets_and_avoids_friendly_fire():
    result = run_episode(1_440_001, "coordinated", DuoConfig(max_steps=100))
    assert result["hits_by_agent"][0] > 0
    assert result["hits_by_agent"][1] > 0
    assert result["friendly_fire"] == 0
    assert result["learner"]["target_messages"] > 0


def test_matched_ablation_rewards_communication(tmp_path):
    report = run_audit(tmp_path / "audit.json", worlds=1, seed=1_440_001, max_steps=120)
    rows = report["aggregate"]
    assert report["controls_grounded_worlds"] == 1
    assert rows["coordinated"]["mean_hits"] > rows["uncommunicative"]["mean_hits"]
    assert rows["coordinated"]["mean_hits"] > rows["solo"]["mean_hits"]
    assert rows["coordinated"]["mean_hits"] > rows["random"]["mean_hits"]
