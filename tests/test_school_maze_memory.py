from pathlib import Path

from gum.protocol import Transition
from gum.school.maze_memory import MazeMemoryLearner
from gum.school.training import _run_episode
from gum.school.worlds import create_foundational_world, load_changing_maze


class _HostilePublicInfo(dict):
    def get(self, *_args, **_kwargs):
        raise AssertionError("the maze learner must not read public-info labels")


def _world(tmp_path: Path, package_seed: int = 31_001):
    package = create_foundational_world(
        tmp_path, "gum-changing-maze-v1", seed=package_seed, mechanism="memory"
    )
    return load_changing_maze(package)


def test_reward_gate_learns_to_activate_episodic_memory(tmp_path):
    learner = MazeMemoryLearner(23)
    rows = []
    for index, seed in enumerate((31_001, 31_002, 31_003, 31_004)):
        rows.append(_run_episode(
            learner,
            _world(tmp_path / f"world-{index}", seed),
            seed=seed,
            training=True,
            interaction_limit=240,
        ))
    assert learner.mode_visits.tolist() == [1, 3]
    assert learner.status()["selected_mode"] == "episodic-map"
    assert sum(row["success"] for row in rows) >= 3


def test_episodic_map_solves_unseen_maze_from_pixels(tmp_path):
    learner = MazeMemoryLearner(7)
    learner.mode_values[1] = 1.0
    row = _run_episode(
        learner,
        _world(tmp_path),
        seed=31_101,
        training=False,
        interaction_limit=240,
    )
    assert row["success"] is True
    assert row["interactions"] < 80
    assert row["unnecessary_actions"] / row["interactions"] < 0.25


def test_observe_ignores_semantic_public_info(tmp_path):
    world = _world(tmp_path)
    observation = world.reset(31_101)
    learner = MazeMemoryLearner(7)
    learner.mode_values[1] = 1.0
    learner.begin(world.public_spec(), observation, training=False)
    action = learner.act(observation, training=False)
    transition = world.step(action)
    hostile = Transition(
        transition.observation,
        transition.reward,
        transition.terminated,
        transition.truncated,
        _HostilePublicInfo(),
    )
    learner.observe(action, hostile, training=False)


def test_save_load_preserves_reward_gate(tmp_path):
    learner = MazeMemoryLearner(29)
    learner.mode_values[:] = (-1.2, 0.8)
    learner.mode_visits[:] = (2, 5)
    path = tmp_path / "MAZE_MEMORY.json"
    learner.save(path)
    restored = MazeMemoryLearner.load(path)
    assert restored.status()["selected_mode"] == "episodic-map"
    assert restored.mode_values.tolist() == [-1.2, 0.8]
    assert restored.mode_visits.tolist() == [2, 5]
