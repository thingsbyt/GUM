import json
from pathlib import Path
import pytest

from gum.harness import GUMHarness
from gum.lineage import HashLedger
from gum.world_creator import WorldCreator, load_world
from gum.builtin_worlds import create_builtin_world, load_asteroids, load_immune
from gum.mind import observation_id
from gum.specialist_minds import CooperativeTeamMind


def test_world_package_separates_public_contract_from_private_genome(tmp_path: Path):
    folder = WorldCreator(tmp_path).create(seed=101, difficulty=1)
    public = json.loads((folder / "world.json").read_text())
    private = json.loads((folder / "genome.private.json").read_text())
    assert public["action_kind"] == "discrete-anonymous"
    assert "action_map" not in public and "goal" not in public and "hazards" not in public
    assert "action_map" in private and "goal" in private and "hazards" in private
    world = load_world(folder); assert world.reset(3).shape == tuple(public["observation_shape"])


def test_hash_ledger_detects_tampering(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"; ledger = HashLedger(path)
    ledger.append("one", {"value": 1}); ledger.append("two", {"value": 2})
    assert ledger.verify()["valid"]
    rows = path.read_text().splitlines(); rows[0] = rows[0].replace('"value": 1', '"value": 9')
    path.write_text("\n".join(rows) + "\n")
    assert not ledger.verify()["valid"]


def test_hash_ledger_anchor_detects_deleted_suffix(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"; ledger = HashLedger(path)
    ledger.append("one", {"value": 1}); ledger.append("two", {"value": 2})
    assert ledger.verify()["valid"] and ledger.verify()["anchored"]
    path.write_text(path.read_text().splitlines()[0] + "\n", encoding="utf-8")
    result = ledger.verify()
    assert not result["valid"]
    assert "record count differs from anchor" in result["errors"]
    reopened = HashLedger(path)
    try: reopened.append("three", {"value": 3})
    except RuntimeError as error: assert "disagrees with its anchor" in str(error)
    else: raise AssertionError("truncated ledger was extended and re-anchored")


def test_atomic_brain_save_keeps_previous_backup(tmp_path: Path):
    from gum.mind import PixelQLearner
    path = tmp_path / "mind.json"; mind = PixelQLearner()
    mind.episodes = 1; mind.save(path)
    first = path.read_bytes(); mind.episodes = 2; mind.save(path)
    assert json.loads(path.read_text())["episodes"] == 2
    assert path.with_name("mind.json.bak").read_bytes() == first


def test_harness_persists_mind_and_conversation(tmp_path: Path):
    harness = GUMHarness(tmp_path); folder = harness.create_world(seed=202, difficulty=0)
    assert folder.exists(); assert "verified records" in harness.communicate("what do you know?")
    assert "gum-grid-v1" in harness.status()["registered_adapters"]
    assert harness.ledger.verify()["valid"]


def test_short_practice_creates_a_complete_transition_audit(tmp_path: Path):
    harness = GUMHarness(tmp_path); harness.create_world(seed=303, difficulty=0)
    report = harness.practice(training_episodes=30, evaluation_episodes=10, seed=404)
    assert report["transition_trace"]["valid"]
    assert report["mind_after"]["learned_states"] > 0
    assert Path(report["transition_trace"]["path"]).exists()


def test_multi_agent_observations_have_stable_fingerprints():
    import numpy as np
    views = [np.zeros((3, 4, 5), dtype=np.uint8), np.ones((3, 4, 5), dtype=np.uint8)]
    assert observation_id(views) == observation_id([views[0].copy(), views[1].copy()])
    changed = [views[0].copy(), views[1].copy()]; changed[1][0, 0, 0] = 2
    assert observation_id(views) != observation_id(changed)


def test_builtin_worlds_keep_hidden_state_out_of_public_contract(tmp_path: Path):
    asteroids = create_builtin_world(tmp_path, "asteroids", seed=12, max_steps=4)
    public = json.loads((asteroids / "world.json").read_text())
    private = json.loads((asteroids / "genome.private.json").read_text())
    assert "seed" not in public and "seed" in private
    assert len(load_asteroids(asteroids).reset()) == 2
    immune = create_builtin_world(tmp_path, "immune", seed=13, max_steps=4)
    public = json.loads((immune / "world.json").read_text())
    assert "correct_pair" not in public and len(load_immune(immune).reset()) == 2


def test_cooperative_mind_snapshot_round_trip_and_checksum(tmp_path: Path):
    path = tmp_path / "immune.json"; mind = CooperativeTeamMind("immune-savior")
    mind.episodes = 3; mind.steps = 17; mind.save(path)
    try: CooperativeTeamMind.load(path)
    except ValueError as error: assert "trusted=True" in str(error)
    else: raise AssertionError("pickle state loaded without explicit trust")
    restored = CooperativeTeamMind.load(path, trusted=True)
    assert restored.episodes == 3 and restored.steps == 17
    state = path.with_suffix(".state.pkl"); state.write_bytes(state.read_bytes() + b"tamper")
    try: CooperativeTeamMind.load(path, trusted=True)
    except ValueError as error: assert "checksum" in str(error)
    else: raise AssertionError("tampered specialist state was accepted")


@pytest.mark.neural
def test_grounded_language_memory_attaches_to_same_interface(tmp_path: Path):
    from jepa_asteroids.grounded_dialogue import GroundedLanguageBridge, teach_default_language
    model = GroundedLanguageBridge(); teach_default_language(model); source = tmp_path / "source_language.json"; model.save(source)
    harness = GUMHarness(tmp_path / "gum"); result = harness.attach_language_memory(source)
    assert result["learned_words"] > 10
    reply = harness.communicate("locate the apple then move to the clock while stay away from the lamp")
    assert "find apple" in reply and "approach clock" in reply and "avoid lamp" in reply
