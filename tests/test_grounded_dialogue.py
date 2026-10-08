from pathlib import Path

from jepa_asteroids.grounded_dialogue import (
    GroundedGridWorld, GroundedLanguageBridge, GroundedTask, teach_default_language,
)


def learned_bridge():
    model = GroundedLanguageBridge(); teach_default_language(model); return model


def test_words_are_learned_from_demonstrations_not_installed():
    model = learned_bridge(); lexicon = model.lexicon()
    assert lexicon["locate"]["meaning"] == "action:find"
    assert lexicon["apple"]["meaning"] == "object:apple"
    assert lexicon["the"]["meaning"] == "meaning:null"


def test_never_seen_sentence_composes_goal_and_constraint():
    result = learned_bridge().interpret(
        "please locate the apple then move to the clock while staying away from the lamp")
    assert result.status == "understood"
    assert result.task == GroundedTask((("find", "apple"), ("approach", "clock")), ("lamp",))
    assert GroundedGridWorld(42).execute(result.task)["success"]


def test_unknown_incomplete_and_conflicting_requests_trigger_clarification():
    model = learned_bridge()
    assert model.interpret("find the bank").status == "clarify"
    assert model.interpret("avoid the lamp").status == "clarify"
    assert model.interpret("approach apple while avoid apple").status == "clarify"


def test_language_memory_round_trip(tmp_path: Path):
    model = learned_bridge(); path = tmp_path / "language.json"; model.save(path)
    loaded = GroundedLanguageBridge.load(path)
    assert loaded.interpret("locate apple then move to clock while stay away from lamp").task == GroundedTask(
        (("find", "apple"), ("approach", "clock")), ("lamp",))
