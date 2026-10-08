"""Strict GUM Growth Challenge with distinct-skill composition."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .concept_skill_factory import ACTION_COUNT, ConceptSkillFactory, MechanismWorld, frame_id
from .lineage import TransitionTrace, file_sha256


FORMAT = "gum-growth-challenge-v2"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _targets(seed: int) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed); values = []
    while len(values) < 4:
        row = tuple(map(int, rng.choice(ACTION_COUNT, 3, replace=False)))
        if row not in values: values.append(row)
    return values


def _trace_callback(trace: TransitionTrace, run_id: str, phase: str):
    counter = {"value": 0}
    def append(before, action, after, reward, done):
        trace.append("transition", {"phase": phase, "index": counter["value"],
            "observation": frame_id(before), "action": int(action),
            "next_observation": frame_id(after), "reward": float(reward),
            "terminated": bool(done), "learner_visible": True}, run_id=run_id)
        counter["value"] += 1
    return append


def _render_replay(world: MechanismWorld, actions: list[int], output: Path) -> None:
    from PIL import Image, ImageDraw
    frames = []; observation = world.reset()
    sequence = [(observation, "start", False)]
    for index, action in enumerate(actions, 1):
        observation, reward, done, info = world.step(action)
        sequence.append((observation, f"step {index} · anonymous action {action}", bool(info["success"])))
    for frame, label, success in sequence:
        image = Image.fromarray(np.moveaxis(frame, 0, -1)).resize((720, 264), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (720, 340), (7, 19, 23)); canvas.paste(image, (0, 0)); draw = ImageDraw.Draw(canvas)
        draw.text((18, 279), "GUM GROWTH CHALLENGE · HELD-OUT FIFTH WORLD", fill=(110, 231, 183))
        draw.text((18, 307), label + (" · SUCCESS" if success else ""), fill=(238, 244, 247))
        frames.append(canvas)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=650, loop=0)


def protocol(source_root: Path, output: Path) -> dict:
    source_root = Path(source_root); output = Path(output)
    files = [source_root / "gum" / "concept_skill_factory.py", source_root / "gum" / "growth_challenge_v2.py"]
    value = {
        "format": FORMAT,
        "classification": "precommitted internal test of autonomous opaque concepts and distinct-skill composition",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_seed_family": 61_000_000,
        "sealed_seed_family": 91_000_000,
        "curriculum": {
            "initial_surface_families": 4,
            "held_out_variants_per_family": 3,
            "fifth_world_created_from_post-curriculum_brain_hash": True,
            "fifth_world_program_length": 6,
            "fifth_world_requires_distinct_parent_skills": True,
        },
        "learner_inputs": ["RGB pixels", "five anonymous actions", "terminal scalar reward", "termination"],
        "learner_not_given": ["human concept names", "hidden primitive identities", "control maps",
                              "target programs", "surface permutations", "family identity", "solution sequences"],
        "thresholds": {
            "initial_world_successes": 4,
            "held_out_variant_successes": 12,
            "retention_successes_after_reload": 4,
            "portable_skill_recipient_success": True,
            "minimum_fifth_world_speedup_over_fresh": 10.0,
            "fifth_world_distinct_parent_skills": True,
            "maximum_created_concepts": 5,
            "maximum_mapping_branches": 0,
            "source_must_remain_unchanged": True,
            "traces_must_verify": True,
        },
        "source_hashes": {str(path.relative_to(source_root)).replace("\\", "/"): file_sha256(path) for path in files},
    }
    _write_json(output, value); return value


def run(source_root: Path, output_root: Path) -> dict:
    source_root = Path(source_root); output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "GROWTH_CHALLENGE_PROTOCOL.json"
    frozen = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else protocol(source_root, protocol_path)
    before = {name: file_sha256(source_root / Path(name)) for name in frozen["source_hashes"]}
    if before != frozen["source_hashes"]: raise RuntimeError("implementation differs from precommitted source hashes")

    run_id = "growth-v2-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    targets = _targets(frozen["development_seed_family"]); mind = ConceptSkillFactory(); acquisition = []; worlds = []
    training_trace = TransitionTrace(output_root / "TRAINING_TRANSITIONS.jsonl")
    for index, target in enumerate(targets):
        world = MechanismWorld(frozen["development_seed_family"] + 1009 * index, target, index)
        worlds.append(world); acquisition.append(mind.acquire(world,
            trace=_trace_callback(training_trace, run_id, f"acquire-{index}")))

    brain_path = output_root / "GROWN_MIND.json"; mind.save(brain_path)
    post_curriculum_hash = file_sha256(brain_path)

    held_out = []
    for family, target in enumerate(targets):
        for variant in range(3):
            world = MechanismWorld(frozen["sealed_seed_family"] + family * 10_000 + variant * 101, target, family + variant + 4)
            result = mind.solve_with_skills(world)
            held_out.append({"family_audit_only": family, "variant": variant, **result, "world_audit": world.audit()})

    mind.save(brain_path); loaded = ConceptSkillFactory.load(brain_path)
    retention = [loaded.revisit(world) for world in worlds]

    selector = int(post_curriculum_hash[:8], 16)
    left = selector % len(targets)
    right = (selector // len(targets)) % (len(targets) - 1)
    if right >= left:
        right += 1
    fifth_target = targets[left] + targets[right]
    fifth_seed = frozen["sealed_seed_family"] + int(post_curriculum_hash[8:16], 16) % 1_000_000
    experienced_world = MechanismWorld(fifth_seed, fifth_target, 11)
    experienced_trace = TransitionTrace(output_root / "FIFTH_EXPERIENCED_TRANSITIONS.jsonl")
    experienced = loaded.solve_composed(experienced_world,
        trace=_trace_callback(experienced_trace, run_id, "fifth-experienced"))

    fresh = ConceptSkillFactory(); fresh_world = MechanismWorld(fifth_seed, fifth_target, 11)
    fresh_trace = TransitionTrace(output_root / "FIFTH_FRESH_TRANSITIONS.jsonl")
    fresh_result = fresh.acquire(fresh_world,
        trace=_trace_callback(fresh_trace, run_id, "fifth-fresh"))

    recipient = ConceptSkillFactory(); recipient.import_portable_knowledge(loaded.share_portable_knowledge())
    share_target = targets[(right + 1) % len(targets)]
    share_world = MechanismWorld(frozen["sealed_seed_family"] + 8_080_808, share_target, 17)
    shared_result = recipient.solve_with_skills(share_world)
    recipient.save(output_root / "RECIPIENT_MIND.json")

    final_mind_path = output_root / "GROWN_MIND_FINAL.json"; loaded.save(final_mind_path)
    speedup = fresh_result["interactions"] / max(1, experienced["interactions"])
    traces = {"training": training_trace.verify(), "fifth_experienced": experienced_trace.verify(),
              "fifth_fresh": fresh_trace.verify()}
    trace_text = "\n".join(path.read_text(encoding="utf-8") for path in
        (training_trace.path, experienced_trace.path, fresh_trace.path))
    hidden_terms = [term for term in ("target_operators", "control_to_operator", "surface_permutation") if term in trace_text]
    after = {name: file_sha256(source_root / Path(name)) for name in frozen["source_hashes"]}
    status = loaded.status()
    checks = {
        "initial_worlds": sum(row["success"] for row in acquisition) == 4,
        "held_out_variants": sum(row["success"] for row in held_out) == 12,
        "retention_after_reload": sum(row["success"] for row in retention) == 4,
        "portable_recipient": bool(shared_result["success"]),
        "fifth_experienced": bool(experienced["success"]),
        "fifth_fresh": bool(fresh_result["success"]),
        "fifth_distinct_parents": left != right and len(set(experienced.get("parents", []))) == 2,
        "fifth_speedup": speedup >= frozen["thresholds"]["minimum_fifth_world_speedup_over_fresh"],
        "concept_budget": status["concepts_created"] <= frozen["thresholds"]["maximum_created_concepts"],
        "no_unnecessary_branches": status["mapping_branches"] <= frozen["thresholds"]["maximum_mapping_branches"],
        "source_frozen": before == after == frozen["source_hashes"],
        "traces_valid": all(row["valid"] for row in traces.values()),
        "hidden_state_absent_from_learner_traces": not hidden_terms,
    }
    report = {
        "format": FORMAT, "run_id": run_id, "classification": frozen["classification"],
        "protocol_sha256": file_sha256(protocol_path),
        "source_freeze": {"expected": frozen["source_hashes"], "before": before, "after": after,
                          "unchanged": before == after == frozen["source_hashes"]},
        "learner_inputs": frozen["learner_inputs"], "learner_not_given": frozen["learner_not_given"],
        "curriculum_audit_only": {"initial_targets": [list(row) for row in targets],
            "fifth_parent_indices": [left, right], "fifth_target": list(fifth_target), "fifth_seed": fifth_seed},
        "initial_acquisition": acquisition,
        "held_out_variants": {"successes": sum(row["success"] for row in held_out), "worlds": len(held_out),
                              "mean_interactions": float(np.mean([row["interactions"] for row in held_out])), "rows": held_out},
        "reload_retention": {"successes": sum(row["success"] for row in retention), "worlds": len(retention),
                             "mean_interactions": float(np.mean([row["interactions"] for row in retention])), "rows": retention},
        "fifth_world": {"experienced": experienced, "fresh": fresh_result, "speedup": speedup,
                        "created_after_curriculum_from_brain_hash": post_curriculum_hash},
        "portable_knowledge": {"recipient": shared_result, "shared_concepts": len(recipient.concepts),
                               "shared_skills": len(recipient.skills), "world_audit": share_world.audit()},
        "mind": status, "brain": {"path": str(final_mind_path), "bytes": final_mind_path.stat().st_size,
                                   "sha256": file_sha256(final_mind_path)},
        "traces": traces, "hidden_state_scan": {"clean": not hidden_terms, "matches": hidden_terms},
        "checks": checks, "passed": all(checks.values()),
        "interpretation": ("Tests self-created structural transition concepts and reusable skill composition in "
            "surface-diverse mechanistic micro-worlds. It does not establish open-world concept invention."),
    }
    _write_json(output_root / "GROWTH_CHALLENGE_AUDIT.json", report)

    context = loaded.context_memory[experienced["context"]]; mapping = {int(k): v for k, v in context["action_to_concept"].items()}
    inverse = {value: key for key, value in mapping.items()}; fifth_skill = loaded.skills[experienced["skill_id"]]
    actions = [inverse[value] for value in fifth_skill.sequence]
    _render_replay(MechanismWorld(fifth_seed, fifth_target, 11), actions, output_root / "FIFTH_WORLD_REPLAY.gif")

    lines = ["# GUM Growth Challenge v2", "", f"**Verdict: {'PASS' if report['passed'] else 'FAIL'}**", "",
        "## Result", "", f"- Autonomous opaque concepts created: **{status['concepts_created']}**",
        f"- Compiled reusable skills: **{status['skills_created']}**",
        f"- Initial worlds solved: **{sum(row['success'] for row in acquisition)}/4**",
        f"- Held-out surface variants solved: **{sum(row['success'] for row in held_out)}/12**",
        f"- Reloaded retention: **{sum(row['success'] for row in retention)}/4**",
        f"- Portable skill recipient: **{'success' if shared_result['success'] else 'failure'}**",
        f"- Fifth world composed two distinct learned skills: **{'yes' if checks['fifth_distinct_parents'] else 'no'}**",
        f"- Fifth composed world: experienced **{experienced['interactions']}** interactions; fresh **{fresh_result['interactions']}**",
        f"- Fifth-world experience advantage: **{speedup:.2f}×**", f"- Unnecessary mapping branches: **{status['mapping_branches']}**",
        "", "## Meaning", "",
        "The learner was not given concept names, primitive identities, control mappings, target programs, or family labels. It created stable identifiers from recurring visual transition structure, compiled successful concept sequences into skills, re-grounded those skills under new controls, shared them with a fresh recipient, and composed two prior skills in a fifth world generated only after the first four were complete.",
        "", "This is a controlled micro-world result. The visual-change representation and search machinery remain engineered; the test does not yet show unrestricted concept invention in arbitrary real environments."]
    (output_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run the frozen GUM Growth Challenge")
    parser.add_argument("--source", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol-only", action="store_true"); args = parser.parse_args(argv)
    path = args.output / "GROWTH_CHALLENGE_PROTOCOL.json"
    result = protocol(args.source, path) if args.protocol_only else run(args.source, args.output)
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
