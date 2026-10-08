"""Frozen audit of self-selected visual concepts and their practical utility."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .concept_genesis import (ACTION_COUNT, ConceptGenesisMind, GenesisEncoder,
                              GenesisWorld, frame_id)
from .concept_skill_factory import structural_signature
from .lineage import TransitionTrace, file_sha256


FORMAT = "gum-concept-genesis-challenge-v1"


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _targets(seed: int):
    rng = np.random.default_rng(seed); rows = []
    while len(rows) < 4:
        row = tuple(map(int, rng.choice(ACTION_COUNT, 3, replace=False)))
        if row not in rows: rows.append(row)
    return rows


def _trace_callback(trace: TransitionTrace, run_id: str, phase: str):
    counter = {"value": 0}
    def append(before, action, after, reward, done):
        trace.append("transition", {"phase": phase, "index": counter["value"],
            "observation": frame_id(before), "action": int(action),
            "next_observation": frame_id(after), "reward": float(reward),
            "terminated": bool(done), "learner_visible": True}, run_id=run_id)
        counter["value"] += 1
    return append


def _alignment_audit(mind: ConceptGenesisMind, worlds: list[GenesisWorld]) -> dict:
    by_mechanism = {index: set() for index in range(ACTION_COUNT)}; rows = []
    for world in worlds:
        _, mapping, _ = mind.ground(world)
        for action, concept in mapping.items():
            mechanism = int(world.control_to_mechanism[action])
            by_mechanism[mechanism].add(concept)
            rows.append({"world_audit_only": world.world_id, "mechanism_audit_only": mechanism,
                         "anonymous_action": action, "learned_concept": concept})
    one_per_mechanism = all(len(value) == 1 for value in by_mechanism.values())
    distinct = len({next(iter(value)) for value in by_mechanism.values() if value}) == ACTION_COUNT
    return {"perfect_one_to_one": one_per_mechanism and distinct,
            "mechanism_to_concepts_audit_only": {str(k): sorted(v) for k, v in by_mechanism.items()},
            "rows": rows}


def _old_signature_baseline(world: GenesisWorld) -> dict:
    signatures = []
    for action in range(ACTION_COUNT):
        before = world.reset(); after, _, _, _ = world.step(action)
        signatures.append(list(structural_signature(before, after)))
    return {"anonymous_action_signatures": signatures, "distinct_signatures": len({tuple(x) for x in signatures}),
            "can_distinguish_all_actions": len({tuple(x) for x in signatures}) == ACTION_COUNT}


def _render_replay(world: GenesisWorld, actions: list[int], output: Path):
    from PIL import Image, ImageDraw, ImageFont
    frames = []; observation = world.reset(); sequence = [(observation, "start", False)]
    for index, action in enumerate(actions, 1):
        observation, _, _, info = world.step(action)
        sequence.append((observation, f"step {index} · anonymous action {action}", bool(info["success"])))
    try:
        title_font = ImageFont.truetype("arial.ttf", 20); body_font = ImageFont.truetype("arial.ttf", 16)
    except OSError: title_font = body_font = ImageFont.load_default()
    for frame, label, success in sequence:
        image = Image.fromarray(np.moveaxis(frame, 0, -1)).resize((528, 528), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (760, 620), (7, 18, 25)); canvas.paste(image, (18, 18)); draw = ImageDraw.Draw(canvas)
        draw.text((566, 30), "CONCEPT", font=body_font, fill=(130, 151, 170))
        draw.text((566, 55), "GENESIS", font=title_font, fill=(100, 230, 177))
        draw.text((566, 95), "raw pixels", font=body_font, fill=(232, 240, 245))
        draw.text((566, 120), "unnamed events", font=body_font, fill=(232, 240, 245))
        draw.text((566, 145), "sparse reward", font=body_font, fill=(232, 240, 245))
        draw.text((20, 562), label + (" · SUCCESS" if success else ""), font=title_font,
                  fill=(100, 230, 177) if success else (232, 240, 245))
        frames.append(canvas)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=700, loop=0, disposal=2)


def protocol(source_root: Path, output: Path) -> dict:
    source_root = Path(source_root)
    files = [source_root / "gum" / "concept_genesis.py",
             source_root / "gum" / "concept_genesis_challenge.py"]
    value = {"format": FORMAT,
        "classification": "precommitted internal audit of unsupervised visual concept selection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": {"discovery": 121_000_000, "curriculum": 131_000_000, "sealed": 171_000_000},
        "learner_inputs": ["raw RGB frames", "five anonymous actions", "terminal scalar reward", "termination"],
        "learner_not_given": ["number of event concepts", "event definitions", "mechanism identities",
                              "control mappings", "target programs", "surface masks", "solution sequences"],
        "protocol": {"discovery_worlds": 8, "curriculum_worlds": 4,
                     "held_out_variants_per_skill": 5, "fifth_world_distinct_parent_skills": True,
                     "fifth_world_created_from_post_curriculum_brain_hash": True},
        "thresholds": {"selected_concepts_min": 5, "selected_concepts_max": 5,
                       "perfect_hidden_alignment_audit_only": True,
                       "curriculum_successes": 4, "held_out_successes": 20,
                       "retention_successes": 4, "fifth_world_success": True,
                       "fifth_world_distinct_parents": True, "minimum_fifth_speedup": 10.0,
                       "old_engineered_signature_must_fail": True, "source_frozen": True,
                       "traces_valid": True},
        "source_hashes": {str(path.relative_to(source_root)).replace("\\", "/"): file_sha256(path) for path in files}}
    _write_json(output, value); return value


def run(source_root: Path, output_root: Path) -> dict:
    source_root = Path(source_root); output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "CONCEPT_GENESIS_PROTOCOL.json"
    frozen = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else protocol(source_root, protocol_path)
    before_hashes = {name: file_sha256(source_root / Path(name)) for name in frozen["source_hashes"]}
    if before_hashes != frozen["source_hashes"]: raise RuntimeError("implementation differs from frozen protocol")
    run_id = "concept-genesis-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seeds = frozen["seeds"]; targets = _targets(seeds["curriculum"])

    discovery_worlds = [GenesisWorld(seeds["discovery"] + index * 1009, (0, 1, 2), index)
                        for index in range(frozen["protocol"]["discovery_worlds"])]
    mind = ConceptGenesisMind(); discovery = mind.discover(discovery_worlds)
    alignment_worlds = [GenesisWorld(seeds["sealed"] + 7_000_000 + index * 313, (0, 1, 2), index + 20)
                        for index in range(6)]
    alignment = _alignment_audit(mind, alignment_worlds)
    old_baseline = _old_signature_baseline(alignment_worlds[0])

    training_trace = TransitionTrace(output_root / "CURRICULUM_TRANSITIONS.jsonl")
    curriculum = []; curriculum_worlds = []
    for index, target in enumerate(targets):
        world = GenesisWorld(seeds["curriculum"] + index * 2017, target, index + 40)
        curriculum_worlds.append(world)
        curriculum.append(mind.acquire(world, _trace_callback(training_trace, run_id, f"curriculum-{index}")))
    curriculum_brain = output_root / "GENESIS_MIND_AFTER_CURRICULUM.json"; mind.save(curriculum_brain)
    brain_hash = file_sha256(curriculum_brain)

    held_out = []
    for family, target in enumerate(targets):
        for variant in range(frozen["protocol"]["held_out_variants_per_skill"]):
            world = GenesisWorld(seeds["sealed"] + family * 100_000 + variant * 401, target, 60 + family * 5 + variant)
            held_out.append({"family_audit_only": family, "variant": variant, **mind.solve(world),
                             "world_audit": world.audit()})

    mind.save(curriculum_brain); restored = ConceptGenesisMind.load(curriculum_brain)
    retention = [restored.revisit(world) for world in curriculum_worlds]

    selector = int(brain_hash[:16], 16); left = selector % len(targets)
    right = (selector // len(targets)) % (len(targets) - 1)
    if right >= left: right += 1
    fifth_target = targets[left] + targets[right]
    fifth_seed = seeds["sealed"] + int(brain_hash[16:24], 16) % 1_000_000
    fifth_trace = TransitionTrace(output_root / "FIFTH_EXPERIENCED_TRANSITIONS.jsonl")
    experienced_world = GenesisWorld(fifth_seed, fifth_target, 101)
    experienced = restored.compose(experienced_world, _trace_callback(fifth_trace, run_id, "fifth-experienced"))

    # A perception-matched blank mind has the same discovered visual vocabulary but no skills or context memory.
    fresh = ConceptGenesisMind(GenesisEncoder.from_json(restored.encoder.to_json()))
    fresh_trace = TransitionTrace(output_root / "FIFTH_FRESH_TRANSITIONS.jsonl")
    fresh_result = fresh.acquire(GenesisWorld(fifth_seed, fifth_target, 101),
                                 _trace_callback(fresh_trace, run_id, "fifth-fresh"))
    speedup = fresh_result["interactions"] / max(1, experienced["interactions"])
    final_brain = output_root / "GENESIS_MIND_FINAL.json"; restored.save(final_brain)

    traces = {"curriculum": training_trace.verify(), "fifth_experienced": fifth_trace.verify(),
              "fifth_fresh": fresh_trace.verify()}
    trace_text = "\n".join(path.read_text(encoding="utf-8") for path in
        (training_trace.path, fifth_trace.path, fresh_trace.path))
    hidden_terms = [term for term in ("target_mechanisms", "control_to_mechanism", "surface_masks") if term in trace_text]
    after_hashes = {name: file_sha256(source_root / Path(name)) for name in frozen["source_hashes"]}
    status = restored.status()
    checks = {
        "autonomous_concept_count": discovery["selected_event_concepts"] == 5,
        "perfect_hidden_alignment_audit_only": alignment["perfect_one_to_one"],
        "old_engineered_signature_failed": not old_baseline["can_distinguish_all_actions"],
        "curriculum": sum(row["success"] for row in curriculum) == 4,
        "held_out": sum(row["success"] for row in held_out) == 20,
        "retention": sum(row["success"] for row in retention) == 4,
        "fifth_success": bool(experienced["success"]), "fresh_success": bool(fresh_result["success"]),
        "fifth_distinct_parents": left != right and len(set(experienced.get("parents", []))) == 2,
        "fifth_speedup": speedup >= frozen["thresholds"]["minimum_fifth_speedup"],
        "no_forgetting_of_concept_count": status["concepts_created"] == 5,
        "source_frozen": before_hashes == after_hashes == frozen["source_hashes"],
        "traces_valid": all(row["valid"] for row in traces.values()),
        "hidden_state_absent_from_learner_traces": not hidden_terms,
    }
    report = {"format": FORMAT, "run_id": run_id, "passed": all(checks.values()),
        "source_freeze": {"expected": frozen["source_hashes"], "before": before_hashes,
                          "after": after_hashes, "unchanged": before_hashes == after_hashes},
        "learner_inputs": frozen["learner_inputs"], "learner_not_given": frozen["learner_not_given"],
        "discovery": discovery, "hidden_alignment_audit_only": alignment,
        "prior_engineered_signature_baseline": old_baseline,
        "curriculum": {"successes": sum(row["success"] for row in curriculum), "rows": curriculum},
        "held_out": {"successes": sum(row["success"] for row in held_out), "worlds": len(held_out),
                     "mean_interactions": float(np.mean([row["interactions"] for row in held_out])), "rows": held_out},
        "retention": {"successes": sum(row["success"] for row in retention), "rows": retention},
        "fifth_world": {"parent_indices_audit_only": [left, right], "target_audit_only": list(fifth_target),
                        "experienced": experienced, "perception_matched_fresh": fresh_result, "speedup": speedup,
                        "created_from_post_curriculum_brain_hash": brain_hash},
        "mind": status, "brain": {"path": str(final_brain), "bytes": final_brain.stat().st_size,
                                     "sha256": file_sha256(final_brain)},
        "traces": traces, "hidden_state_scan": {"clean": not hidden_terms, "matches": hidden_terms},
        "checks": checks,
        "interpretation": ("The concept count and visual categories were selected from raw transition evidence, "
            "then used for skill transfer and composition. The generic changed-pixel/token-distribution bias "
            "remains engineered; arbitrary natural-scene concept discovery is not yet established.")}
    _write_json(output_root / "CONCEPT_GENESIS_AUDIT.json", report)

    context = restored.contexts[experienced["context"]]
    inverse = {concept: int(action) for action, concept in context["mapping"].items()}
    skill = restored.skills[experienced["skill_id"]]; actions = [inverse[x] for x in skill.sequence]
    _render_replay(GenesisWorld(fifth_seed, fifth_target, 101), actions, output_root / "CONCEPT_GENESIS_REPLAY.gif")
    lines = ["# GUM Concept Genesis Challenge", "", f"**Verdict: {'PASS' if report['passed'] else 'FAIL'}**", "",
        "## Results", "", f"- Event concepts selected without a requested count: **{discovery['selected_event_concepts']}**",
        f"- Hidden-mechanism alignment (audit only): **{'perfect' if alignment['perfect_one_to_one'] else 'imperfect'}**",
        f"- Old engineered signature distinguished: **{old_baseline['distinct_signatures']}/5** mechanisms",
        f"- Curriculum worlds: **{sum(row['success'] for row in curriculum)}/4**",
        f"- Unseen surface/control variants: **{sum(row['success'] for row in held_out)}/20**",
        f"- Retention after reload: **{sum(row['success'] for row in retention)}/4**",
        f"- Distinct-skill composition: **{'yes' if checks['fifth_distinct_parents'] else 'no'}**",
        f"- Fifth world: experienced **{experienced['interactions']}** interactions; perception-matched blank **{fresh_result['interactions']}**",
        f"- Experience advantage: **{speedup:.2f}×**", "", "## What changed", "",
        "The previous learner received a hand-written event signature. This learner instead builds a pixel-change vocabulary, chooses its own event partition from unlabeled visual evidence, assigns opaque concepts, and uses those concepts to compile and compose skills.",
        "", "It is a controlled visual-world result, but it directly removes the predefined five-event vocabulary used by the prior Growth Challenge."]
    (output_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args(); path = args.output / "CONCEPT_GENESIS_PROTOCOL.json"
    result = protocol(args.source, path) if args.protocol_only else run(args.source, args.output)
    summary = result if args.protocol_only else {"format": result["format"], "passed": result["passed"],
        "discovery": result["discovery"], "curriculum": result["curriculum"]["successes"],
        "held_out": result["held_out"]["successes"], "retention": result["retention"]["successes"],
        "fifth_world": result["fifth_world"], "checks": result["checks"]}
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
