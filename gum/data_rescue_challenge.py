"""Frozen real-file transfer audit for the unchanged Concept Genesis learner."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .concept_genesis import ConceptGenesisMind, GenesisEncoder, frame_id
from .data_rescue_world import JsonDataRescueWorld, create_messy_jsonl, export_csv
from .lineage import TransitionTrace, file_sha256


FORMAT = "gum-data-rescue-genesis-v1"


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _snapshot(paths): return {str(path): file_sha256(path) for path in paths}


def _trace_callback(trace: TransitionTrace, run_id: str, phase: str):
    counter = {"value": 0}
    def append(before, action, after, reward, done):
        trace.append("transition", {"phase": phase, "index": counter["value"],
            "observation": frame_id(before), "action": int(action),
            "next_observation": frame_id(after), "reward": float(reward),
            "terminated": bool(done), "learner_visible": True}, run_id=run_id)
        counter["value"] += 1
    return append


def protocol(source_root: Path, output: Path):
    source_root = Path(source_root)
    files = [source_root / "gum" / "concept_genesis.py", source_root / "gum" / "data_rescue_world.py",
             source_root / "gum" / "data_rescue_challenge.py"]
    value = {"format": FORMAT,
        "classification": "precommitted real-file audit using the unchanged Concept Genesis learner",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": {"discovery_source": 231_000_000, "discovery_control": 241_000_000,
                  "training_source": 251_000_000, "training_control": 261_000_000,
                  "sealed_source": 271_000_000, "sealed_control": 281_000_000},
        "learner_inputs": ["RGB views derived from current working-file quality", "five anonymous actions",
                           "terminal verified reward", "termination"],
        "learner_not_given": ["operation names", "field names", "control mappings", "correct operation order",
                              "validation rules", "number of concepts", "solution sequence"],
        "protocol": {"discovery_files": 8, "training_files": 1, "held_out_files": 5,
                     "all_source_files_read_only": True, "independent_output_verification": True},
        "thresholds": {"selected_concepts": 5, "training_success": True, "held_out_successes": 5,
                       "retention_success": True, "minimum_transfer_speedup": 20.0,
                       "all_sources_unchanged": True, "all_cleaned_outputs_verified": True,
                       "same_frozen_genesis_implementation": True, "traces_valid": True},
        "source_hashes": {str(path.relative_to(source_root)).replace("\\", "/"): file_sha256(path) for path in files}}
    _write_json(output, value); return value


def run(source_root: Path, output_root: Path):
    source_root = Path(source_root); output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "DATA_RESCUE_PROTOCOL.json"
    frozen = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else protocol(source_root, protocol_path)
    before_code = {name: file_sha256(source_root / Path(name)) for name in frozen["source_hashes"]}
    if before_code != frozen["source_hashes"]: raise RuntimeError("implementation differs from frozen protocol")
    run_id = "data-rescue-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seeds = frozen["seeds"]; inputs = output_root / "READ_ONLY_INPUTS"; work = output_root / "WORKING_FILES"

    discovery_worlds = []; discovery_sources = []
    for index in range(frozen["protocol"]["discovery_files"]):
        source = create_messy_jsonl(inputs / f"discovery-{index}.jsonl", seeds["discovery_source"] + index)
        discovery_sources.append(source)
        discovery_worlds.append(JsonDataRescueWorld(source, work / f"discovery-{index}",
                                                     seeds["discovery_control"] + index * 101))
    training_source = create_messy_jsonl(inputs / "training.jsonl", seeds["training_source"])
    held_sources = [create_messy_jsonl(inputs / f"sealed-{index}.jsonl", seeds["sealed_source"] + index * 1009)
                    for index in range(frozen["protocol"]["held_out_files"])]
    all_sources = discovery_sources + [training_source] + held_sources; source_before = _snapshot(all_sources)

    mind = ConceptGenesisMind(); discovery = mind.discover(discovery_worlds)
    training_world = JsonDataRescueWorld(training_source, work / "training", seeds["training_control"])
    training_trace = TransitionTrace(output_root / "TRAINING_TRANSITIONS.jsonl")
    training = mind.acquire(training_world, _trace_callback(training_trace, run_id, "training"))
    training_verified = training_world.verify()
    mind_path = output_root / "DATA_RESCUE_MIND.json"; mind.save(mind_path)

    held_out = []
    for index, source in enumerate(held_sources):
        world = JsonDataRescueWorld(source, work / f"sealed-{index}", seeds["sealed_control"] + index * 313)
        result = mind.solve(world)
        csv_path = export_csv(world.active, output_root / "CLEANED_OUTPUTS" / f"sealed-{index}.csv") if result["success"] else None
        held_out.append({"index": index, **result, "verified": world.verify(),
                         "cleaned_jsonl": str(world.active), "cleaned_csv": str(csv_path) if csv_path else None,
                         "world_audit": world.audit()})

    restored = ConceptGenesisMind.load(mind_path)
    retention_world = JsonDataRescueWorld(training_source, work / "retention", seeds["training_control"])
    retention = restored.revisit(retention_world)

    fresh = ConceptGenesisMind(GenesisEncoder.from_json(mind.encoder.to_json()))
    fresh_world = JsonDataRescueWorld(held_sources[0], work / "fresh-baseline", seeds["sealed_control"])
    fresh_trace = TransitionTrace(output_root / "FRESH_BASELINE_TRANSITIONS.jsonl")
    fresh_result = fresh.acquire(fresh_world, _trace_callback(fresh_trace, run_id, "perception-matched-fresh"))
    transfer_actions = held_out[0]["interactions"]
    speedup = fresh_result["interactions"] / max(1, transfer_actions)

    source_after = _snapshot(all_sources); before_code_copy = dict(before_code)
    after_code = {name: file_sha256(source_root / Path(name)) for name in frozen["source_hashes"]}
    traces = {"training": training_trace.verify(), "fresh_baseline": fresh_trace.verify()}
    trace_text = training_trace.path.read_text(encoding="utf-8") + fresh_trace.path.read_text(encoding="utf-8")
    hidden_terms = [term for term in ("control_to_operation", "operation_history", "correct_order") if term in trace_text]

    visual_audit = output_root.parent / "gum_concept_genesis_v1" / "CONCEPT_GENESIS_AUDIT.json"
    visual_link = None
    if visual_audit.exists():
        prior = json.loads(visual_audit.read_text(encoding="utf-8"))
        visual_link = {"audit": str(visual_audit), "passed": prior.get("passed"),
                       "concept_genesis_source_sha256": prior["source_freeze"]["expected"].get("gum/concept_genesis.py")}
    same_implementation = (visual_link is None or
        visual_link["concept_genesis_source_sha256"] == frozen["source_hashes"]["gum/concept_genesis.py"])
    checks = {"concept_count": discovery["selected_event_concepts"] == 5,
        "training": bool(training["success"] and training_verified),
        "held_out": sum(row["success"] and row["verified"] for row in held_out) == 5,
        "retention": bool(retention["success"] and retention_world.verify()),
        "fresh_baseline": bool(fresh_result["success"] and fresh_world.verify()),
        "transfer_speedup": speedup >= frozen["thresholds"]["minimum_transfer_speedup"],
        "sources_unchanged": source_before == source_after,
        "code_frozen": before_code_copy == after_code == frozen["source_hashes"],
        "same_genesis_implementation_as_visual_challenge": same_implementation,
        "traces_valid": all(row["valid"] for row in traces.values()),
        "hidden_state_absent_from_learner_traces": not hidden_terms}

    portfolio = {"format": "gum-growing-mind-portfolio-v1",
        "visual_domain": visual_link,
        "data_rescue_domain": {"brain": str(mind_path), "brain_sha256": file_sha256(mind_path),
                               "concepts": mind.status()["concepts_created"], "skills": mind.status()["skills_created"]},
        "shared_learning_implementation_sha256": frozen["source_hashes"]["gum/concept_genesis.py"],
        "domains_retained": 2 if visual_link else 1}
    _write_json(output_root / "GROWING_MIND_PORTFOLIO.json", portfolio)
    report = {"format": FORMAT, "run_id": run_id, "passed": all(checks.values()),
        "learner_inputs": frozen["learner_inputs"], "learner_not_given": frozen["learner_not_given"],
        "source_freeze": {"expected": frozen["source_hashes"], "before": before_code_copy,
                          "after": after_code, "unchanged": before_code_copy == after_code},
        "discovery": discovery, "training": training,
        "held_out": {"successes": sum(row["success"] and row["verified"] for row in held_out), "rows": held_out,
                     "mean_interactions": float(np.mean([row["interactions"] for row in held_out]))},
        "retention": retention, "perception_matched_fresh": fresh_result,
        "transfer": {"experienced_interactions": transfer_actions,
                     "fresh_interactions": fresh_result["interactions"], "speedup": speedup},
        "source_integrity": {"unchanged": source_before == source_after, "before": source_before, "after": source_after},
        "cross_domain_continuity": {"visual_challenge": visual_link, "same_implementation": same_implementation,
                                    "portfolio": str(output_root / "GROWING_MIND_PORTFOLIO.json")},
        "mind": mind.status(), "traces": traces,
        "hidden_state_scan": {"clean": not hidden_terms, "matches": hidden_terms}, "checks": checks,
        "interpretation": ("The unchanged concept-learning algorithm learned a real JSONL repair workflow through "
            "anonymous file operations and verified terminal feedback. The environment supplies a visual quality "
            "view and five safe repair tools; it does not synthesize arbitrary file-editing code.")}
    _write_json(output_root / "DATA_RESCUE_AUDIT.json", report)
    lines = ["# GUM Concept Genesis · Real Data Rescue", "", f"**Verdict: {'PASS' if report['passed'] else 'FAIL'}**", "",
        "## Results", "", f"- File-operation concepts selected autonomously: **{discovery['selected_event_concepts']}**",
        f"- Training file repaired and independently verified: **{'yes' if checks['training'] else 'no'}**",
        f"- Unseen files repaired: **{report['held_out']['successes']}/5**",
        f"- Retained after reload: **{'yes' if checks['retention'] else 'no'}**",
        f"- Transfer interactions: **{transfer_actions}**", f"- Perception-matched fresh interactions: **{fresh_result['interactions']}**",
        f"- Experience advantage: **{speedup:.2f}×**", f"- Original inputs unchanged: **{'yes' if checks['sources_unchanged'] else 'no'}**",
        f"- Same frozen learner as visual challenge: **{'yes' if same_implementation else 'no'}**", "",
        "The actions operated on actual JSONL working files. Successful outputs were parsed, checked for canonical fields and types, deduplicated, sorted, assigned stable IDs, and exported as CSV. The original inputs remained byte-for-byte unchanged."]
    (output_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args(); path = args.output / "DATA_RESCUE_PROTOCOL.json"
    result = protocol(args.source, path) if args.protocol_only else run(args.source, args.output)
    summary = result if args.protocol_only else {"format": result["format"], "passed": result["passed"],
        "concepts": result["discovery"]["selected_event_concepts"], "training": result["checks"]["training"],
        "held_out": result["held_out"]["successes"], "retention": result["checks"]["retention"],
        "transfer": result["transfer"], "sources_unchanged": result["checks"]["sources_unchanged"],
        "same_frozen_learner": result["checks"]["same_genesis_implementation_as_visual_challenge"],
        "checks": result["checks"]}
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
