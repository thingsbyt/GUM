"""World lifecycle, communication, practice, evidence, and persistence."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np

from .lineage import HashLedger, TransitionTrace, file_sha256
from .mind import PixelQLearner, observation_id
from .registry import AdapterRegistry
from .world_creator import WorldCreator, load_world
from .builtin_worlds import (ASTEROIDS_ADAPTER, IMMUNE_ADAPTER, create_builtin_world,
                             load_asteroids, load_immune)
from .specialist_minds import CooperativeTeamMind


class GUMHarness:
    format = "gum-harness-v1"

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace); self.workspace.mkdir(parents=True, exist_ok=True)
        self.world_root = self.workspace / "worlds"; self.run_root = self.workspace / "runs"
        self.artifact_root = self.workspace / "artifacts"; self.world_root.mkdir(exist_ok=True)
        self.run_root.mkdir(exist_ok=True); self.artifact_root.mkdir(exist_ok=True)
        self.ledger = HashLedger(self.workspace / "EVOLUTION.jsonl"); self.creator = WorldCreator(self.world_root)
        self.adapters = AdapterRegistry(); self.adapters.register("gum-grid-v1", load_world)
        self.adapters.register(ASTEROIDS_ADAPTER, load_asteroids)
        self.adapters.register(IMMUNE_ADAPTER, load_immune)
        self.mind_path = self.workspace / "GUM_MIND.json"
        self.mind = PixelQLearner.load(self.mind_path) if self.mind_path.exists() else PixelQLearner()
        self.language_path = self.workspace / "GROUNDED_LANGUAGE_MEMORY.json"
        self.language = None
        if self.language_path.exists():
            from jepa_asteroids.grounded_dialogue import GroundedLanguageBridge
            self.language = GroundedLanguageBridge.load(self.language_path)
        self.active_world: Path | None = None; self.messages = []
        self.services = {}
        self.ledger.append("harness-opened", {"format": self.format, "mind": self.mind.status()})

    def create_world(self, *, seed: int, difficulty=0, parent: Path | None = None):
        if parent is None:
            folder = self.creator.create(seed=seed, difficulty=difficulty)
        else:
            folder = self.creator.mutate(parent, seed=seed, harder=difficulty >= 0)
        self.active_world = folder; public = json.loads((folder / "world.json").read_text())
        private_hash = file_sha256(folder / "genome.private.json")
        self.ledger.append("world-created", {"world": public, "private_genome_sha256": private_hash})
        return folder

    def load_world(self, folder: Path):
        folder = Path(folder)
        if not (folder / "world.json").exists() or not (folder / "genome.private.json").exists():
            raise ValueError("world package needs world.json and genome.private.json")
        public = json.loads((folder / "world.json").read_text())
        if public.get("protocol_version") != 1: raise ValueError("unsupported world protocol")
        self.adapters.load(public.get("adapter"), folder)  # Validate before activation.
        self.active_world = folder; self.ledger.append("world-loaded", {"world_id": public["world_id"]})
        return public

    def register_adapter(self, name, loader):
        self.adapters.register(name, loader); self.ledger.append("adapter-registered", {"adapter": name})

    def create_builtin_world(self, family: str, *, seed: int, **options):
        folder = create_builtin_world(self.world_root, family, seed=seed, **options)
        self.load_world(folder)
        self.ledger.append("builtin-world-created", {"family": family, "world_id": folder.name,
            "private_genome_sha256": file_sha256(folder / "genome.private.json")})
        return folder

    def register_perception_service(self, name: str, service):
        self.services[str(name)] = service
        self.ledger.append("perception-service-registered", {"name": str(name), "status": service.status()})

    def perceive_file(self, service_name: str, image: Path, **options):
        if service_name not in self.services: raise ValueError(f"unknown perception service {service_name!r}")
        result = self.services[service_name].identify(Path(image), **options)
        self.ledger.append("perception-used", {"service": service_name, "image_sha256": file_sha256(image),
            "prediction": result["prediction"], "confidence": result["confidence"]})
        return result

    def attach_language_memory(self, source: Path):
        from jepa_asteroids.grounded_dialogue import GroundedLanguageBridge
        source = Path(source); model = GroundedLanguageBridge.load(source)
        shutil.copy2(source, self.language_path); self.language = model
        self.ledger.append("language-memory-attached", {"source_sha256": file_sha256(source),
            "examples": len(model.examples), "learned_words": len(model.probability)})
        return {"examples": len(model.examples), "learned_words": len(model.probability)}

    def _open_active_world(self):
        if self.active_world is None: raise RuntimeError("no active world")
        public = json.loads((self.active_world / "world.json").read_text())
        return self.adapters.load(public["adapter"], self.active_world)

    def _episodes(self, world, count, *, training, seed_base, trace, run_id):
        results = []
        for episode in range(count):
            observation = world.reset(seed_base + episode); self.mind.begin(world.public_spec(), observation, training=training)
            total = 0.0; steps = 0; success = False
            while True:
                action = self.mind.act(observation, training=training); before = observation_id(observation)
                transition = world.step(action); self.mind.observe(action, transition, training=training)
                trace.append("transition", {"episode": episode, "training": training, "step": steps,
                    "observation": before, "action": action, "reward": transition.reward,
                    "next_observation": observation_id(transition.observation),
                    "terminated": transition.terminated, "truncated": transition.truncated,
                    "public_info": transition.public_info}, run_id=run_id)
                observation = transition.observation; total += transition.reward; steps += 1
                if transition.terminated or transition.truncated:
                    success = bool(transition.public_info.get("success")); break
            if training: self.mind.finish_episode()
            results.append({"episode": episode, "success": success, "return": total, "steps": steps})
            if training and (episode + 1) % max(1, count // 10) == 0:
                self.ledger.append("learning-checkpoint", {"episode": episode + 1,
                    "recent_success": float(np.mean([row["success"] for row in results[-max(10, count // 10):]])),
                    "mind": self.mind.status()}, run_id=run_id)
        return results

    @staticmethod
    def _summary(rows):
        return {"episodes": len(rows), "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows]))}

    def practice(self, *, training_episodes=600, evaluation_episodes=120, seed=9_100_001):
        if self.active_world is None: raise RuntimeError("no active world")
        world = self._open_active_world(); run_id = str(uuid.uuid4()); run_folder = self.run_root / run_id
        run_folder.mkdir(parents=True); trace = TransitionTrace(run_folder / "TRANSITIONS.jsonl")
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.ledger.append("practice-started", {"world": world.public_spec().to_json(),
            "training_episodes": training_episodes, "evaluation_episodes": evaluation_episodes,
            "mind_before": self.mind.status(), "harness_source_sha256": source_hash}, run_id=run_id)
        self.mind.epsilon = 0.0
        before_rows = self._episodes(world, min(60, evaluation_episodes), training=False,
                                     seed_base=seed + 1_000_000, trace=trace, run_id=run_id)
        for episode in range(training_episodes):
            self.mind.epsilon = max(.04, 1.0 - episode / max(1, training_episodes * .82))
            rows = self._episodes(world, 1, training=True, seed_base=seed + episode,
                                  trace=trace, run_id=run_id)
            if (episode + 1) % max(1, training_episodes // 10) == 0:
                self.ledger.append("learning-checkpoint", {"episode": episode + 1,
                    "latest_success": rows[-1]["success"], "mind": self.mind.status()}, run_id=run_id)
        self.mind.epsilon = 0.0
        after_rows = self._episodes(world, evaluation_episodes, training=False,
                                    seed_base=seed + 2_000_000, trace=trace, run_id=run_id)
        self.mind.save(self.mind_path); trace_check = trace.verify()
        report = {"format": "gum-practice-report-v1", "run_id": run_id,
            "world": world.public_spec().to_json(), "before": self._summary(before_rows),
            "after": self._summary(after_rows), "mind_after": self.mind.status(),
            "transition_trace": {"path": str(trace.path), **trace_check},
            "mind_sha256": file_sha256(self.mind_path),
            "learner_inputs": ["RGB pixels", "anonymous action slots", "scalar reward", "termination"],
            "prohibited_inputs": ["goal coordinates", "hazard coordinates", "wall coordinates", "action meanings"]}
        report["learned"] = bool(report["after"]["success_rate"] >= .80 and
                                 report["after"]["success_rate"] >= report["before"]["success_rate"] + .30)
        report_path = run_folder / "REPORT.json"; report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.ledger.append("practice-completed", {"report": str(report_path),
            "report_sha256": file_sha256(report_path), "before": report["before"], "after": report["after"],
            "learned": report["learned"], "mind": self.mind.status()}, run_id=run_id)
        return report

    def run_specialist(self, family: str, *, episodes=1, seed=9_400_001, communicate=True):
        """Run a cooperative specialist entirely through the common world/mind protocol."""
        if self.active_world is None: raise RuntimeError("no active world")
        world = self._open_active_world(); spec = world.public_spec()
        if spec.agents != 2: raise ValueError("cooperative specialist needs a two-agent world")
        specialist_root = self.artifact_root / "specialists"; specialist_root.mkdir(exist_ok=True)
        safe_family = str(family).replace("/", "-"); persistent = specialist_root / f"{safe_family}--{spec.world_id}.json"
        resumed = persistent.exists()
        mind = (CooperativeTeamMind.load(persistent, trusted=True) if resumed
                else CooperativeTeamMind(family, communicate=communicate))
        run_id = str(uuid.uuid4()); folder = self.run_root / run_id; folder.mkdir(parents=True)
        trace = TransitionTrace(folder / "TRANSITIONS.jsonl"); rows = []
        self.ledger.append("specialist-run-started", {"world": spec.to_json(), "mind": mind.status(),
            "episodes": int(episodes), "resumed_from_persistent_memory": resumed}, run_id=run_id)
        for episode in range(int(episodes)):
            observation = world.reset(int(seed) + episode); mind.begin(spec, observation, training=True)
            total = 0.0; steps = 0
            while True:
                before = observation_id(observation); actions = mind.act(observation, training=True)
                transition = world.step(actions); mind.observe(actions, transition, training=True)
                trace.append("transition", {"episode": episode, "step": steps, "observation": before,
                    "actions": actions, "reward": transition.reward,
                    "next_observation": observation_id(transition.observation),
                    "terminated": transition.terminated, "truncated": transition.truncated,
                    "public_info": transition.public_info}, run_id=run_id)
                observation = transition.observation; total += transition.reward; steps += 1
                if transition.terminated or transition.truncated: break
            audit = world.audit_state(); rows.append({"episode": episode, "steps": steps,
                "return": total, "success": bool(transition.public_info.get("success")),
                "public_final": transition.public_info, "audit_final": audit})
            self.ledger.append("specialist-episode-completed", {"episode": episode,
                "result": rows[-1], "mind": mind.status()}, run_id=run_id)
        mind.save(persistent)
        snapshot = folder / "MIND_SNAPSHOT.json"; mind.save(snapshot)
        report = {"format": "gum-specialist-run-v1", "run_id": run_id, "world": spec.to_json(),
            "family": family, "episodes": rows, "mind": mind.status(),
            "resumed_from_persistent_memory": resumed,
            "persistent_mind": {"path": str(persistent), "sha256": file_sha256(persistent)},
            "mind_snapshot": {"path": str(snapshot), "sha256": file_sha256(snapshot)},
            "transition_trace": {"path": str(trace.path), **trace.verify()},
            "learner_inputs": ["each agent's local pixels", "anonymous actions", "shared reward", "termination"],
            "prohibited_inputs": ["hidden control maps", "object coordinates", "hidden recipe", "cell labels"]}
        report_path = folder / "REPORT.json"; report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.ledger.append("specialist-run-completed", {"report": str(report_path),
            "report_sha256": file_sha256(report_path), "trace_valid": report["transition_trace"]["valid"],
            "mind": mind.status()}, run_id=run_id)
        return report

    def evaluate_retention(self, world_folder: Path, *, episodes=120, seed=9_900_001):
        previous = self.active_world; self.load_world(world_folder); world = self._open_active_world()
        run_id = str(uuid.uuid4()); folder = self.run_root / run_id; folder.mkdir(parents=True)
        trace = TransitionTrace(folder / "TRANSITIONS.jsonl"); self.mind.epsilon = 0.0
        rows = self._episodes(world, episodes, training=False, seed_base=seed, trace=trace, run_id=run_id)
        self.mind.save(self.mind_path)
        result = self._summary(rows) | {"world_id": world.public_spec().world_id, "trace_valid": trace.verify()["valid"]}
        self.ledger.append("retention-evaluated", result, run_id=run_id); self.active_world = previous; return result

    def status(self):
        world = None if self.active_world is None else json.loads((self.active_world / "world.json").read_text())
        return {"format": self.format, "project": "GUM — Growing Understanding Machine",
            "active_world": world, "mind": self.mind.status(), "evolution": self.ledger.verify(),
            "registered_adapters": self.adapters.names(),
            "perception_services": {name: service.status() for name, service in self.services.items()},
            "language": None if self.language is None else {"attached": True,
                "learned_words": len(self.language.probability), "demonstrations": len(self.language.examples)},
            "messages": self.messages[-20:]}

    def communicate(self, text: str):
        lower = text.lower(); response = None
        if "status" in lower or "what do you know" in lower:
            state = self.status(); response = (f"I retain {state['mind']['learned_states']} learned visual states across "
                f"{state['mind']['worlds_retained']} worlds. The evolutionary ledger has "
                f"{state['evolution']['records']} verified records.")
        elif "create" in lower and "world" in lower:
            difficulty = 1 if "hard" in lower else 0
            seed = int(datetime.now(timezone.utc).timestamp() * 1000) % 2_000_000_000
            parent = self.active_world if "hard" in lower and self.active_world is not None else None
            folder = self.create_world(seed=seed, difficulty=difficulty, parent=parent)
            response = f"Created and loaded {folder.name}."
        elif "practice" in lower or "learn this world" in lower:
            report = self.practice(); response = (f"Practice finished. Success changed from "
                f"{report['before']['success_rate']:.1%} to {report['after']['success_rate']:.1%}.")
        elif self.language is not None:
            interpretation = self.language.interpret(text); response = interpretation.response
        else:
            response = ("I can create a world, create a harder world, practice, or report status. "
                        "In-world grounded instructions are handled by the language bridge.")
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "human": text, "gum": response}
        self.messages.append(row); self.ledger.append("conversation", row); return response
