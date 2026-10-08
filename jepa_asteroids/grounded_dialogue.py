"""A learned human-language intermediary for grounded concepts and actions.

Words are not installed in a dictionary.  The bridge estimates their meanings
from utterance + successful-experience pairs, then composes familiar meanings
inside sentences that were never demonstrated verbatim.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np

from .concept_bridge import CONCEPTS


NULL = "meaning:null"
ACTIONS = ("find", "approach", "avoid", "report")


def _normalize(word: str) -> str:
    word = word.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            candidate = word[:-len(suffix)]
            if suffix == "ing" and candidate.endswith("y"):  # staying -> stay
                return candidate
            return candidate
    return word


def tokenize(text: str) -> list[str]:
    return [_normalize(word) for word in re.findall(r"[a-zA-Z_]+", text)]


@dataclass(frozen=True)
class GroundedTask:
    steps: tuple[tuple[str, str], ...]
    avoid: tuple[str, ...] = ()

    def plain(self) -> str:
        pieces = [f"{action} {target}" for action, target in self.steps]
        value = "; then ".join(pieces) if pieces else "no primary action"
        if self.avoid: value += "; avoid " + ", ".join(self.avoid)
        return value


@dataclass(frozen=True)
class Interpretation:
    status: str
    task: GroundedTask | None
    confidence: float
    response: str
    uncertain_words: tuple[str, ...] = ()

    def to_json(self):
        return {"status": self.status, "task": None if self.task is None else asdict(self.task),
                "confidence": self.confidence, "response": self.response,
                "uncertain_words": list(self.uncertain_words)}


class GroundedLanguageBridge:
    """IBM-1-style lexical induction plus a small grounded task workspace."""
    format = "wailah-grounded-language-bridge-v1"

    def __init__(self):
        self.examples: list[tuple[list[str], list[str]]] = []
        self.probability: dict[str, dict[str, float]] = {}
        self.support = Counter(); self.training_iterations = 0

    def demonstrate(self, utterance: str, successful_experience: list[tuple[str, str]]) -> None:
        """Learn from language paired with a successful observed action trace."""
        atoms = []
        for action, target in successful_experience:
            atoms.extend((f"action:{action}", f"object:{target}"))
        self.examples.append((tokenize(utterance), atoms))
        self.support.update(atoms)

    def learn(self, iterations: int = 80) -> None:
        vocabulary = sorted({word for words, _ in self.examples for word in words})
        meanings = sorted({atom for _, atoms in self.examples for atom in atoms}) + [NULL]
        if not vocabulary or len(meanings) == 1:
            raise RuntimeError("language learning needs demonstrations")
        probability = {word: {meaning: 1 / len(meanings) for meaning in meanings}
                       for word in vocabulary}
        for _ in range(iterations):
            counts = {word: Counter() for word in vocabulary}
            for words, atoms in self.examples:
                available = list(dict.fromkeys(atoms)) + [NULL]
                for word in words:
                    normalizer = sum(probability[word][meaning] for meaning in available)
                    for meaning in available:
                        counts[word][meaning] += probability[word][meaning] / max(normalizer, 1e-12)
            for word in vocabulary:
                # A light prior keeps rare words uncertain instead of overclaiming.
                total = sum(counts[word].values()) + .02 * len(meanings)
                probability[word] = {meaning: (counts[word][meaning] + .02) / total
                                     for meaning in meanings}
        self.probability = probability; self.training_iterations += iterations

    def _word_meaning(self, word: str):
        rows = self.probability.get(word)
        if rows is None: return None, 0.0, 0.0
        semantic = sorted(((meaning, score) for meaning, score in rows.items() if meaning != NULL),
                          key=lambda row: row[1], reverse=True)
        best, score = semantic[0]; runner = semantic[1][1] if len(semantic) > 1 else 0.0
        margin = float(score - runner); null_score = float(rows[NULL])
        # NULL competes with the whole phrase. A word may share its meaning with
        # adjacent phrase tokens, but must beat every *other* grounded meaning.
        if score >= .25 and margin >= .12 and score >= .8 * null_score:
            return best, float(score), margin
        return NULL, null_score, max(0.0, null_score - float(score))

    def lexicon(self) -> dict:
        output = {}
        for word in sorted(self.probability):
            meaning, score, margin = self._word_meaning(word)
            output[word] = {"meaning": meaning, "probability": score, "margin": margin}
        return output

    def interpret(self, utterance: str) -> Interpretation:
        words = tokenize(utterance); decoded = []; uncertain = []
        for position, word in enumerate(words):
            meaning, score, margin = self._word_meaning(word)
            if meaning is None:
                uncertain.append(word); continue
            if meaning != NULL and score >= .25 and margin >= .12:
                decoded.append((position, meaning, score))
            elif meaning != NULL:
                uncertain.append(word)
        # Multiword phrases such as "move to", "stay away from", and compound
        # object names may assign identical meaning to adjacent tokens.
        decoded = [row for index, row in enumerate(decoded)
                   if index == 0 or row[1] != decoded[index - 1][1]]
        if uncertain:
            unique = tuple(dict.fromkeys(uncertain))
            quoted = ", ".join(f"'{word}'" for word in unique)
            return Interpretation("clarify", None, 0.0,
                f"I have not grounded {quoted} well enough. Please demonstrate what it means.", unique)

        actions = [(position, atom.split(":", 1)[1], score) for position, atom, score in decoded
                   if atom.startswith("action:")]
        objects = [(position, atom.split(":", 1)[1], score) for position, atom, score in decoded
                   if atom.startswith("object:")]
        used = set(); pairs = []
        for action_position, action, action_score in actions:
            candidates = [(position, target, score, index)
                          for index, (position, target, score) in enumerate(objects)
                          if index not in used and position > action_position]
            if not candidates:
                return Interpretation("clarify", None, 0.0,
                    f"I understood the action '{action}', but not what it should apply to.")
            position, target, target_score, index = min(candidates)
            used.add(index); pairs.append((action_position, action, target, min(action_score, target_score)))
        if not pairs:
            return Interpretation("clarify", None, 0.0,
                "I could not connect that sentence to a grounded action.")
        constraints = tuple(target for _, action, target, _ in pairs if action == "avoid")
        steps = tuple((action, target) for _, action, target, _ in pairs if action != "avoid")
        if not steps:
            return Interpretation("clarify", None, 0.0,
                "I understood what to avoid, but I still need a positive goal.")
        conflicts = sorted(set(target for _, target in steps) & set(constraints))
        if conflicts:
            return Interpretation("clarify", None, 0.0,
                f"You asked me to approach and avoid {', '.join(conflicts)}. Which instruction has priority?")
        task = GroundedTask(steps, constraints)
        confidence = float(np.mean([row[3] for row in pairs]))
        return Interpretation("understood", task, confidence,
            f"I understood: {task.plain()}.")

    def save(self, path: Path) -> None:
        value = {"format": self.format, "training_iterations": self.training_iterations,
            "examples": [{"words": words, "atoms": atoms} for words, atoms in self.examples],
            "probability": self.probability, "support": dict(self.support)}
        Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != cls.format: raise ValueError("unsupported language memory")
        model = cls(); model.examples = [(row["words"], row["atoms"]) for row in value["examples"]]
        model.probability = {word: {meaning: float(score) for meaning, score in rows.items()}
                             for word, rows in value["probability"].items()}
        model.support = Counter({key: int(count) for key, count in value["support"].items()})
        model.training_iterations = int(value["training_iterations"]); return model


class GroundedGridWorld:
    """A tiny task executor proving that interpreted meaning controls behavior."""
    def __init__(self, seed: int, concepts=CONCEPTS, size: int = 9):
        self.size = size; rng = random.Random(seed); cells = [(x, y) for y in range(size) for x in range(size)]
        rng.shuffle(cells); self.agent = cells.pop(); self.objects = {name: cells.pop() for name in concepts}

    def _path(self, start, goal, forbidden):
        queue = deque([start]); previous = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == goal: break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cell[0] + dx, cell[1] + dy)
                if (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size
                        and nxt not in forbidden and nxt not in previous):
                    previous[nxt] = cell; queue.append(nxt)
        if goal not in previous: return None
        path = []; cell = goal
        while previous[cell] is not None: path.append(cell); cell = previous[cell]
        return list(reversed(path))

    def execute(self, task: GroundedTask) -> dict:
        forbidden = {self.objects[name] for name in task.avoid}; position = self.agent
        trace = []; visited = []; success = True
        for action, target in task.steps:
            if target not in self.objects: success = False; break
            path = self._path(position, self.objects[target], forbidden)
            if path is None: success = False; break
            trace.extend(path); position = self.objects[target]; visited.append((action, target))
        if any(cell in forbidden for cell in trace): success = False
        return {"success": success, "visited": visited, "avoided": list(task.avoid),
                "movement_steps": len(trace), "path": trace}


FIND = ("find {x}", "please locate the {x}", "search for the {x}")
APPROACH = ("approach {x}", "move to the {x}", "go toward the {x}")
AVOID = ("avoid {x}", "stay away from the {x}", "keep clear of {x}")
REPORT = ("report {x}", "tell me about the {x}", "describe the {x}")
TEMPLATES = {"find": FIND, "approach": APPROACH, "avoid": AVOID, "report": REPORT}


def teach_default_language(model: GroundedLanguageBridge) -> list[str]:
    """A curriculum of human demonstrations; meanings are not installed directly."""
    utterances = []
    for action, templates in TEMPLATES.items():
        for index, concept in enumerate(CONCEPTS):
            for template in templates:
                text = template.format(x=concept.replace("_", " "))
                model.demonstrate(text, [(action, concept)]); utterances.append(text)
            polite = f"please {templates[0].format(x=concept.replace('_', ' '))}"
            model.demonstrate(polite, [(action, concept)]); utterances.append(polite)
    # Demonstrate connector words without exposing the final three-part commands.
    pairs = (("apple", "clock"), ("dolphin", "lamp"), ("bicycle", "telephone"),
             ("butterfly", "wardrobe"), ("pickup_truck", "pine_tree"))
    for left, right in pairs:
        text = f"find the {left.replace('_', ' ')} then report the {right.replace('_', ' ')}"
        model.demonstrate(text, [("find", left), ("report", right)]); utterances.append(text)
        text = f"approach the {left.replace('_', ' ')} while avoid the {right.replace('_', ' ')}"
        model.demonstrate(text, [("approach", left), ("avoid", right)]); utterances.append(text)
    # Vary connective words across meanings so the learner can discover that
    # they organize a sentence rather than denote a particular action/object.
    connector_rows = (("find", "approach"), ("approach", "report"),
                      ("report", "find"), ("avoid", "report"))
    for index, (first_action, second_action) in enumerate(connector_rows):
        left, right = CONCEPTS[index + 1], CONCEPTS[index + 5]
        first = TEMPLATES[first_action][0].format(x=left.replace("_", " "))
        second = TEMPLATES[second_action][0].format(x=right.replace("_", " "))
        text = f"{first} then {second}"
        model.demonstrate(text, [(first_action, left), (second_action, right)]); utterances.append(text)
        text = f"{first} while {second}"
        model.demonstrate(text, [(first_action, left), (second_action, right)]); utterances.append(text)
    model.learn(); return utterances


def benchmark(output_dir: Path, seed: int = 12_700_003) -> dict:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    model = GroundedLanguageBridge(); training_utterances = set(teach_default_language(model))
    rng = random.Random(seed); cases = []
    for index in range(60):
        first, second, danger = rng.sample(list(CONCEPTS), 3)
        command = (f"please locate the {first.replace('_', ' ')} then move to the "
                   f"{second.replace('_', ' ')} while staying away from the {danger.replace('_', ' ')}")
        expected = GroundedTask((("find", first), ("approach", second)), (danger,))
        interpretation = model.interpret(command)
        execution = (GroundedGridWorld(seed + index).execute(interpretation.task)
                     if interpretation.task is not None else {"success": False})
        cases.append({"command": command, "never_demonstrated_verbatim": command not in training_utterances,
            "expected": asdict(expected), "interpretation": interpretation.to_json(), "execution": execution,
            "correct_meaning": interpretation.task == expected})
    clarifications = []
    for command in ("find the bank", "approach the apple while avoid the apple", "avoid the lamp"):
        value = model.interpret(command); clarifications.append({"command": command, **value.to_json()})
    memory = output_dir / "GROUNDED_LANGUAGE_MEMORY.json"; model.save(memory)
    loaded = GroundedLanguageBridge.load(memory)
    reload_command = "locate the apple then move to the clock while stay away from the lamp"
    reload_result = loaded.interpret(reload_command)
    correct = sum(row["correct_meaning"] and row["execution"]["success"] for row in cases)
    report = {"format": "wailah-grounded-dialogue-v23-audit-v1",
        "protocol": {"word_dictionary_preinstalled": False,
            "learning_signal": "human utterance paired with a successful observed action trace",
            "perception_vocabulary": list(CONCEPTS),
            "training_utterances": len(training_utterances), "sealed_novel_compositions": len(cases),
            "exact_test_sentences_seen_during_training": 0,
            "language_model_used": "none"},
        "results": {"correct_and_executed": correct, "trials": len(cases),
            "accuracy": correct / len(cases),
            "all_test_sentences_novel": all(row["never_demonstrated_verbatim"] for row in cases),
            "clarifications_requested": sum(row["status"] == "clarify" for row in clarifications),
            "clarification_trials": len(clarifications),
            "memory_reload_preserved_meaning": reload_result.task == GroundedTask(
                (("find", "apple"), ("approach", "clock")), ("lamp",))},
        "sample_cases": cases[:8], "clarification_cases": clarifications,
        "learned_lexicon": model.lexicon(),
        "integrity": {"implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "memory_sha256": hashlib.sha256(memory.read_bytes()).hexdigest()}}
    report["passed"] = bool(report["results"]["accuracy"] >= .95 and
        report["results"]["all_test_sentences_novel"] and
        report["results"]["clarifications_requested"] == len(clarifications) and
        report["results"]["memory_reload_preserved_meaning"])
    (output_dir / "GROUNDED_DIALOGUE_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    transcript = ["GROUNDED DIALOGUE TRANSCRIPT", "", "Training used demonstrations, not a word dictionary.", ""]
    for row in cases[:5]:
        transcript.extend((f"HUMAN: {row['command']}",
            f"AGENT: {row['interpretation']['response']}",
            f"RESULT: success={row['execution']['success']}", ""))
    for row in clarifications:
        transcript.extend((f"HUMAN: {row['command']}", f"AGENT: {row['response']}", ""))
    (output_dir / "TRANSCRIPT.txt").write_text("\n".join(transcript), encoding="utf-8")
    print(json.dumps({"accuracy": report["results"]["accuracy"], "trials": len(cases),
        "clarifications": report["results"]["clarifications_requested"],
        "reload": report["results"]["memory_reload_preserved_meaning"],
        "passed": report["passed"]}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--memory", type=Path); parser.add_argument("--say", type=str)
    args = parser.parse_args(argv)
    if args.benchmark: benchmark(args.benchmark); return
    if not args.memory or not args.say: parser.error("use --benchmark OUTPUT or --memory FILE --say TEXT")
    print(json.dumps(GroundedLanguageBridge.load(args.memory).interpret(args.say).to_json(), indent=2))


if __name__ == "__main__": main()
