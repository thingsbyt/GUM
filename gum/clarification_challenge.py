"""Grounded reference clarification from pixels and demonstrations.

The learner is not given a color or shape dictionary.  It sees rendered
objects, hears phrases paired with a pointed-to object, and discovers which
words consistently identify which visual features.  At evaluation time it
must ask a question when a command denotes several visible objects, use the
answer to resolve the reference, and only then act.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageDraw


COLORS = {
    "red": (220, 62, 62),
    "blue": (65, 115, 225),
    "green": (55, 180, 100),
}
SHAPES = ("circle", "square", "triangle")
NULL_WORDS = {"approach", "choose", "find", "object", "one", "please", "the"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


@dataclass(frozen=True)
class ObjectSpec:
    color: str
    shape: str
    center: tuple[int, int]


@dataclass(frozen=True)
class PerceivedObject:
    local_id: str
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    features: tuple[str, ...]


@dataclass(frozen=True)
class Turn:
    status: str
    response: str
    target_id: str | None = None
    candidates: tuple[str, ...] = ()
    unknown_words: tuple[str, ...] = ()

    def to_json(self) -> dict:
        return asdict(self)


def render_scene(objects: list[ObjectSpec], size: int = 96) -> np.ndarray:
    """Render a scene.  The learner receives only the returned pixels."""
    image = Image.new("RGB", (size, size), (8, 11, 18)); draw = ImageDraw.Draw(image)
    for row in objects:
        x, y = row.center; color = COLORS[row.color]; radius = 8
        if row.shape == "circle":
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        elif row.shape == "square":
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)
        elif row.shape == "triangle":
            draw.polygon(((x, y - radius), (x - radius, y + radius),
                          (x + radius, y + radius)), fill=color)
        else:
            raise ValueError(f"unknown test shape: {row.shape}")
    return np.asarray(image, dtype=np.uint8)


def _components(frame: np.ndarray) -> list[PerceivedObject]:
    """Extract anonymous object features from pixels, not simulator labels."""
    background = np.array((8, 11, 18), dtype=np.uint8)
    mask = np.any(frame != background, axis=2); height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool); raw = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]; seen[y, x] = True; pixels = []
            while stack:
                px, py = stack.pop(); pixels.append((px, py))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = px + dx, py + dy
                    if (0 <= nx < width and 0 <= ny < height and mask[ny, nx]
                            and not seen[ny, nx]):
                        seen[ny, nx] = True; stack.append((nx, ny))
            xs = [p[0] for p in pixels]; ys = [p[1] for p in pixels]
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
            area = len(pixels); box_area = (right - left + 1) * (bottom - top + 1)
            fill = area / box_area
            # Names remain opaque.  Natural-language labels are learned later.
            geometry = "form:high" if fill > .90 else ("form:mid" if fill > .62 else "form:low")
            rgb = tuple(int(v) for v in frame[ys[0], xs[0]])
            appearance = f"appearance:{rgb[0]}-{rgb[1]}-{rgb[2]}"
            center = (round(sum(xs) / area), round(sum(ys) / area))
            raw.append((center, (left, top, right, bottom), (appearance, geometry)))
    raw.sort(key=lambda row: (row[0][1], row[0][0]))
    return [PerceivedObject(f"object-{index}", center, bbox, features)
            for index, (center, bbox, features) in enumerate(raw)]


class VisualWordGrounder:
    """Cross-situational word learner over anonymous visual features."""
    format = "gum-visual-word-grounder-v1"

    def __init__(self):
        self.examples: list[tuple[list[str], tuple[str, ...]]] = []
        self.word_to_feature: dict[str, str] = {}
        self.feature_to_word: dict[str, str] = {}
        self.scores: dict[str, float] = {}

    def demonstrate(self, phrase: str, frame: np.ndarray, point: tuple[int, int]) -> None:
        objects = _components(frame)
        selected = [row for row in objects if row.bbox[0] <= point[0] <= row.bbox[2]
                    and row.bbox[1] <= point[1] <= row.bbox[3]]
        if len(selected) != 1:
            raise ValueError("demonstration point must identify exactly one visible object")
        self.examples.append((_tokens(phrase), selected[0].features))

    def learn(self) -> None:
        if not self.examples:
            raise RuntimeError("visual word learning needs demonstrations")
        feature_base = Counter(feature for _, features in self.examples for feature in features)
        word_count = Counter(word for words, _ in self.examples for word in set(words))
        joint: dict[str, Counter] = defaultdict(Counter)
        for words, features in self.examples:
            for word in set(words):
                joint[word].update(features)
        total = len(self.examples); learned = {}
        for word, count in word_count.items():
            ranked = []
            for feature, together in joint[word].items():
                conditional = together / count; prior = feature_base[feature] / total
                lift = conditional / max(prior, 1e-12)
                ranked.append((conditional * math.log2(max(lift, 1e-12)), conditional, lift, feature))
            ranked.sort(reverse=True)
            score, conditional, lift, feature = ranked[0]
            if count >= 3 and conditional >= .80 and lift >= 2.0 and word not in NULL_WORDS:
                learned[word] = feature; self.scores[word] = float(score)
        self.word_to_feature = learned
        reverse = {}
        for word, feature in learned.items():
            if feature not in reverse or self.scores[word] > self.scores[reverse[feature]]:
                reverse[feature] = word
        self.feature_to_word = reverse

    def constraints(self, phrase: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        constraints = []; unknown = []
        observed_words = {word for words, _ in self.examples for word in words}
        for word in _tokens(phrase):
            if word in self.word_to_feature:
                constraints.append(self.word_to_feature[word])
            elif word not in observed_words and word not in NULL_WORDS:
                unknown.append(word)
        return tuple(dict.fromkeys(constraints)), tuple(dict.fromkeys(unknown))

    def save(self, path: Path) -> None:
        value = {"format": self.format, "examples": self.examples,
                 "word_to_feature": self.word_to_feature,
                 "feature_to_word": self.feature_to_word, "scores": self.scores}
        Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != cls.format:
            raise ValueError("unsupported visual lexicon")
        model = cls()
        model.examples = [(list(words), tuple(features)) for words, features in value["examples"]]
        model.word_to_feature = dict(value["word_to_feature"])
        model.feature_to_word = dict(value["feature_to_word"])
        model.scores = {word: float(score) for word, score in value["scores"].items()}
        return model


class ClarifyingAgent:
    """Resolve references conservatively; ask rather than choose arbitrarily."""
    def __init__(self, lexicon: VisualWordGrounder):
        self.lexicon = lexicon; self.pending: tuple[PerceivedObject, ...] = ()

    def _question(self, candidates: list[PerceivedObject]) -> Turn:
        dimensions = defaultdict(list)
        for row in candidates:
            for feature in row.features:
                dimensions[feature.split(":", 1)[0]].append(feature)
        choices = []
        for dimension, values in dimensions.items():
            unique = sorted(set(values))
            labels = [self.lexicon.feature_to_word.get(value) for value in unique]
            if len(unique) > 1 and all(labels):
                counts = Counter(values); entropy = -sum((n / len(values)) * math.log2(n / len(values))
                                                        for n in counts.values())
                choices.append((entropy, dimension, unique, labels))
        if not choices:
            self.pending = ()
            return Turn("clarify", "I can see several matches, but I do not know words that distinguish them.",
                        candidates=tuple(row.local_id for row in candidates))
        _, _, _, labels = max(choices)
        self.pending = tuple(candidates)
        alternatives = " or ".join(f"the {word}" for word in labels)
        return Turn("clarify", f"Which one do you mean: {alternatives}?",
                    candidates=tuple(row.local_id for row in candidates))

    def _interpret(self, phrase: str, pool: list[PerceivedObject]) -> Turn:
        constraints, unknown = self.lexicon.constraints(phrase)
        if unknown:
            self.pending = ()
            quoted = ", ".join(f"'{word}'" for word in unknown)
            return Turn("clarify", f"I have not grounded {quoted}. Please show me what it means.",
                        unknown_words=unknown)
        candidates = [row for row in pool if all(feature in row.features for feature in constraints)]
        if not constraints:
            self.pending = ()
            return Turn("clarify", "I understood the request, but not which visible object you mean.")
        if not candidates:
            self.pending = ()
            return Turn("clarify", "Nothing I can see matches that description.")
        if len(candidates) == 1:
            self.pending = ()
            target = candidates[0]
            return Turn("execute", f"Understood. I will approach {target.local_id}.", target.local_id,
                        (target.local_id,))
        return self._question(candidates)

    def begin(self, frame: np.ndarray, instruction: str) -> Turn:
        self.pending = ()
        return self._interpret(instruction, _components(frame))

    def answer(self, phrase: str) -> Turn:
        if not self.pending:
            return Turn("clarify", "I do not have an unresolved reference right now.")
        pool = list(self.pending)
        return self._interpret(phrase, pool)


def teach_visual_words(model: VisualWordGrounder) -> None:
    """Provide pointing demonstrations; no word-to-feature table is passed in."""
    centers = ((48, 48),)
    for color in COLORS:
        for shape in SHAPES:
            scene = render_scene([ObjectSpec(color, shape, centers[0])])
            for phrase in (f"approach the {color} {shape}",
                           f"please choose the {color} {shape}"):
                model.demonstrate(phrase, scene, centers[0])
    model.learn()


def _scene_for_case(rng: random.Random, mode: str):
    colors = list(COLORS); shapes = list(SHAPES)
    if mode == "color":
        shared = rng.choice(colors); left, right = rng.sample(shapes, 2)
        specs = [ObjectSpec(shared, left, (25, 30)), ObjectSpec(shared, right, (70, 30)),
                 ObjectSpec(rng.choice([c for c in colors if c != shared]), rng.choice(shapes), (48, 70))]
        prompt = f"approach the {shared} object"; answer = f"the {right}"; target = specs[1]
    elif mode == "shape":
        shared = rng.choice(shapes); left, right = rng.sample(colors, 2)
        specs = [ObjectSpec(left, shared, (25, 30)), ObjectSpec(right, shared, (70, 30)),
                 ObjectSpec(rng.choice(colors), rng.choice([s for s in shapes if s != shared]), (48, 70))]
        prompt = f"approach the {shared}"; answer = f"the {right}"; target = specs[1]
    else:
        target = ObjectSpec(rng.choice(colors), rng.choice(shapes), (70, 30))
        third_options = [(color, shape) for color in colors for shape in shapes
                         if (color, shape) != (target.color, target.shape)]
        third_color, third_shape = rng.choice(third_options)
        specs = [target, ObjectSpec(rng.choice([c for c in colors if c != target.color]),
                                    rng.choice([s for s in shapes if s != target.shape]), (25, 30)),
                 ObjectSpec(third_color, third_shape, (48, 70))]
        prompt = f"approach the {target.color} {target.shape}"; answer = ""
    return specs, prompt, answer, target


def _target_id(frame: np.ndarray, spec: ObjectSpec) -> str:
    return min(_components(frame), key=lambda row: abs(row.center[0] - spec.center[0])
               + abs(row.center[1] - spec.center[1])).local_id


def run_challenge(output_dir: Path, seed: int = 73_200_019, ambiguous_trials: int = 30,
                  clear_trials: int = 15) -> dict:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    lexicon = VisualWordGrounder(); teach_visual_words(lexicon)
    memory = output_dir / "CLARIFICATION_LANGUAGE_MEMORY.json"; lexicon.save(memory)
    lexicon = VisualWordGrounder.load(memory); agent = ClarifyingAgent(lexicon)
    rng = random.Random(seed); baseline_rng = random.Random(seed ^ 0x5A17); rows = []
    for index in range(ambiguous_trials):
        mode = "color" if index % 2 == 0 else "shape"
        specs, prompt, answer, target = _scene_for_case(rng, mode); frame = render_scene(specs)
        first = agent.begin(frame, prompt); second = agent.answer(answer)
        expected = _target_id(frame, target)
        blind_guess = baseline_rng.choice(first.candidates) if first.candidates else None
        rows.append({"kind": "ambiguous", "prompt": prompt, "question": first.response,
                     "answer": answer, "first_status": first.status, "final": second.to_json(),
                     "expected_target": expected,
                     "blind_guess_success": blind_guess == expected,
                     "useful_question": first.status == "clarify" and len(first.candidates) == 2,
                     "success": second.status == "execute" and second.target_id == expected})
    for _ in range(clear_trials):
        specs, prompt, answer, target = _scene_for_case(rng, "clear"); frame = render_scene(specs)
        result = agent.begin(frame, prompt); expected = _target_id(frame, target)
        rows.append({"kind": "clear", "prompt": prompt, "question": None, "answer": answer,
                     "first_status": result.status, "final": result.to_json(),
                     "expected_target": expected, "useful_question": False,
                     "success": result.status == "execute" and result.target_id == expected})
    unknown = agent.begin(render_scene([ObjectSpec("red", "circle", (48, 48))]),
                          "approach the crimson object")
    ambiguous = [row for row in rows if row["kind"] == "ambiguous"]
    clear = [row for row in rows if row["kind"] == "clear"]
    successes = sum(row["success"] for row in rows)
    report = {
        "format": "gum-grounded-clarification-challenge-v1",
        "protocol": {"seed": seed, "ambiguous_trials": ambiguous_trials,
                     "clear_trials": clear_trials, "preinstalled_word_dictionary": False,
                     "language_model_used": "none",
                     "learner_input": "rendered pixels, pointing demonstrations, utterances, answers",
                     "action_family": "approach is fixed; visual reference meanings are learned",
                     "blind_guess_baseline_on_ambiguous_trials": 0.5},
        "results": {"overall_successes": successes, "overall_trials": len(rows),
                    "ambiguous_resolved": sum(row["success"] for row in ambiguous),
                    "ambiguous_trials": len(ambiguous),
                    "blind_guess_successes": sum(row["blind_guess_success"] for row in ambiguous),
                    "blind_guess_trials": len(ambiguous),
                    "useful_questions": sum(row["useful_question"] for row in ambiguous),
                    "clear_executed_without_question": sum(row["success"] for row in clear),
                    "clear_trials": len(clear),
                    "unknown_word_refused": unknown.status == "clarify" and
                                            "crimson" in unknown.unknown_words,
                    "learned_words": len(lexicon.word_to_feature)},
        "learned_lexicon": lexicon.word_to_feature,
        "unknown_case": unknown.to_json(), "sample_dialogues": rows[:8],
        "integrity": {"implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "memory_sha256": hashlib.sha256(memory.read_bytes()).hexdigest()},
    }
    result = report["results"]
    result["agent_gain_over_blind_guess"] = (
        result["ambiguous_resolved"] - result["blind_guess_successes"]) / max(1, result["blind_guess_trials"])
    report["passed"] = bool(result["overall_successes"] == result["overall_trials"]
        and result["useful_questions"] == result["ambiguous_trials"]
        and result["clear_executed_without_question"] == result["clear_trials"]
        and result["unknown_word_refused"]
        and result["ambiguous_resolved"] > result["blind_guess_successes"])
    (output_dir / "CLARIFICATION_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    transcript = ["GUM GROUNDED CLARIFICATION TEST", ""]
    for row in rows[:6]:
        transcript.append(f"HUMAN: {row['prompt']}")
        if row["question"]:
            transcript.extend((f"GUM: {row['question']}", f"HUMAN: {row['answer']}"))
        transcript.extend((f"GUM: {row['final']['response']}",
                           f"RESULT: {'success' if row['success'] else 'failure'}", ""))
    transcript.extend(("HUMAN: approach the crimson object", f"GUM: {unknown.response}"))
    (output_dir / "TRANSCRIPT.txt").write_text("\n".join(transcript) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73_200_019)
    args = parser.parse_args(argv); report = run_challenge(args.output, args.seed)
    print(json.dumps({"results": report["results"], "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
