"""Task-independent online exploration, causal modeling, and graph planning.

This mechanism is deliberately ignorant of any particular game.  It consumes
categorical pixels, an action set, sparse progress rewards, and episode endings.
It builds its state abstraction and transition graph online, searches for
untested interventions, propagates sparse outcomes through experience, and
retains successful visual-state/action associations for later attempts.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .hypothesis_engine import CausalHypothesisEngine
from .concept_memory import ConceptMemory, concept_state_key, object_concept
from .causal_discovery import CausalDiscoveryEngine
from .visual_goal_planner import VisualGoalPlanner
from .predictive_search import PredictiveSearchPlanner
from .event_options import EventOptionPlanner
from .semantic_world_model import SharedSemanticWorldModel
from .semantic_composer import SemanticSkillComposer


def compose_layers(frame_layers) -> np.ndarray:
    layers = np.asarray(frame_layers, dtype=np.uint8)
    if layers.ndim == 2: layers = layers[None]
    if layers.ndim != 3: raise ValueError(f"expected 2-D layers, got {layers.shape}")
    composed = np.zeros(layers.shape[1:], dtype=np.uint8)
    for layer in layers: composed = np.where(layer != 0, layer, composed)
    return composed


def _background_color(grid: np.ndarray) -> int:
    """Infer empty space from the dominant rendered color, never a fixed ID."""
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    """Generic connected-component description with no game-specific labels."""
    height, width = grid.shape; seen = np.zeros_like(grid, dtype=bool); result = []
    background = _background_color(grid)
    for y in range(height):
        for x in range(width):
            color = int(grid[y, x])
            if color == background or seen[y, x]: continue
            queue = [(y, x)]; seen[y, x] = True; points = []
            while queue:
                yy, xx = queue.pop(); points.append((yy, xx))
                for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                    ny, nx = yy + dy, xx + dx
                    if (0 <= ny < height and 0 <= nx < width and not seen[ny, nx]
                            and int(grid[ny, nx]) == color):
                        seen[ny, nx] = True; queue.append((ny, nx))
            if len(points) < 2: continue
            ys = [p[0] for p in points]; xs = [p[1] for p in points]
            # Position is intentionally quantized: tiny animation and sensor
            # changes should not create an entirely new causal state.
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            # Border-attached decorations, meters, and status bars are common
            # sources of irrelevant animation in visual environments.  Keep
            # small/medium interior objects as the reusable causal state.
            if (cx < .04 * width or cx > .96 * width or cy < .04 * height or
                    cy > .86 * height or len(points) > height * width // 8):
                continue
            result.append((color, min(15, int(math.log2(len(points))) if points else 0),
                           int(round(cx / 4)), int(round(cy / 4)),
                           min(15, (max(xs) - min(xs) + 1) // 2),
                           min(15, (max(ys) - min(ys) + 1) // 2)))
    return sorted(result)


def _objects_exact(grid: np.ndarray) -> list[dict]:
    """Return small interior objects with exact geometry for causal tracking."""
    height, width = grid.shape; seen = np.zeros_like(grid, dtype=bool); result = []
    background = _background_color(grid)
    for y in range(height):
        for x in range(width):
            color = int(grid[y, x])
            if color == background or seen[y, x]: continue
            queue = [(y, x)]; seen[y, x] = True; points = []
            while queue:
                yy, xx = queue.pop(); points.append((yy, xx))
                for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                    ny, nx = yy + dy, xx + dx
                    if (0 <= ny < height and 0 <= nx < width and not seen[ny, nx]
                            and int(grid[ny, nx]) == color):
                        seen[ny, nx] = True; queue.append((ny, nx))
            ys = np.asarray([p[0] for p in points]); xs = np.asarray([p[1] for p in points])
            cx, cy = float(xs.mean()), float(ys.mean()); area = len(points)
            if (area < 1 or area > height * width // 8 or cx < .04 * width or cx > .96 * width
                    or cy < .04 * height or cy > .86 * height): continue
            w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            result.append({'color': color, 'area': area, 'cx': cx, 'cy': cy, 'w': w, 'h': h,
                           'sig': (color, min(15, int(math.log2(area))), min(15, w), min(15, h))})
    return result


def state_key(frame_layers) -> str:
    """A noise-tolerant, object-centric identity learned directly from pixels."""
    grid = compose_layers(frame_layers)
    components = _components(grid)
    descriptor = {"shape": list(grid.shape), "components": components}
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def visual_context_key(frame_layers) -> str:
    """A stable scene-family fingerprint used to isolate exploration budgets.

    It intentionally omits absolute object positions, so a moving avatar or a
    changed random seed does not manufacture a new world.  Unlike the abstract
    concept key, it retains coarse palette, multiplicity, and texture evidence;
    visually unrelated worlds therefore cannot accidentally share the same
    "already understood" flag merely because both lack small interior objects.
    """
    grid = compose_layers(frame_layers)
    values, counts = np.unique(grid, return_counts=True)
    palette = [(int(value), int(round(math.log2(int(count) + 1))),
                int(round(64 * int(count) / grid.size)))
               for value, count in zip(values, counts)]
    morphology = [(color, area_bin, width_bin, height_bin)
                  for color, area_bin, _x, _y, width_bin, height_bin in _components(grid)]
    horizontal = int(np.count_nonzero(grid[:, 1:] != grid[:, :-1]))
    vertical = int(np.count_nonzero(grid[1:, :] != grid[:-1, :]))
    descriptor = {"shape": list(grid.shape), "palette": palette,
                  "morphology": sorted(morphology),
                  "texture": [int(round(math.log2(horizontal + 1))),
                              int(round(math.log2(vertical + 1)))]}
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


class UniversalExplorer:
    """Active learner shared unchanged across arbitrary discrete visual worlds."""
    format = "wailah-universal-explorer-v1"

    def __init__(self, seed: int = 99173):
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.context_rngs = {}
        self.q = defaultdict(dict)
        self.visits = defaultdict(Counter)
        self.transitions = defaultdict(Counter)
        self.global_actions = Counter()
        self.state_visits = Counter()
        self.bad = Counter()
        self.success_policy = defaultdict(dict)
        self.level = 0; self.segment = []; self.episode = []
        self.total_steps = 0; self.progress_events = 0; self.model_predictions = 0
        self.model_correct = 0; self.contradictions = 0
        self.episode_index = -1; self.macro_counter = 0
        self.macro_action = None; self.macro_remaining = 0
        self.motion = defaultdict(lambda: defaultdict(list))
        self.controlled_sig = None; self.controlled_position = None
        self.passable_colors = Counter(); self.object_plan_steps = 0
        self.last_object_target = None; self.last_object_mode = None
        self.active_target = None; self.active_target_steps = 0; self.tried_targets = set()
        self.subgoal_events = 0
        self.hypothesis_engine = CausalHypothesisEngine()
        self.causal_discovery = CausalDiscoveryEngine()
        self.visual_goal_planner = VisualGoalPlanner()
        self.predictive_search = PredictiveSearchPlanner()
        self.event_options = EventOptionPlanner()
        # Unlike the world-specific modules below, this semantic causal memory
        # is deliberately shared.  It transfers event structure, never action
        # numbers, coordinates, colors, or object identities.
        self.semantic_world_model = SharedSemanticWorldModel()
        self.semantic_composer = SemanticSkillComposer()
        self.concept_memory = ConceptMemory()
        self.click_visits = Counter(); self.click_effect_sum = Counter(); self.click_effect_count = Counter()
        self.pending_click = None; self.click_events = 0
        self.click_hazard = Counter(); self.click_success = Counter(); self.click_recent = []
        self.click_spawn_credit = Counter()
        self.click_chain_cooldown = 0
        self.episode_effectful_instances = set()
        self.episode_action_counts = Counter(); self.episode_progress = False
        self.successful_clicks = defaultdict(list); self.exploration_regime = 'model-based'
        self.context_progress = Counter(); self.context_episodes = Counter(); self.current_context = 'default'
        self.context_modules = {}; self.active_context = None
        self.last_action_data = None

    _MODULE_FIELDS = (
        "q", "visits", "transitions", "global_actions", "state_visits", "bad",
        "success_policy", "macro_counter", "motion", "controlled_sig",
        "controlled_position", "passable_colors", "object_plan_steps",
        "last_object_target", "last_object_mode", "hypothesis_engine", "causal_discovery",
        "visual_goal_planner", "predictive_search", "event_options",
        "concept_memory", "click_visits", "click_effect_sum", "click_effect_count",
        "click_events", "click_hazard", "click_success", "click_spawn_credit",
        "successful_clicks")

    def _capture_module(self) -> dict:
        return {name: getattr(self, name) for name in self._MODULE_FIELDS}

    @staticmethod
    def _fresh_module() -> dict:
        return {
            "q": defaultdict(dict), "visits": defaultdict(Counter),
            "transitions": defaultdict(Counter), "global_actions": Counter(),
            "state_visits": Counter(), "bad": Counter(),
            "success_policy": defaultdict(dict), "macro_counter": 0,
            "motion": defaultdict(lambda: defaultdict(list)), "controlled_sig": None,
            "controlled_position": None, "passable_colors": Counter(),
            "object_plan_steps": 0, "last_object_target": None,
            "last_object_mode": None, "hypothesis_engine": CausalHypothesisEngine(),
            "causal_discovery": CausalDiscoveryEngine(),
            "visual_goal_planner": VisualGoalPlanner(),
            "predictive_search": PredictiveSearchPlanner(),
            "event_options": EventOptionPlanner(),
            "concept_memory": ConceptMemory(), "click_visits": Counter(),
            "click_effect_sum": Counter(), "click_effect_count": Counter(),
            "click_events": 0, "click_hazard": Counter(), "click_success": Counter(),
            "click_spawn_credit": Counter(),
            "successful_clicks": defaultdict(list)}

    def _switch_module(self, context: str) -> None:
        if self.active_context == context: return
        if self.active_context is not None:
            self.context_modules[self.active_context] = self._capture_module()
        module = self.context_modules.get(context)
        if module is None:
            module = self._fresh_module(); self.context_modules[context] = module
        for name, value in module.items(): setattr(self, name, value)
        self.active_context = context

    @staticmethod
    def _sa(level: int, state: str, action: int) -> str:
        return f"{level}:{state}:{action}"

    @staticmethod
    def _ls(level: int, state: str) -> str:
        return f"{level}:{state}"

    def reset_episode(self, level: int = 0, initial_frame=None) -> None:
        self.level = int(level); self.segment = []; self.episode = []
        self.macro_action = None; self.macro_remaining = 0
        self.active_target = None; self.active_target_steps = 0; self.tried_targets = set()
        self.subgoal_events = 0
        self.pending_click = None; self.click_recent = []
        self.click_chain_cooldown = 0
        self.episode_effectful_instances = set()
        self.episode_action_counts = Counter(); self.episode_progress = False
        if initial_frame is not None:
            context = visual_context_key(initial_frame)
        else:
            context = 'default'
        self._switch_module(context); self.current_context = context
        self.hypothesis_engine.reset_episode()
        self.causal_discovery.reset_episode()
        self.visual_goal_planner.reset_episode()
        self.predictive_search.reset_episode()
        self.event_options.reset_episode()
        self.semantic_world_model.reset_episode(self.current_context)
        self.semantic_composer.reset_episode(self.current_context)
        self.concept_memory.reset_episode()
        # Independent context streams make a continual learner's exploration
        # reproducible in a novel world. Experience elsewhere must not change
        # its luck simply by consuming values from one global generator.
        if self.current_context not in self.context_rngs:
            # Every novel context receives the same broad curriculum rather
            # than arbitrary luck determined by its hash. The streams remain
            # independent, so practice in A cannot advance B's random state.
            self.context_rngs[self.current_context] = np.random.default_rng(self.base_seed)
        self.rng = self.context_rngs[self.current_context]
        context_episode = self.context_episodes[self.current_context]
        self.context_episodes[self.current_context] += 1
        self.episode_index = context_episode
        self.exploration_regime = ('broad-random' if self.context_progress[self.current_context] == 0
                                   and context_episode % 2 == 1
                                   else 'model-based')

    def set_visual_goal(self, goal_frame) -> list[str]:
        """Task the agent with a visual symbolic goal it has grounded itself."""
        return self.semantic_composer.set_goal(goal_frame)

    def _point_action(self, frame_layers, broad_random: bool = False) -> dict[str, int]:
        """Choose ACTION6 coordinates from visual objects plus systematic coverage."""
        grid = compose_layers(frame_layers); state = state_key(frame_layers)
        if broad_random:
            x, y = int(self.rng.integers(64)), int(self.rng.integers(64))
            point_key = f"{state}:{x}:{y}"
            self.pending_click = {'x': x, 'y': y, 'concept': f'random:{x//8}:{y//8}',
                                  'state': state, 'point_key': point_key}
            return {'x': x, 'y': y}
        if self.successful_clicks[state]:
            x, y = self.successful_clicks[state][-1]
            point_key = f"{state}:{x}:{y}"
            self.pending_click = {'x': x, 'y': y, 'concept': 'recalled-progress-point',
                                  'state': state, 'point_key': point_key}
            return {'x': x, 'y': y}
        candidates = []
        for obj in _objects_exact(grid):
            concept = object_concept(obj)
            x = int(round((obj['cx'] + .5) * 64 / grid.shape[1] - .5))
            y = int(round((obj['cy'] + .5) * 64 / grid.shape[0] - .5))
            candidates.append((x, y, concept,
                               self.concept_memory.target_priority(obj)[0]
                               + math.log2(obj['area'] + 1)))
        # A lattice guarantees coverage when segmentation misses a large board,
        # blank cell, or border-attached control.
        for y in range(4, 64, 8):
            for x in range(4, 64, 8):
                candidates.append((x, y, f"cell:{x//8}:{y//8}", 0.0))
        scored = []
        for x, y, concept, role_score in candidates:
            x = max(0, min(63, int(x))); y = max(0, min(63, int(y)))
            key = f"{state}:{concept}:{x//3}:{y//3}"
            point_key = f"{state}:{x}:{y}"
            instance_key = self._click_instance_key(concept, x, y)
            visits = self.click_visits[key]
            effect = self.click_effect_sum[concept] / max(1, self.click_effect_count[concept])
            # The old fractional measure made a decisive 20-pixel object
            # interaction look nearly identical to a one-pixel animation.
            # Convert it back to changed pixels so the scale is independent of
            # frame resolution while retaining backward-compatible memories.
            effect_pixels = grid.size * effect
            score = (4.0 / math.sqrt(1 + visits) + .25 * effect_pixels + .25 * role_score
                     + 8.0 * self.click_success[point_key]
                     - 12.0 * self.click_hazard[point_key]
                     - 100.0 * int(instance_key in self.episode_effectful_instances))
            scored.append((score, -visits, float(self.rng.random()), x, y, concept, key,
                           point_key, instance_key))
        _, _, _, x, y, concept, key, point_key, instance_key = max(scored)
        self.click_visits[key] += 1
        self.pending_click = {'x': x, 'y': y, 'concept': concept, 'state': state,
                              'point_key': point_key, 'instance_key': instance_key}
        return {'x': x, 'y': y}

    @staticmethod
    def _click_instance_key(concept: str, x: float, y: float) -> str:
        return f"{concept}:{int(round(x / 3))}:{int(round(y / 3))}"

    def _effectful_click_available(self, objects: list[dict], grid_size: int) -> bool:
        """Return whether a visible object matches a learned point affordance."""
        visible_effectful = set()
        for obj in objects:
            concept = object_concept(obj)
            count = self.click_effect_count[concept]
            if count and grid_size * self.click_effect_sum[concept] / count >= 4.0:
                visible_effectful.add(concept)
        # A concept disappearing marks the end of one visible sweep. If it
        # later reappears, its instances are new opportunities rather than
        # permanent duplicates of the previous scene.
        tracked_concepts = {key.rsplit(':', 2)[0] for key in self.episode_effectful_instances}
        absent = tracked_concepts - visible_effectful
        if absent:
            self.episode_effectful_instances = {key for key in self.episode_effectful_instances
                                                if key.rsplit(':', 2)[0] not in absent}
        return any(self._click_instance_key(object_concept(obj), obj['cx'], obj['cy'])
                   not in self.episode_effectful_instances
                   for obj in objects if object_concept(obj) in visible_effectful)

    def _spawned_target_available(self, objects: list[dict]) -> bool:
        return any(self.click_spawn_credit[object_concept(obj)] > 0 for obj in objects)

    def _frontier_action(self, level: int, start: str, actions: list[int]) -> int | None:
        """Plan through the learned graph to the nearest untested intervention."""
        root = self._ls(level, start); queue = deque([root]); previous = {root: None}; via = {}
        target = None
        while queue and len(previous) <= 3000:
            node = queue.popleft(); _, state = node.split(":", 1)
            tried = self.visits[node]
            if any(tried[action] == 0 for action in actions): target = node; break
            for action in actions:
                outcomes = self.transitions.get(f"{node}:{action}")
                if not outcomes: continue
                nxt = outcomes.most_common(1)[0][0]
                if nxt not in previous:
                    previous[nxt] = node; via[nxt] = action; queue.append(nxt)
        if target is None or target == root: return None
        cursor = target
        while previous[cursor] != root: cursor = previous[cursor]
        return int(via[cursor])

    def _motion_vectors(self, actions: list[int]) -> dict[int, tuple[int, int]]:
        if self.controlled_sig is None: return {}
        result = {}
        for action in actions:
            values = self.motion[self.controlled_sig].get(action, [])
            if len(values) >= 2:
                dx = int(round(float(np.median([row[0] for row in values[-32:]]))))
                dy = int(round(float(np.median([row[1] for row in values[-32:]]))))
                if dx or dy: result[action] = (dx, dy)
        return result

    def _independent_motion_actions(self, actions: list[int]) -> list[int]:
        """Keep one reliable control for each learned movement direction."""
        vectors = self._motion_vectors(actions); by_direction = {}
        for action, (dx, dy) in vectors.items():
            direction = (int(np.sign(dx)), int(np.sign(dy)))
            values = self.motion[self.controlled_sig].get(action, [])[-32:]
            variability = (float(np.median([abs(row[0] - dx) + abs(row[1] - dy)
                                            for row in values])) if values else float('inf'))
            score = (variability, self.global_actions[action], action)
            if direction not in by_direction or score < by_direction[direction][0]:
                by_direction[direction] = (score, action)
        return [row[1] for _, row in sorted(by_direction.items())]

    def _object_action(self, frame_layers, actions: list[int]) -> int | None:
        """Use learned motor effects to move a controlled object toward a visual peer."""
        vectors = self._motion_vectors(actions)
        # Some worlds expose only one controllable axis while other buttons
        # rotate, fire, or interact. Two independently learned displacements
        # are enough to begin spatial planning; unknown buttons remain under
        # the intervention learner instead of being mislabeled as movement.
        if len(vectors) < min(2, len(actions)) or self.controlled_sig is None: return None
        grid = compose_layers(frame_layers); objects = _objects_exact(grid)
        controlled_candidates = [obj for obj in objects if obj['sig'] == self.controlled_sig]
        if not controlled_candidates: return None
        if self.controlled_position is None: controlled = controlled_candidates[0]
        else:
            controlled = min(controlled_candidates, key=lambda obj:
                (obj['cx'] - self.controlled_position[0]) ** 2 + (obj['cy'] - self.controlled_position[1]) ** 2)
        self.controlled_position = (controlled['cx'], controlled['cy'])
        # Nearby components moving as one object form its visual palette.
        # Destinations sharing that palette are strong hypotheses, while other
        # isolated objects remain possible keys, switches, or prerequisites.
        controlled_palette = {obj['color'] for obj in objects
                              if abs(obj['cx'] - controlled['cx']) + abs(obj['cy'] - controlled['cy']) <= 5}
        peers = [obj for obj in objects if obj is not controlled and obj['area'] <= max(32, controlled['area'] * 4)
                 and abs(obj['cx'] - controlled['cx']) + abs(obj['cy'] - controlled['cy']) > 7]
        if not peers: return None
        # Keep pursuing one hypothesis across frames. If it disappears, a real
        # world event occurred: reopen previously blocked goals under the new state.
        target = None
        if self.active_target is not None:
            matches = [obj for obj in peers if obj['color'] == self.active_target['color'] and
                       abs(obj['cx'] - self.active_target['cx']) + abs(obj['cy'] - self.active_target['cy']) <= 4]
            if matches:
                target = min(matches, key=lambda obj: abs(obj['cx'] - self.active_target['cx']) +
                             abs(obj['cy'] - self.active_target['cy']))
                self.active_target_steps += 1
                if self.active_target_steps > 48:
                    self.tried_targets.add((self.active_target['color'], round(self.active_target['cx']),
                                            round(self.active_target['cy'])))
                    self.active_target = None; self.active_target_steps = 0; target = None
            else:
                vanished = self.active_target
                self.concept_memory.note_contact_attempt(vanished['concept'])
                self.active_target = None; self.active_target_steps = 0; self.tried_targets.clear()
                self.subgoal_events += 1
                # A target often vanishes because the controlled object has
                # visually occluded it just before the environment registers
                # contact.  Perform one final causal probe in the approach
                # direction before selecting a new hypothesis.  This is a
                # generic interaction rule, not a game-specific action.
                distance = abs(controlled['cx'] - vanished['cx']) + abs(controlled['cy'] - vanished['cy'])
                max_step = max(abs(dx) + abs(dy) for dx, dy in vectors.values())
                if distance <= max_step * 2 + max(controlled['w'], controlled['h']):
                    ranked = []
                    for action, (dx, dy) in vectors.items():
                        after = abs(controlled['cx'] + dx - vanished['cx']) + abs(
                            controlled['cy'] + dy - vanished['cy'])
                        ranked.append((after, self.global_actions[action], action))
                    self.object_plan_steps += 1
                    self.last_object_mode = 'contact-probe'
                    self.last_object_target = {
                        'controlled': [controlled['cx'], controlled['cy']],
                        'target': [vanished['cx'], vanished['cy']],
                        'controlled_color': controlled['color'], 'target_color': vanished['color']}
                    return int(min(ranked)[2])
        if target is None:
            eligible = [obj for obj in peers if (obj['color'], round(obj['cx']), round(obj['cy']))
                        not in self.tried_targets]
            if not eligible:
                self.tried_targets.clear(); eligible = peers
            eligible.sort(key=lambda obj: (-self.click_spawn_credit[object_concept(obj)],
                -self.concept_memory.target_priority(obj)[0],
                0 if obj['color'] in controlled_palette else 1,
                abs(int(math.log2(max(1, obj['area']))) - int(math.log2(max(1, controlled['area'])))),
                obj['color'], round(obj['cy'], 1), round(obj['cx'], 1)))
            # Rotate the first hypothesis between episodes, but preserve the
            # salience ordering for subsequent subgoals inside an episode.
            index = (self.episode_index % len(eligible)
                     if not self.tried_targets and self.subgoal_events == 0 else 0)
            target = eligible[index]
            self.active_target = {'color': target['color'], 'cx': target['cx'], 'cy': target['cy'],
                                  'concept': object_concept(target)}
            self.active_target_steps = 1
        start = (int(round(controlled['cy'])), int(round(controlled['cx'])))
        goal = (int(round(target['cy'])), int(round(target['cx'])))
        self.last_object_target = {'controlled': [controlled['cx'], controlled['cy']],
                                   'target': [target['cx'], target['cy']],
                                   'controlled_color': controlled['color'], 'target_color': target['color']}

        # Once movement reveals underlying terrain, search it as a graph using
        # the action effects learned by intervention.  Otherwise take the move
        # that most reduces target distance while calibration continues.
        allowed_colors = {color for color, count in self.passable_colors.items() if count >= 2}
        if allowed_colors:
            free = np.isin(grid, list(allowed_colors)); radius = max(1, max(controlled['w'], controlled['h']) // 2)
            sy, sx = start; gy, gx = goal
            free[max(0, sy-radius):sy+radius+1, max(0, sx-radius):sx+radius+1] = True
            free[max(0, gy-radius):gy+radius+1, max(0, gx-radius):gx+radius+1] = True
            queue = deque([start]); previous = {start: None}; via = {}
            reached = None
            while queue and len(previous) < grid.size:
                point = queue.popleft()
                if abs(point[0] - gy) + abs(point[1] - gx) <= radius + 1:
                    reached = point; break
                for action, (dx, dy) in vectors.items():
                    nxt = (point[0] + dy, point[1] + dx)
                    if (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]
                            and free[nxt] and nxt not in previous):
                        previous[nxt] = point; via[nxt] = action; queue.append(nxt)
            if reached is not None and reached != start:
                cursor = reached
                while previous[cursor] != start: cursor = previous[cursor]
                self.object_plan_steps += 1
                self.last_object_mode = 'terrain-bfs'
                return int(via[cursor])
        ranked = []
        distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        for action, (dx, dy) in vectors.items():
            after = abs(start[0] + dy - goal[0]) + abs(start[1] + dx - goal[1])
            ranked.append((after - distance, self.global_actions[action], action))
        if ranked:
            self.object_plan_steps += 1
            self.last_object_mode = 'greedy-vector'
            return int(min(ranked)[2])
        return None

    def act(self, frame_layers, available_actions, level: int = 0) -> tuple[int, dict]:
        actions = sorted(int(x) for x in available_actions if int(x) != 0)
        if not actions: raise RuntimeError("no usable action")
        level = int(level); state = state_key(frame_layers); node = self._ls(level, state)
        self.level = level; self.state_visits[node] += 1

        remembered = self.success_policy[level].get(state)
        movement_actions = set(self._motion_vectors(actions))
        spatial_ready = self.controlled_sig is not None and len(movement_actions) >= min(2, len(actions))
        mode, experiment_action = self.hypothesis_engine.recommend(
            node, actions, self.total_steps, spatial_ready, movement_actions, self.global_actions)
        discovery = self.causal_discovery.replay(node, actions)
        discovery_data = None
        grid = compose_layers(frame_layers); objects = _objects_exact(grid)
        predictive = self.visual_goal_planner.recommend(
            grid, objects, actions, self.motion, self.passable_colors,
            allow_feature_goal=remembered is None)
        # An ungrounded world model is one hypothesis among several, not the
        # controller. Give it periodic interventions until it predicts real
        # progress; afterward it has earned continuous access to planning.
        model_grounded = self.predictive_search.progress_examples > 0
        search_turn = sum(self.episode_action_counts.values()) % 5 == 0
        search_plan = (self.predictive_search.recommend(grid, actions)
                       if model_grounded or search_turn else None)
        event_plan = self.event_options.recommend(concept_state_key(objects), actions)
        semantic_plan = self.semantic_world_model.recommend(self.current_context, actions)
        composer_plan = self.semantic_composer.recommend(actions)
        repeat_effectful_click = (6 in actions and
                                  self._effectful_click_available(objects, grid.size))
        spawned_action = None
        if self._spawned_target_available(objects) and spatial_ready:
            # A novel object created by a completed interaction chain is a
            # stronger causal lead than an unrelated old contact hypothesis.
            self.active_target = None; self.active_target_steps = 0
            spawned_action = self._object_action(frame_layers, actions)

        if discovery is not None and discovery[2] == "replay-successful-coordinate-procedure":
            choice, discovery_data, reason = discovery
        elif mode == "procedure-replay" and experiment_action in actions:
            choice = int(experiment_action); reason = "replay-validated-procedure"
        elif predictive is not None and predictive[1] == "predict-visual-collision":
            choice, reason = predictive
        elif repeat_effectful_click:
            choice = 6; reason = "repeat-effectful-object-interaction"
            self.macro_action = None; self.macro_remaining = 0
        elif spawned_action is not None:
            choice = spawned_action; reason = "pursue-interaction-created-object"
        elif self.click_chain_cooldown > 0 and (
                movement_choices := self._independent_motion_actions(actions)):
            choice = int(movement_choices[int(self.rng.integers(len(movement_choices)))])
            reason = "advance-after-interaction-chain"
            self.macro_action = None; self.macro_remaining = 0
        elif composer_plan is not None:
            choice, reason = composer_plan
            self.macro_action = None; self.macro_remaining = 0
        elif self.exploration_regime == 'broad-random':
            if discovery is not None:
                choice, discovery_data, reason = discovery
            else:
                choice = self.causal_discovery.experiment_action(node, actions)
                reason = "causal-information-experiment"
            self.hypothesis_engine.last_mode = "graph-frontier"
            self.macro_action = None; self.macro_remaining = 0
        elif semantic_plan is not None:
            choice, reason = semantic_plan
            self.macro_action = None; self.macro_remaining = 0
        elif event_plan is not None:
            choice, discovery_data, reason = event_plan
            self.macro_action = None; self.macro_remaining = 0
        elif search_plan is not None:
            choice, reason = search_plan
            self.macro_action = None; self.macro_remaining = 0
        elif predictive is not None:
            choice, reason = predictive
        elif discovery is not None:
            choice, discovery_data, reason = discovery
        elif self.macro_remaining > 0 and self.macro_action in actions:
            choice = int(self.macro_action); self.macro_remaining -= 1
            self.hypothesis_engine.last_mode = "temporal-persistence"
            reason = "temporal-intervention"
        # Prefer a live causal plan over rote replay.  A coarse visual state can
        # recur at a different physical position or in a changed world, where
        # repeating yesterday's successful button becomes a destructive habit.
        elif mode == "spatial-contact" and (
                (object_action := self._object_action(frame_layers, actions)) is not None):
            choice = object_action; reason = "learned-object-causal-plan"
        elif experiment_action in actions:
            choice = int(experiment_action); reason = f"hypothesis-test:{mode}"
            if mode == "temporal-persistence":
                self.macro_action = choice; self.macro_remaining = 3
                self.hypothesis_engine.action_runs[(choice, 4)] += 1
        elif remembered in actions and self.bad[self._sa(level, state, remembered)] == 0:
            choice = remembered; reason = "replay-progress-causing-policy"
        else:
            untried = [a for a in actions if self.visits[node][a] == 0]
            if untried:
                # Global balancing prevents the initial network bias that caused
                # the frozen brain to press one button forever.
                minimum = min(self.global_actions[a] for a in untried)
                candidates = [a for a in untried if self.global_actions[a] == minimum]
                choice = int(candidates[int(self.rng.integers(len(candidates)))])
                reason = "controlled-intervention"
            else:
                frontier = self._frontier_action(level, state, actions)
                if frontier is not None:
                    choice = frontier; reason = "plan-to-unknown"
                else:
                    total = sum(self.visits[node].values()) + 1
                    scored = []
                    for action in actions:
                        count = self.visits[node][action]
                        value = self.q[node].get(action, 0.0)
                        explore = 1.35 * math.sqrt(math.log(total + 1) / max(1, count))
                        diversity = .20 / math.sqrt(1 + self.global_actions[action])
                        risk = 2.0 * self.bad[self._sa(level, state, action)] / max(1, count)
                        scored.append((value + explore + diversity - risk, action))
                    best = max(score for score, _ in scored)
                    candidates = [a for score, a in scored if abs(score - best) < 1e-9]
                    choice = int(candidates[int(self.rng.integers(len(candidates)))])
                    reason = "value-plus-uncertainty"
            # Discover temporally extended control effects instead of assuming
            # that every meaningful intervention lasts exactly one frame.
            duration = (1, 2, 4, 8, 16)[self.macro_counter % 5]
            self.macro_counter += 1; self.macro_action = choice
            self.macro_remaining = duration - 1
        if (choice == 6 and self.click_chain_cooldown > 0 and not repeat_effectful_click
                and "successful" not in reason and "procedure" not in reason):
            # Once a discovered object interaction has been exhausted, give
            # ordinary controls time to change the scene. This avoids spending
            # the whole episode probing empty or known-ineffective points while
            # still resuming immediately when another matching object appears.
            alternatives = [action for action in actions if action != 6]
            alternatives = self._independent_motion_actions(alternatives) or alternatives
            if alternatives:
                choice = int(alternatives[int(self.rng.integers(len(alternatives)))])
                reason = "advance-after-interaction-chain"
                self.macro_action = None; self.macro_remaining = 0
        if choice != 6 and self.click_chain_cooldown > 0:
            self.click_chain_cooldown -= 1
        # With several legal controls and no success yet, prevent one
        # unvalidated action from consuming the entire episode. Validated
        # procedures, urgent avoidance, and genuinely new effectful objects
        # retain priority.
        episode_total = sum(self.episode_action_counts.values())
        protected = ("procedure" in reason or "successful" in reason or
                     reason == "predict-visual-collision" or
                     reason == "repeat-effectful-object-interaction")
        projected_share = ((self.episode_action_counts[choice] + 1) /
                           max(1, episode_total + 1))
        if (len(actions) > 1 and not self.episode_progress and episode_total >= 12
                and projected_share > .65 and not protected):
            alternatives = [action for action in actions if action != choice]
            if alternatives:
                choice = min(alternatives, key=lambda action:
                             (self.episode_action_counts[action], self.global_actions[action], action))
                reason = "restore-control-diversity"
                self.macro_action = None; self.macro_remaining = 0
        self.episode_action_counts[choice] += 1
        self.global_actions[choice] += 1; self.total_steps += 1
        evidence = {"state": state, "reason": reason,
                    "state_visits": self.state_visits[node],
                    "action_visits": self.visits[node][choice]}
        if choice == 6:
            evidence["action_data"] = (discovery_data or self._point_action(
                frame_layers, broad_random=False))
            evidence["reason"] += ":visual-point"
        self.last_action_data = evidence.get("action_data")
        self.causal_discovery.note_action(node, choice, self.last_action_data)
        return choice, evidence

    def observe(self, frame_layers, action: int, next_frame_layers, reward: float,
                done: bool, level: int = 0, next_level: int | None = None) -> dict:
        level = int(level); next_level = level if next_level is None else int(next_level)
        before_grid, after_grid = compose_layers(frame_layers), compose_layers(next_frame_layers)
        before_objects, after_objects = _objects_exact(before_grid), _objects_exact(after_grid)
        is_progress = float(reward) > 0 or next_level > level
        option_action_data = self.last_action_data
        concept_events = self.concept_memory.observe(
            before_objects, int(action), after_objects, is_progress)
        # Greedily match same-looking objects. Static peers match at distance
        # zero, leaving the displaced object as the causal candidate.
        movements = []
        signatures = {obj['sig'] for obj in before_objects} & {obj['sig'] for obj in after_objects}
        for signature in signatures:
            before_group = [obj for obj in before_objects if obj['sig'] == signature]
            after_group = [obj for obj in after_objects if obj['sig'] == signature]
            unique_track = len(before_group) == 1 and len(after_group) == 1
            pairs = []
            # Repeated goal/agent glyphs can look identical. Match every
            # unchanged instance first, then match the remaining objects by
            # globally smallest displacement; this prevents identity swaps.
            while before_group and after_group:
                choices = [(abs(a['cx'] - b['cx']) + abs(a['cy'] - b['cy']), i, j)
                           for i, b in enumerate(before_group) for j, a in enumerate(after_group)]
                _, i, j = min(choices)
                pairs.append((before_group.pop(i), after_group.pop(j)))
            for before_obj, after_obj in pairs:
                dx = after_obj['cx'] - before_obj['cx']; dy = after_obj['cy'] - before_obj['cy']
                if .4 <= abs(dx) + abs(dy) <= 12:
                    movements.append((before_obj, after_obj, dx, dy))
                    # Repeated identical glyphs create identity swaps, while
                    # one-pixel cursors and animation specks create false
                    # controllers. Require a unique, persistent object with
                    # enough visual support before learning motor causality.
                    if unique_track and before_obj['area'] >= 4:
                        self.motion[before_obj['sig']][int(action)].append((dx, dy))
                    oy, ox = int(round(before_obj['cy'])), int(round(before_obj['cx']))
                    if 0 <= oy < after_grid.shape[0] and 0 <= ox < after_grid.shape[1]:
                        revealed = int(after_grid[oy, ox])
                        if revealed != before_obj['color']: self.passable_colors[revealed] += 1
        # A controllable object's displacement varies consistently with at
        # least two different actions. Autonomous animation does not.
        scored = []
        for signature, by_action in self.motion.items():
            if signature[1] > 6 or signature[2] >= 15 or signature[3] >= 15:
                continue
            vectors = []
            for candidate_action, values in by_action.items():
                if values:
                    vectors.append((candidate_action,
                        round(float(np.median([x[0] for x in values[-32:]])), 1),
                        round(float(np.median([x[1] for x in values[-32:]])), 1)))
            distinct = len({(dx, dy) for _, dx, dy in vectors if abs(dx) + abs(dy) >= .4})
            scored.append((distinct, sum(len(x) for x in by_action.values()), signature))
        if scored and max(scored)[0] >= 2:
            self.controlled_sig = max(scored)[2]
            matching = [after for before, after, _, _ in movements if before['sig'] == self.controlled_sig]
            if matching: self.controlled_position = (matching[0]['cx'], matching[0]['cy'])
        self.visual_goal_planner.observe(before_grid, int(action), after_grid, is_progress,
                                         bool(done), before_objects, after_objects, self.motion)
        self.predictive_search.observe(before_grid, int(action), after_grid,
                                       is_progress, bool(done))
        self.event_options.observe(concept_state_key(before_objects), int(action),
                                   concept_state_key(after_objects), concept_events,
                                   is_progress, bool(done), option_action_data)
        self.semantic_world_model.observe(self.current_context, int(action), concept_events,
                                          is_progress, bool(done))
        self.semantic_composer.observe(int(action), concept_events, is_progress, bool(done))
        state = state_key(frame_layers); nxt = state_key(next_frame_layers)
        node = self._ls(level, state); next_node = self._ls(next_level, nxt)
        edge = self._sa(level, state, int(action))
        outcomes = self.transitions[edge]
        if outcomes:
            self.model_predictions += 1
            predicted = outcomes.most_common(1)[0][0]
            self.model_correct += int(predicted == next_node)
            self.contradictions += int(predicted != next_node)
        outcomes[next_node] += 1; self.visits[node][int(action)] += 1
        new_state = self.state_visits[next_node] == 0
        self.state_visits[next_node] += 1

        extrinsic = float(reward)
        progress = extrinsic > 0 or next_level > level
        changed_fraction = float(np.mean(before_grid != after_grid))
        self.causal_discovery.observe(node, int(action), next_node, before_grid, after_grid,
                                      progress, bool(done))
        self.last_action_data = None
        if int(action) == 6 and self.pending_click is not None:
            click_x = int(round((self.pending_click['x'] + .5) * before_grid.shape[1] / 64 - .5))
            click_y = int(round((self.pending_click['y'] + .5) * before_grid.shape[0] / 64 - .5))
            radius = max(3, int(round(max(before_grid.shape) / 16)))
            y0, y1 = max(0, click_y - radius), min(before_grid.shape[0], click_y + radius + 1)
            x0, x1 = max(0, click_x - radius), min(before_grid.shape[1], click_x + radius + 1)
            local_changed = int(np.count_nonzero(before_grid[y0:y1, x0:x1] !=
                                                 after_grid[y0:y1, x0:x1]))
            concept = self.pending_click['concept']
            effect = local_changed / before_grid.size + (10.0 if progress else 0.0)
            point_key = self.pending_click['point_key']; self.click_recent.append(point_key)
            self.click_recent = self.click_recent[-16:]
            self.click_effect_sum[concept] += effect; self.click_effect_count[concept] += 1
            self.click_events += int(changed_fraction > 0 or progress)
            if local_changed >= 4:
                self.click_chain_cooldown = max(self.click_chain_cooldown, 96)
                instance_key = self.pending_click.get('instance_key')
                if instance_key:
                    self.episode_effectful_instances.add(instance_key)
            # Credit visual concepts that appeared only after the point
            # intervention. This captures keys, exits, rewards, and other
            # delayed products without assigning any semantic label to them.
            before_concepts = Counter(object_concept(obj) for obj in before_objects)
            after_concepts = Counter(object_concept(obj) for obj in after_objects)
            changed_pixels = changed_fraction * before_grid.size
            for spawned_concept, count in (after_concepts - before_concepts).items():
                self.click_spawn_credit[spawned_concept] += changed_pixels * count
            if progress:
                for index, key in enumerate(reversed(self.click_recent)):
                    self.click_success[key] += .92 ** index
                self.successful_clicks[self.pending_click['state']].append(
                    (int(self.pending_click['x']), int(self.pending_click['y'])))
            elif done:
                for index, key in enumerate(reversed(self.click_recent)):
                    self.click_hazard[key] += .78 ** index
        self.pending_click = None; self.last_action_data = None
        self.hypothesis_engine.observe(node, int(action), next_node, changed_fraction,
                                       progress, bool(done))
        if progress:
            self.context_progress[self.current_context] += 1
            self.episode_progress = True
            self.macro_action = None; self.macro_remaining = 0
        shaped = 60.0 * extrinsic + (.15 if new_state else 0.0)
        if done and extrinsic <= 0: shaped -= 6.0
        future = max(self.q[next_node].values(), default=0.0) if not done else 0.0
        old = self.q[node].get(int(action), 0.0)
        self.q[node][int(action)] = old + .35 * (shaped + .97 * future - old)
        row = (level, state, int(action), next_level, nxt)
        self.segment.append(row); self.episode.append(row)

        if progress:
            self.progress_events += 1
            value = 60.0
            for prior_level, prior_state, prior_action, _, _ in reversed(self.segment):
                prior_node = self._ls(prior_level, prior_state)
                current = self.q[prior_node].get(prior_action, 0.0)
                self.q[prior_node][prior_action] = max(current, value)
                self.success_policy[prior_level][prior_state] = prior_action
                value *= .97
            self.segment = []
        elif done:
            value = -6.0
            for prior_level, prior_state, prior_action, _, _ in reversed(self.segment[-24:]):
                prior_node = self._ls(prior_level, prior_state)
                current = self.q[prior_node].get(prior_action, 0.0)
                self.q[prior_node][prior_action] = min(current, value)
                self.bad[self._sa(prior_level, prior_state, prior_action)] += 1
                if self.success_policy[prior_level].get(prior_state) == prior_action:
                    del self.success_policy[prior_level][prior_state]
                value *= .92
            self.segment = []
        return {"new_state": new_state, "progress_events": self.progress_events,
                "model_accuracy": self.model_correct / max(1, self.model_predictions)}

    def status(self) -> dict:
        return {"format": self.format, "steps": self.total_steps,
                "abstract_states": len(self.state_visits), "state_actions": sum(map(len, self.q.values())),
                "transition_edges": len(self.transitions), "progress_events": self.progress_events,
                "model_predictions": self.model_predictions,
                "model_accuracy": self.model_correct / max(1, self.model_predictions),
                "contradictions": self.contradictions,
                "remembered_progress_states": sum(len(x) for x in self.success_policy.values()),
                "global_action_counts": dict(self.global_actions),
                "controlled_signature": list(self.controlled_sig) if self.controlled_sig else None,
                "learned_action_vectors": {str(action): list(vector) for action, vector in
                                           self._motion_vectors(list(self.global_actions)).items()},
                "passable_colors": dict(self.passable_colors),
                "object_plan_steps": self.object_plan_steps,
                "last_object_target": self.last_object_target,
                "last_object_mode": self.last_object_mode,
                "point_interaction": {"tested_points": len(self.click_visits),
                                      "effectful_clicks": self.click_events,
                                      "learned_click_concepts": len(self.click_effect_count),
                                      "hazardous_points": len(self.click_hazard),
                                      "credited_points": len(self.click_success),
                                      "interaction_created_concepts": len(self.click_spawn_credit),
                                      "remembered_progress_points": sum(map(len, self.successful_clicks.values())),
                                      "exploration_regime": self.exploration_regime},
                "experienced_contexts": len(self.context_episodes),
                "stored_skill_modules": len(self.context_modules),
                "hypothesis_engine": self.hypothesis_engine.status(),
                "causal_discovery": self.causal_discovery.status(),
                "visual_goal_planner": self.visual_goal_planner.status(),
                "predictive_search": self.predictive_search.status(),
                "event_options": self.event_options.status(),
                "semantic_world_model": self.semantic_world_model.status(),
                "semantic_composer": self.semantic_composer.status(),
                "concept_memory": self.concept_memory.status()}

    @staticmethod
    def _export_module(module: dict) -> dict:
        return {
            "q": {node: {str(a): v for a, v in values.items()} for node, values in module["q"].items()},
            "visits": {node: dict(values) for node, values in module["visits"].items()},
            "transitions": {edge: dict(values) for edge, values in module["transitions"].items()},
            "global_actions": dict(module["global_actions"]),
            "state_visits": dict(module["state_visits"]), "bad": dict(module["bad"]),
            "success_policy": {str(level): values for level, values in module["success_policy"].items()},
            "macro_counter": module["macro_counter"],
            "motion": [{"signature": list(signature),
                        "actions": {str(action): values for action, values in rows.items()}}
                       for signature, rows in module["motion"].items()],
            "controlled_sig": list(module["controlled_sig"]) if module["controlled_sig"] else None,
            "controlled_position": module["controlled_position"],
            "passable_colors": dict(module["passable_colors"]),
            "object_plan_steps": module["object_plan_steps"],
            "last_object_target": module["last_object_target"],
            "last_object_mode": module["last_object_mode"],
            "hypothesis_memory": module["hypothesis_engine"].export(),
            "causal_discovery_memory": module["causal_discovery"].export(),
            "visual_goal_planner_memory": module["visual_goal_planner"].export(),
            "predictive_search_memory": module["predictive_search"].export(),
            "event_options_memory": module["event_options"].export(),
            "concept_memory_data": module["concept_memory"].export(),
            "click_visits": dict(module["click_visits"]),
            "click_effect_sum": dict(module["click_effect_sum"]),
            "click_effect_count": dict(module["click_effect_count"]),
            "click_events": module["click_events"],
            "click_hazard": dict(module["click_hazard"]),
            "click_success": dict(module["click_success"]),
            "click_spawn_credit": dict(module["click_spawn_credit"]),
            "successful_clicks": {state: points for state, points in module["successful_clicks"].items()}}

    @classmethod
    def _restore_module(cls, value: dict) -> dict:
        module = cls._fresh_module()
        module["q"] = defaultdict(dict, {node: {int(a): float(v) for a, v in rows.items()}
                                           for node, rows in value.get("q", {}).items()})
        module["visits"] = defaultdict(Counter, {node: Counter({int(a): int(v) for a, v in rows.items()})
                                                  for node, rows in value.get("visits", {}).items()})
        module["transitions"] = defaultdict(Counter, {edge: Counter(rows)
                                                       for edge, rows in value.get("transitions", {}).items()})
        module["global_actions"] = Counter({int(a): int(v)
                                             for a, v in value.get("global_actions", {}).items()})
        module["state_visits"] = Counter(value.get("state_visits", {}))
        module["bad"] = Counter(value.get("bad", {}))
        module["success_policy"] = defaultdict(dict, {
            int(level): {state: int(action) for state, action in rows.items()}
            for level, rows in value.get("success_policy", {}).items()})
        module["macro_counter"] = int(value.get("macro_counter", 0))
        motion = defaultdict(lambda: defaultdict(list))
        for row in value.get("motion", []):
            signature = tuple(int(x) for x in row["signature"])
            for action, vectors in row.get("actions", {}).items():
                motion[signature][int(action)] = [tuple(map(float, vector)) for vector in vectors]
        module["motion"] = motion
        sig = value.get("controlled_sig"); module["controlled_sig"] = tuple(sig) if sig else None
        position = value.get("controlled_position")
        module["controlled_position"] = tuple(position) if position else None
        module["passable_colors"] = Counter({int(k): int(v)
                                               for k, v in value.get("passable_colors", {}).items()})
        module["object_plan_steps"] = int(value.get("object_plan_steps", 0))
        module["last_object_target"] = value.get("last_object_target")
        module["last_object_mode"] = value.get("last_object_mode")
        module["hypothesis_engine"].restore(value.get("hypothesis_memory", {}))
        module["causal_discovery"].restore(value.get("causal_discovery_memory", {}))
        planner_memory = value.get("visual_goal_planner_memory")
        if planner_memory:
            module["visual_goal_planner"].restore(planner_memory)
        search_memory = value.get("predictive_search_memory")
        if search_memory:
            module["predictive_search"].restore(search_memory)
        option_memory = value.get("event_options_memory")
        if option_memory:
            module["event_options"].restore(option_memory)
        module["concept_memory"].restore(value.get("concept_memory_data", {}))
        module["click_visits"] = Counter(value.get("click_visits", {}))
        module["click_effect_sum"] = Counter({k: float(v) for k, v in value.get("click_effect_sum", {}).items()})
        module["click_effect_count"] = Counter({k: int(v) for k, v in value.get("click_effect_count", {}).items()})
        module["click_events"] = int(value.get("click_events", 0))
        module["click_hazard"] = Counter({k: float(v) for k, v in value.get("click_hazard", {}).items()})
        module["click_success"] = Counter({k: float(v) for k, v in value.get("click_success", {}).items()})
        module["click_spawn_credit"] = Counter({k: float(v)
                                                   for k, v in value.get("click_spawn_credit", {}).items()})
        module["successful_clicks"] = defaultdict(list, {
            state: [tuple(map(int, point)) for point in points]
            for state, points in value.get("successful_clicks", {}).items()})
        return module

    def save(self, path: Path) -> None:
        if self.active_context is not None:
            self.context_modules[self.active_context] = self._capture_module()
        value = self.status() | {"base_seed": self.base_seed, "active_context": self.active_context,
            "context_progress": dict(self.context_progress),
            "context_episodes": dict(self.context_episodes),
            "semantic_world_model_data": self.semantic_world_model.export(),
            "semantic_composer_data": self.semantic_composer.export(),
            "context_modules_data": {context: self._export_module(module)
                                     for context, module in self.context_modules.items()},
            "context_rng_states": {context: rng.bit_generator.state
                                   for context, rng in self.context_rngs.items()}}
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "UniversalExplorer":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != cls.format: raise ValueError("unsupported universal explorer format")
        agent = cls(int(value.get("base_seed", 99173)))
        agent.total_steps = int(value.get("steps", 0)); agent.progress_events = int(value.get("progress_events", 0))
        agent.model_predictions = int(value.get("model_predictions", 0))
        agent.model_correct = int(round(float(value.get("model_accuracy", 0.0)) * agent.model_predictions))
        agent.contradictions = int(value.get("contradictions", 0))
        agent.context_progress = Counter({str(k): int(v) for k, v in value.get("context_progress", {}).items()})
        agent.context_episodes = Counter({str(k): int(v) for k, v in value.get("context_episodes", {}).items()})
        semantic_memory = value.get("semantic_world_model_data")
        if semantic_memory:
            agent.semantic_world_model.restore(semantic_memory)
        composer_memory = value.get("semantic_composer_data")
        if composer_memory:
            agent.semantic_composer.restore(composer_memory)
        agent.context_modules = {context: cls._restore_module(module)
                                 for context, module in value.get("context_modules_data", {}).items()}
        agent.context_rngs = {}
        for context, state in value.get("context_rng_states", {}).items():
            rng = np.random.default_rng(agent.base_seed); rng.bit_generator.state = state
            agent.context_rngs[context] = rng
        active = value.get("active_context")
        agent.active_context = None
        if active in agent.context_modules:
            agent._switch_module(active); agent.current_context = active
            agent.rng = agent.context_rngs.get(active, np.random.default_rng(agent.base_seed))
        return agent
