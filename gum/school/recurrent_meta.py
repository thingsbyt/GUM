"""Reward-grounded recurrent policy for anonymous causal-control tasks.

The policy follows the RL-squared information boundary: pixels, the previous
anonymous action, scalar reward, and an episode-boundary flag.  It deliberately
has no input for event labels, hidden world state, action masks, or prescribed
probe order.  Its recurrent state is the only within-episode memory.

PyTorch is an optional GUM dependency, so this module is intentionally not
imported by :mod:`gum.school`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gum.protocol import PublicWorldSpec, Transition


@dataclass(frozen=True)
class RecurrentMetaConfig:
    """Small, CPU-friendly recurrent actor-critic configuration."""

    action_count: int = 8
    pooled_size: int = 8
    visual_width: int = 32
    hidden_size: int = 64
    action_memory_size: int = 16
    learning_rate: float = 1e-3
    discount: float = 0.97
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.02
    gradient_clip: float = 1.0
    batch_episodes: int = 32
    evaluation_temperature: float = 0.25
    training_exploration_mix: float = 0.0
    episodic_action_exploration_mix: float = 0.0
    episodic_novelty_coefficient: float = 0.0

    def validate(self) -> None:
        if self.action_count < 2:
            raise ValueError("action_count must be at least two")
        for name in ("pooled_size", "visual_width", "hidden_size", "action_memory_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if self.value_coefficient < 0.0 or self.entropy_coefficient < 0.0:
            raise ValueError("loss coefficients cannot be negative")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        if self.batch_episodes < 1:
            raise ValueError("batch_episodes must be positive")
        if self.evaluation_temperature <= 0.0:
            raise ValueError("evaluation_temperature must be positive")
        if not 0.0 <= self.training_exploration_mix < 1.0:
            raise ValueError("training_exploration_mix must be in [0, 1)")
        if not 0.0 <= self.episodic_action_exploration_mix <= 1.0:
            raise ValueError("episodic_action_exploration_mix must be in [0, 1]")
        if self.episodic_novelty_coefficient < 0.0:
            raise ValueError("episodic_novelty_coefficient cannot be negative")


class RecurrentMetaPolicy(nn.Module):
    """Compact pixel encoder, recurrent memory, and actor/value heads."""

    def __init__(self, config: RecurrentMetaConfig):
        super().__init__()
        config.validate()
        self.config = config
        pixels = 3 * config.pooled_size * config.pooled_size
        self.visual = nn.Sequential(
            nn.Linear(pixels, config.visual_width),
            nn.Tanh(),
        )
        recurrent_input = config.visual_width + 2
        self.memory = nn.GRUCell(recurrent_input, config.hidden_size)
        self.action_memory = nn.GRUCell(recurrent_input, config.action_memory_size)
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_size + config.action_memory_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1),
        )
        self.critic = nn.Linear(config.hidden_size, 1)

    def encode_pixels(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 3:
            observation = observation.unsqueeze(0)
        if observation.ndim != 4 or observation.shape[-1] != 3:
            raise ValueError("observation must have shape H x W x 3 or B x H x W x 3")
        pixels = observation.to(dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        pooled = F.adaptive_avg_pool2d(
            pixels, (self.config.pooled_size, self.config.pooled_size)
        )
        return self.visual(pooled.flatten(1))

    def forward(
        self,
        observation: torch.Tensor,
        previous_action: torch.Tensor,
        previous_reward: torch.Tensor,
        previous_done: torch.Tensor,
        hidden: torch.Tensor,
        action_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        visual = self.encode_pixels(observation)
        context = torch.cat((visual, previous_reward[:, None], previous_done[:, None]), dim=1)
        hidden = self.memory(context, hidden)
        chosen = previous_action.clamp_max(self.config.action_count - 1).to(torch.int64)
        prior = action_memory.gather(
            1,
            chosen[:, None, None].expand(-1, 1, self.config.action_memory_size),
        ).squeeze(1)
        updated = self.action_memory(context, prior)
        active = (previous_action < self.config.action_count).to(visual.dtype)
        mask = F.one_hot(chosen, self.config.action_count).to(visual.dtype)
        mask = mask[:, :, None] * active[:, None, None]
        action_memory = action_memory * (1.0 - mask) + updated[:, None, :] * mask
        repeated_hidden = hidden[:, None, :].expand(-1, self.config.action_count, -1)
        actor_input = torch.cat((repeated_hidden, action_memory), dim=2)
        logits = self.actor(actor_input).squeeze(2)
        return logits, self.critic(hidden).squeeze(1), hidden, action_memory


class RecurrentCausalLearner:
    """A recurrent actor-critic with a strict public observation boundary."""

    format = "gum-school-recurrent-meta-policy-v1"

    def __init__(
        self,
        seed: int,
        *,
        config: RecurrentMetaConfig | None = None,
        device: str | torch.device = "cpu",
    ):
        self.seed = int(seed)
        self.config = config or RecurrentMetaConfig()
        self.config.validate()
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        self.policy = RecurrentMetaPolicy(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )
        self.training_episodes = 0
        self.training_interactions = 0
        self.evaluation_episodes = 0
        self.evaluation_interactions = 0
        self._spec: PublicWorldSpec | None = None
        self._hidden: torch.Tensor | None = None
        self._action_memory: torch.Tensor | None = None
        self._previous_action = self.config.action_count
        self._previous_reward = 0.0
        self._previous_done = True
        self._log_probs: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._entropies: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self._pending: list[dict[str, Any]] = []
        self._last_confidence = 0.0
        self._training = False
        self._generator = torch.Generator(device="cpu")
        self._episodic_counts: dict[str, int] = {}
        self._episodic_action_counts: dict[str, np.ndarray] = {}
        self._action_effect_changes = np.zeros(self.config.action_count, dtype=np.float64)
        self._action_effect_trials = np.zeros(self.config.action_count, dtype=np.float64)
        self._acted_observation_key: str | None = None
        self.intrinsic_reward_total = 0.0

    @staticmethod
    def _episode_seed(spec: PublicWorldSpec) -> int:
        digest = hashlib.sha256(spec.world_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % (2**63 - 1)

    def begin(self, spec: PublicWorldSpec, observation: Any, *, training: bool) -> None:
        array = np.asarray(observation)
        if not 2 <= spec.action_count <= self.config.action_count:
            raise ValueError("world action count exceeds recurrent policy capacity")
        if tuple(array.shape) != tuple(spec.observation_shape) or array.dtype != np.uint8:
            raise ValueError("observation differs from the public world contract")
        self._spec = spec
        self._hidden = torch.zeros(1, self.config.hidden_size, device=self.device)
        self._action_memory = torch.zeros(
            1,
            self.config.action_count,
            self.config.action_memory_size,
            device=self.device,
        )
        self._previous_action = self.config.action_count
        self._previous_reward = 0.0
        self._previous_done = True
        self._log_probs = []
        self._values = []
        self._entropies = []
        self._rewards = []
        self._training = bool(training)
        self._episodic_counts = {self._observation_key(array): 1}
        self._episodic_action_counts = {}
        self._action_effect_changes.fill(0.0)
        self._action_effect_trials.fill(0.0)
        self._acted_observation_key = None
        # Evaluation remains stochastic but reproducible for each public world.
        self._generator.manual_seed(self._episode_seed(spec))

    def _policy_step(self, observation: Any):
        if self._spec is None or self._hidden is None or self._action_memory is None:
            raise RuntimeError("begin must be called before act")
        array = np.asarray(observation)
        if tuple(array.shape) != tuple(self._spec.observation_shape) or array.dtype != np.uint8:
            raise ValueError("observation differs from the public world contract")
        pixels = torch.from_numpy(array.copy()).to(self.device)
        previous_action = torch.tensor([self._previous_action], device=self.device)
        previous_reward = torch.tensor(
            [self._previous_reward], dtype=torch.float32, device=self.device
        )
        previous_done = torch.tensor(
            [float(self._previous_done)], dtype=torch.float32, device=self.device
        )
        return self.policy(
            pixels,
            previous_action,
            previous_reward,
            previous_done,
            self._hidden,
            self._action_memory,
        )

    @staticmethod
    def _observation_key(observation: np.ndarray) -> str:
        array = np.ascontiguousarray(observation)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        return digest.hexdigest()[:24]

    def act(self, observation: Any, *, training: bool) -> int:
        if bool(training) != self._training:
            raise ValueError("act training mode differs from begin")
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            logits, value, hidden, action_memory = self._policy_step(observation)
            assert self._spec is not None
            available_logits = logits[:, :self._spec.action_count]
            temperature = 1.0 if training else self.config.evaluation_temperature
            probabilities = torch.softmax(available_logits / temperature, dim=-1)
            if training and self.config.training_exploration_mix > 0.0:
                mix = self.config.training_exploration_mix
                probabilities = (
                    (1.0 - mix) * probabilities
                    + mix / self._spec.action_count
                )
            action_mix = self.config.episodic_action_exploration_mix
            if action_mix > 0.0:
                observation_key = self._observation_key(np.asarray(observation))
                counts = self._episodic_action_counts.setdefault(
                    observation_key,
                    np.zeros(self._spec.action_count, dtype=np.int64),
                )
                novelty = 1.0 / np.sqrt(counts.astype(np.float64) + 1.0)
                effect = (
                    self._action_effect_changes[:self._spec.action_count] + 1.0
                ) / (
                    self._action_effect_trials[:self._spec.action_count] + 2.0
                )
                exploration = novelty * np.sqrt(effect)
                exploration /= exploration.sum()
                exploration_tensor = torch.tensor(
                    exploration,
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                )[None, :]
                probabilities = (
                    (1.0 - action_mix) * probabilities
                    + action_mix * exploration_tensor
                )
            if training:
                action_tensor = torch.multinomial(probabilities, 1).squeeze(1)
            else:
                action_tensor = torch.multinomial(
                    probabilities.cpu(), 1, generator=self._generator
                ).squeeze(1).to(self.device)
            log_probabilities = torch.log(
                probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
            )
            log_prob = log_probabilities.gather(1, action_tensor[:, None]).squeeze(1)
            entropy = -(probabilities * log_probabilities).sum(dim=1)
            normalized_entropy = entropy / np.log(self._spec.action_count)
            self._last_confidence = float(
                torch.clamp(1.0 - normalized_entropy, 0.0, 1.0).item()
            )
            self._hidden = hidden
            self._action_memory = action_memory
            if action_mix > 0.0:
                counts[int(action_tensor.item())] += 1
                self._acted_observation_key = observation_key
            if training:
                self._log_probs.append(log_prob.squeeze(0))
                self._values.append(value.squeeze(0))
                self._entropies.append(entropy.squeeze(0))
        return int(action_tensor.item())

    def observe(self, action: Any, transition: Transition, *, training: bool) -> None:
        if self._spec is None:
            raise RuntimeError("begin and act must precede observe")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer")
        action = int(action)
        if not 0 <= action < self._spec.action_count:
            raise ValueError("action is outside the public contract")
        reward = float(transition.reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        # Deliberately do not read transition.public_info.  Learning is grounded
        # only in consequence pixels, scalar reward, action, and termination.
        done = bool(transition.terminated or transition.truncated)
        if (
            self.config.episodic_action_exploration_mix > 0.0
            and self._acted_observation_key is not None
        ):
            next_key = self._observation_key(np.asarray(transition.observation))
            self._action_effect_trials[action] += 1.0
            self._action_effect_changes[action] += float(
                next_key != self._acted_observation_key
            )
        self._previous_action = action
        self._previous_reward = reward
        self._previous_done = done
        if training:
            shaped_reward = reward
            if self.config.episodic_novelty_coefficient > 0.0:
                array = np.asarray(transition.observation)
                if tuple(array.shape) != tuple(self._spec.observation_shape):
                    raise ValueError("transition observation differs from the public contract")
                key = self._observation_key(array)
                visits = self._episodic_counts.get(key, 0) + 1
                self._episodic_counts[key] = visits
                intrinsic = (
                    self.config.episodic_novelty_coefficient
                    if visits == 1 else 0.0
                )
                shaped_reward += intrinsic
                self.intrinsic_reward_total += intrinsic
            self._rewards.append(shaped_reward)
            self.training_interactions += 1
        else:
            self.evaluation_interactions += 1

    def finish_episode(self, *, training: bool) -> dict[str, float] | None:
        if bool(training) != self._training:
            raise ValueError("finish training mode differs from begin")
        if not training:
            self.evaluation_episodes += 1
            return None
        if not self._rewards or len(self._rewards) != len(self._log_probs):
            raise RuntimeError("training trajectory is incomplete")
        self._pending.append({
            "log_probs": torch.stack(self._log_probs),
            "values": torch.stack(self._values),
            "entropies": torch.stack(self._entropies),
            "rewards": list(self._rewards),
        })
        self.training_episodes += 1
        if len(self._pending) < self.config.batch_episodes:
            return None
        return self._update_pending()

    def _update_pending(self) -> dict[str, float] | None:
        if not self._pending:
            return None
        returns_by_episode = []
        for trajectory in self._pending:
            returns = []
            value = 0.0
            for reward in reversed(trajectory["rewards"]):
                value = reward + self.config.discount * value
                returns.append(value)
            returns_by_episode.append(torch.tensor(
                list(reversed(returns)), dtype=torch.float32, device=self.device
            ))
        returns_tensor = torch.cat(returns_by_episode)
        values = torch.cat([item["values"] for item in self._pending])
        log_probs = torch.cat([item["log_probs"] for item in self._pending])
        entropies = torch.cat([item["entropies"] for item in self._pending])
        advantages = returns_tensor - values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-6
            )
        actor_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns_tensor)
        entropy = entropies.mean()
        loss = (
            actor_loss
            + self.config.value_coefficient * value_loss
            - self.config.entropy_coefficient * entropy
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        batch_return = sum(sum(item["rewards"]) for item in self._pending)
        batch_episodes = len(self._pending)
        self._pending = []
        return {
            "loss": float(loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "mean_episode_return": float(batch_return / batch_episodes),
        }

    def flush_training(self) -> dict[str, float] | None:
        """Apply a final partial batch after a bounded training run."""
        return self._update_pending()

    def fit_self_imitation(
        self,
        trajectories: list[dict[str, Any]],
        *,
        epochs: int = 100,
        batch_episodes: int = 16,
    ) -> dict[str, float]:
        """Rehearse the learner's own successful reward-selected trajectories.

        Trajectories are selected only by terminal scalar reward.  The method
        receives no event labels, hidden state, correct-action annotations, or
        handcrafted counterfactual targets.
        """
        if epochs < 1 or batch_episodes < 1:
            raise ValueError("epochs and batch_episodes must be positive")
        if not trajectories:
            raise ValueError("self-imitation requires at least one trajectory")
        generator = np.random.default_rng(self.seed + 71)
        updates = 0
        final_loss = 0.0
        for _ in range(epochs):
            order = generator.permutation(len(trajectories))
            pending_losses = []
            for position, trajectory_index in enumerate(order):
                trajectory = trajectories[int(trajectory_index)]
                observations = trajectory["observations"]
                actions = trajectory["actions"]
                rewards = trajectory["rewards"]
                action_count = int(
                    trajectory.get("action_count", self.config.action_count)
                )
                if not 2 <= action_count <= self.config.action_count:
                    raise ValueError("self-imitation action count is invalid")
                if not actions or not (
                    len(observations) == len(actions) == len(rewards)
                ) or any(not 0 <= int(action) < action_count for action in actions):
                    raise ValueError("invalid self-imitation trajectory")
                hidden = torch.zeros(
                    1, self.config.hidden_size, device=self.device
                )
                action_memory = torch.zeros(
                    1,
                    self.config.action_count,
                    self.config.action_memory_size,
                    device=self.device,
                )
                previous_action = self.config.action_count
                previous_reward = 0.0
                logits_by_step = []
                for step, observation in enumerate(observations):
                    pixels = torch.from_numpy(np.asarray(observation).copy()).to(self.device)
                    logits, _, hidden, action_memory = self.policy(
                        pixels,
                        torch.tensor([previous_action], device=self.device),
                        torch.tensor(
                            [previous_reward], dtype=torch.float32, device=self.device
                        ),
                        torch.tensor(
                            [float(step == 0)], dtype=torch.float32, device=self.device
                        ),
                        hidden,
                        action_memory,
                    )
                    logits_by_step.append(logits.squeeze(0)[:action_count])
                    previous_action = int(actions[step])
                    previous_reward = float(rewards[step])
                logits = torch.stack(logits_by_step)
                targets = torch.tensor(actions, dtype=torch.int64, device=self.device)
                returns = []
                value = 0.0
                for reward in reversed(rewards):
                    value = float(reward) + self.config.discount * value
                    returns.append(value)
                weights = torch.tensor(
                    list(reversed(returns)), dtype=torch.float32, device=self.device
                ).clamp_min(0.0)
                weights = weights / weights.mean().clamp_min(1e-6)
                token_losses = F.cross_entropy(logits, targets, reduction="none")
                pending_losses.append((token_losses * weights).mean())
                at_boundary = len(pending_losses) == batch_episodes
                at_end = position == len(order) - 1
                if at_boundary or at_end:
                    loss = torch.stack(pending_losses).mean()
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.config.gradient_clip
                    )
                    self.optimizer.step()
                    updates += 1
                    final_loss = float(loss.detach())
                    pending_losses = []
        return {
            "selected_trajectories": len(trajectories),
            "epochs": epochs,
            "updates": updates,
            "final_cross_entropy": final_loss,
        }

    def fit_reward_event_imitation(
        self,
        trajectories: list[dict[str, Any]],
        *,
        epochs: int = 100,
        batch_episodes: int = 16,
    ) -> dict[str, float]:
        """Imitate only actions with an immediate positive scalar consequence.

        Full trajectories still construct recurrent state. No event label,
        hidden action identity, or authored target is consulted.
        """
        if epochs < 1 or batch_episodes < 1:
            raise ValueError("epochs and batch_episodes must be positive")
        selected = [
            trajectory for trajectory in trajectories
            if any(float(reward) > 0.0 for reward in trajectory.get("rewards", []))
        ]
        if not selected:
            raise ValueError("reward-event imitation requires positive reward events")
        generator = np.random.default_rng(self.seed + 173)
        updates = 0
        final_loss = 0.0
        positive_tokens = sum(
            sum(float(reward) > 0.0 for reward in trajectory["rewards"])
            for trajectory in selected
        )
        for _ in range(epochs):
            order = generator.permutation(len(selected))
            pending_losses = []
            for position, trajectory_index in enumerate(order):
                trajectory = selected[int(trajectory_index)]
                observations = trajectory["observations"]
                actions = trajectory["actions"]
                rewards = trajectory["rewards"]
                if not actions or not (
                    len(observations) == len(actions) == len(rewards)
                ):
                    raise ValueError("invalid reward-event imitation trajectory")
                hidden = torch.zeros(1, self.config.hidden_size, device=self.device)
                action_memory = torch.zeros(
                    1,
                    self.config.action_count,
                    self.config.action_memory_size,
                    device=self.device,
                )
                previous_action = self.config.action_count
                previous_reward = 0.0
                logits_by_step = []
                for step, observation in enumerate(observations):
                    pixels = torch.from_numpy(np.asarray(observation).copy()).to(self.device)
                    logits, _, hidden, action_memory = self.policy(
                        pixels,
                        torch.tensor([previous_action], device=self.device),
                        torch.tensor(
                            [previous_reward], dtype=torch.float32, device=self.device
                        ),
                        torch.tensor(
                            [float(step == 0)], dtype=torch.float32, device=self.device
                        ),
                        hidden,
                        action_memory,
                    )
                    logits_by_step.append(logits.squeeze(0))
                    previous_action = int(actions[step])
                    previous_reward = float(rewards[step])
                logits = torch.stack(logits_by_step)
                targets = torch.tensor(actions, dtype=torch.int64, device=self.device)
                weights = torch.tensor(
                    [float(float(reward) > 0.0) for reward in rewards],
                    dtype=torch.float32,
                    device=self.device,
                )
                token_losses = F.cross_entropy(logits, targets, reduction="none")
                pending_losses.append((token_losses * weights).sum() / weights.sum())
                at_boundary = len(pending_losses) == batch_episodes
                at_end = position == len(order) - 1
                if at_boundary or at_end:
                    loss = torch.stack(pending_losses).mean()
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.config.gradient_clip
                    )
                    self.optimizer.step()
                    updates += 1
                    final_loss = float(loss.detach())
                    pending_losses = []
        return {
            "selected_trajectories": len(selected),
            "positive_reward_tokens": positive_tokens,
            "epochs": epochs,
            "updates": updates,
            "final_cross_entropy": final_loss,
        }

    def fit_reward_outcome_replay(
        self,
        trajectories: list[dict[str, Any]],
        *,
        epochs: int = 100,
        batch_episodes: int = 16,
    ) -> dict[str, float]:
        """Replay rewarded actions and suppress actions with negative outcomes.

        Positive actions receive ordinary imitation loss. Negative actions
        receive unlikelihood loss in the same recurrent context. Selection and
        supervision use scalar reward only.
        """
        if epochs < 1 or batch_episodes < 1:
            raise ValueError("epochs and batch_episodes must be positive")
        selected = [
            trajectory for trajectory in trajectories
            if any(float(reward) > 0.0 for reward in trajectory.get("rewards", []))
        ]
        if not selected:
            raise ValueError("reward-outcome replay requires positive reward events")
        generator = np.random.default_rng(self.seed + 191)
        updates = 0
        final_loss = 0.0
        positive_tokens = sum(
            sum(float(reward) > 0.0 for reward in trajectory["rewards"])
            for trajectory in selected
        )
        negative_tokens = sum(
            sum(float(reward) < 0.0 for reward in trajectory["rewards"])
            for trajectory in selected
        )
        for _ in range(epochs):
            order = generator.permutation(len(selected))
            pending_losses = []
            for position, trajectory_index in enumerate(order):
                trajectory = selected[int(trajectory_index)]
                observations = trajectory["observations"]
                actions = trajectory["actions"]
                rewards = trajectory["rewards"]
                if not actions or not (
                    len(observations) == len(actions) == len(rewards)
                ):
                    raise ValueError("invalid reward-outcome replay trajectory")
                hidden = torch.zeros(1, self.config.hidden_size, device=self.device)
                action_memory = torch.zeros(
                    1,
                    self.config.action_count,
                    self.config.action_memory_size,
                    device=self.device,
                )
                previous_action = self.config.action_count
                previous_reward = 0.0
                logits_by_step = []
                for step, observation in enumerate(observations):
                    pixels = torch.from_numpy(np.asarray(observation).copy()).to(self.device)
                    logits, _, hidden, action_memory = self.policy(
                        pixels,
                        torch.tensor([previous_action], device=self.device),
                        torch.tensor(
                            [previous_reward], dtype=torch.float32, device=self.device
                        ),
                        torch.tensor(
                            [float(step == 0)], dtype=torch.float32, device=self.device
                        ),
                        hidden,
                        action_memory,
                    )
                    logits_by_step.append(logits.squeeze(0))
                    previous_action = int(actions[step])
                    previous_reward = float(rewards[step])
                logits = torch.stack(logits_by_step)
                targets = torch.tensor(actions, dtype=torch.int64, device=self.device)
                reward_tensor = torch.tensor(
                    rewards, dtype=torch.float32, device=self.device
                )
                positive = reward_tensor > 0.0
                negative = reward_tensor < 0.0
                token_losses = []
                if positive.any():
                    token_losses.append(F.cross_entropy(logits[positive], targets[positive]))
                if negative.any():
                    probabilities = torch.softmax(logits[negative], dim=-1)
                    chosen = probabilities.gather(1, targets[negative, None]).squeeze(1)
                    token_losses.append(-torch.log1p(-chosen.clamp_max(1.0 - 1e-6)).mean())
                pending_losses.append(torch.stack(token_losses).mean())
                at_boundary = len(pending_losses) == batch_episodes
                at_end = position == len(order) - 1
                if at_boundary or at_end:
                    loss = torch.stack(pending_losses).mean()
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.config.gradient_clip
                    )
                    self.optimizer.step()
                    updates += 1
                    final_loss = float(loss.detach())
                    pending_losses = []
        return {
            "selected_trajectories": len(selected),
            "positive_reward_tokens": positive_tokens,
            "negative_reward_tokens": negative_tokens,
            "epochs": epochs,
            "updates": updates,
            "final_outcome_loss": final_loss,
        }

    def confidence(self) -> float:
        return self._last_confidence

    def status(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "seed": self.seed,
            "parameter_count": sum(item.numel() for item in self.policy.parameters()),
            "training_episodes": self.training_episodes,
            "training_interactions": self.training_interactions,
            "evaluation_episodes": self.evaluation_episodes,
            "evaluation_interactions": self.evaluation_interactions,
            "intrinsic_reward_total": self.intrinsic_reward_total,
            "information_boundary": [
                "pixels", "previous_action", "scalar_reward", "termination", "memory"
            ],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": self.format,
                "seed": self.seed,
                "config": asdict(self.config),
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "status": self.status(),
            },
            path,
        )

    @classmethod
    def load(
        cls, path: Path, *, device: str | torch.device = "cpu"
    ) -> "RecurrentCausalLearner":
        value = torch.load(Path(path), map_location=device, weights_only=True)
        if not isinstance(value, dict) or value.get("format") != cls.format:
            raise ValueError("unsupported recurrent meta-policy state")
        config = RecurrentMetaConfig(**value["config"])
        result = cls(value["seed"], config=config, device=device)
        result.policy.load_state_dict(value["policy"])
        result.optimizer.load_state_dict(value["optimizer"])
        status = value["status"]
        for field in (
            "training_episodes", "training_interactions",
            "evaluation_episodes", "evaluation_interactions",
        ):
            setattr(result, field, int(status[field]))
        result.intrinsic_reward_total = float(status.get("intrinsic_reward_total", 0.0))
        return result
