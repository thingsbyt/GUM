"""Inspectable recurrent PPO; four members instantiate four independent objects.

L_pi = -mean(min(r*A, clip(r,1-eps,1+eps)*A)); r=exp(logp-logp_old).
delta_t=r_t+gamma*V_old(t+1)-V_old(t), A_t=delta_t+gamma*lambda*A_(t+1).
Targets=A_raw+V_old; normalize advantages over valid actions in the rollout.
Loss=L_pi + c_v*MSE(V,target) - c_e*H. Old quantities are immutable.
Only whole episodes are shuffled. Each optimization pass rebuilds GRU state
from zero using the full public history; right padding never enters any loss.
The declared finite-horizon task ends at either termination or timeout (zero
bootstrap). Individual escape is not a joint boundary. Its discounted future
reward tail is folded into the last executed action, without fictitious actions.
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
from .recurrent_meta import RecurrentCausalLearner


@dataclass(frozen=True)
class PPOConfig:
    action_count: int = 5
    pooled_size: int = 16
    visual_width: int = 64
    hidden_size: int = 64
    learning_rate: float = 3e-4
    discount: float = .99
    gae_lambda: float = .95
    clip_epsilon: float = .2
    value_coefficient: float = .5
    entropy_coefficient: float = .01
    gradient_clip: float = .5
    batch_episodes: int = 8
    minibatch_episodes: int = 4
    epochs: int = 4
    target_kl: float = .03

    def validate(self):
        if min(self.action_count, self.pooled_size, self.visual_width, self.hidden_size,
               self.batch_episodes, self.minibatch_episodes, self.epochs) < 1:
            raise ValueError("dimensions and batch sizes must be positive")
        if not 0 <= self.discount <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid discount or lambda")
        if min(self.learning_rate, self.clip_epsilon, self.gradient_clip, self.target_kl) <= 0:
            raise ValueError("optimization scales must be positive")
        if min(self.value_coefficient, self.entropy_coefficient) < 0:
            raise ValueError("loss coefficients must be nonnegative")


def generalized_advantages(rewards, values, discount, gae_lambda):
    """One complete finite-horizon episode; never recurse across boundaries."""
    rewards, values = np.asarray(rewards, np.float32), np.asarray(values, np.float32)
    if rewards.shape != values.shape or rewards.ndim != 1 or not len(rewards):
        raise ValueError("rewards/values must be nonempty aligned vectors")
    advantages = np.zeros_like(values)
    following_value = accumulator = 0.
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + discount * following_value - values[t]
        accumulator = delta + discount * gae_lambda * accumulator
        advantages[t] = accumulator
        following_value = values[t]
    return advantages, advantages + values


def clipped_actor_loss(logp, old_logp, advantages, epsilon):
    ratio = torch.exp(logp - old_logp.detach())
    return -torch.minimum(ratio * advantages,
                          ratio.clamp(1 - epsilon, 1 + epsilon) * advantages).mean()


class RecurrentPPOPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = nn.Sequential(nn.Linear(3 * config.pooled_size**2, config.visual_width), nn.Tanh())
        self.memory = nn.GRU(config.visual_width + config.action_count + 3,
                             config.hidden_size, batch_first=True)
        self.actor = nn.Linear(config.hidden_size, config.action_count)
        self.critic = nn.Linear(config.hidden_size, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, .01)
        nn.init.orthogonal_(self.critic.weight, 1.)
        for name, parameter in self.memory.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)

    def pool(self, pixels):
        if pixels.ndim == 3:
            pixels = pixels[None]
        return F.adaptive_avg_pool2d(pixels.float().permute(0, 3, 1, 2) / 255.,
                                     self.config.pooled_size).flatten(1)

    def forward(self, features, previous_action, previous_reward, boundary, hidden=None):
        visual = self.visual(features)
        action = F.one_hot(previous_action.long(), self.config.action_count + 1).float()
        context = torch.cat((visual, action, previous_reward[..., None], boundary[..., None]), -1)
        sequence, hidden = self.memory(context, hidden)
        return self.actor(sequence), self.critic(sequence).squeeze(-1), hidden


class IndependentPPOLearner:
    format = "gum-independent-recurrent-ppo-v1"

    def __init__(self, seed, *, config=None, device="cpu"):
        self.seed, self.config, self.device = int(seed), config or PPOConfig(), torch.device(device)
        self.config.validate()
        # Initialization is independent without perturbing another member's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.policy = RecurrentPPOPolicy(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate, eps=1e-5)
        self.training_generator = torch.Generator().manual_seed(self.seed)
        self.update_generator = torch.Generator().manual_seed(self.seed + 92821)
        self.evaluation_generator = torch.Generator()
        self.training_episodes = self.training_interactions = 0
        self.evaluation_episodes = self.evaluation_interactions = 0
        self.optimizer_steps = self.optimization_epochs = self.delayed_reward_events = 0
        self.pending = []
        self.last_rollout = None
        self.last_update = None

    def begin(self, spec: PublicWorldSpec, observation, *, training):
        if spec.action_count != self.config.action_count:
            raise ValueError("PPO requires the declared action count")
        self.spec, self.training = spec, bool(training)
        self.policy.train(bool(training))
        self.hidden = None
        self.previous_action, self.previous_reward = spec.action_count, 0.
        self.records = []
        self.inactive_ticks = 0
        self.waiting_for_observe = False
        if not training:
            digest = hashlib.sha256(spec.world_id.encode()).digest()
            stream = int.from_bytes(digest[:8], "big") % (2**63 - 1)
            self.evaluation_generator.manual_seed((self.seed + stream) % (2**63 - 1))
        self._validate_observation(observation)

    def _mode(self, training):
        if bool(training) != self.training:
            raise ValueError("mode differs from begin")

    def _validate_observation(self, observation):
        pixels = np.asarray(observation)
        if pixels.shape != tuple(self.spec.observation_shape) or pixels.dtype != np.uint8:
            raise ValueError("observation differs from public pixel contract")
        return pixels

    def act(self, observation, *, training):
        self._mode(training)
        if self.waiting_for_observe:
            raise RuntimeError("each sampled action must be observed")
        with torch.no_grad():
            features = self.policy.pool(torch.from_numpy(self._validate_observation(observation).copy()).to(self.device))
            pa = torch.tensor([[self.previous_action]], device=self.device)
            pr = torch.tensor([[self.previous_reward]], device=self.device)
            boundary = torch.tensor([[float(len(self.records) == 0)]], device=self.device)
            logits, values, self.hidden = self.policy(features[:, None], pa, pr, boundary, self.hidden)
            probabilities = logits[0, 0].softmax(-1)
            action = int(torch.multinomial(probabilities.cpu(), 1, generator=(
                self.training_generator if training else self.evaluation_generator)).item())
            logp = float(probabilities[action].log())
        self.records.append({"features": features[0].cpu(), "previous_action": self.previous_action,
                             "previous_reward": self.previous_reward, "boundary": float(len(self.records) == 0),
                             "action": action, "old_logp": logp, "old_value": float(values[0, 0]), "reward": 0.})
        self.waiting_for_observe = True
        return action

    def observe(self, action, transition: Transition, *, training):
        self._mode(training)
        if not self.waiting_for_observe or action != self.records[-1]["action"]:
            raise ValueError("executed action differs from sampled action")
        self._validate_observation(transition.observation)
        if not np.isfinite(transition.reward):
            raise ValueError("nonfinite reward")
        self.records[-1]["reward"] = float(transition.reward)
        self.previous_action, self.previous_reward = int(action), float(transition.reward)
        self.waiting_for_observe = False
        if training:
            self.training_interactions += 1
        else:
            self.evaluation_interactions += 1

    def credit_delayed_reward(self, reward, *, training):
        self._mode(training)
        self.inactive_ticks += 1
        if not np.isfinite(reward) or not self.records:
            raise ValueError("invalid inactive reward")
        if training:
            self.records[-1]["reward"] += self.config.discount**self.inactive_ticks * float(reward)
            self.delayed_reward_events += int(reward != 0.)

    def finish_episode(self, *, training):
        self._mode(training)
        if self.waiting_for_observe or not self.records:
            raise RuntimeError("incomplete trajectory")
        if not training:
            self.evaluation_episodes += 1
            return None
        rollout = {key: torch.stack([row[key] for row in self.records]) if key == "features"
                   else torch.tensor([row[key] for row in self.records]) for key in self.records[0]}
        advantage, target = generalized_advantages(rollout["reward"].numpy(), rollout["old_value"].numpy(),
                                                   self.config.discount, self.config.gae_lambda)
        rollout["advantages"] = torch.from_numpy(advantage)
        rollout["targets"] = torch.from_numpy(target)
        rollout["inactive_ticks"] = self.inactive_ticks
        self.last_rollout = rollout
        self.pending.append(rollout)
        self.training_episodes += 1
        return self.update() if len(self.pending) >= self.config.batch_episodes else None

    def flush_training(self):
        # The team calls this every episode. Partial batches are preserved,
        # not silently flushed. Protocol checkpoints are batch-aligned.
        return None

    def batch(self, trajectories):
        lengths = [len(row["action"]) for row in trajectories]
        length = max(lengths)
        batch = {}
        for key in ("features", "previous_action", "previous_reward", "boundary", "action",
                    "old_logp", "old_value", "advantages", "targets"):
            values = []
            for row, count in zip(trajectories, lengths):
                padding = (0, 0, 0, length - count) if key == "features" else (0, length - count)
                values.append(F.pad(row[key], padding))
            batch[key] = torch.stack(values).to(self.device)
        batch["mask"] = torch.arange(length, device=self.device)[None] < torch.tensor(lengths, device=self.device)[:, None]
        return batch

    def update(self):
        old = [row["old_logp"].clone() for row in self.pending]
        all_advantages = torch.cat([row["advantages"] for row in self.pending])
        mean, std = all_advantages.mean(), all_advantages.std(unbiased=False).clamp_min(1e-8)
        initial = self.batch(self.pending)
        with torch.no_grad():
            logits, _, _ = self.policy(initial["features"], initial["previous_action"], initial["previous_reward"], initial["boundary"])
            reconstructed = logits.log_softmax(-1).gather(-1, initial["action"].long()[..., None]).squeeze(-1)
            discrepancy = float((reconstructed - initial["old_logp"])[initial["mask"]].abs().max())
        if discrepancy > 2e-5:
            raise RuntimeError(f"old log probabilities fail recurrent reconstruction: {discrepancy}")
        kl = loss_value = 0.
        completed_epochs = steps = 0
        for _ in range(self.config.epochs):
            order = torch.randperm(len(self.pending), generator=self.update_generator).tolist()
            stop = False
            for start in range(0, len(order), self.config.minibatch_episodes):
                batch = self.batch([self.pending[i] for i in order[start:start + self.config.minibatch_episodes]])
                mask = batch["mask"]
                logits, values, _ = self.policy(batch["features"], batch["previous_action"], batch["previous_reward"], batch["boundary"])
                distribution = torch.distributions.Categorical(logits=logits)
                logp = distribution.log_prob(batch["action"].long())[mask]
                old_logp = batch["old_logp"][mask]
                advantage = ((batch["advantages"] - mean.to(self.device)) / std.to(self.device))[mask]
                actor = clipped_actor_loss(logp, old_logp, advantage, self.config.clip_epsilon)
                value = F.mse_loss(values[mask], batch["targets"][mask])
                loss = actor + self.config.value_coefficient * value - self.config.entropy_coefficient * distribution.entropy()[mask].mean()
                with torch.no_grad():
                    difference = logp - old_logp
                    kl = float((difference.exp() - 1 - difference).mean())
                if kl > 1.5 * self.config.target_kl:
                    stop = True
                    break
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                steps += 1
                loss_value = float(loss.detach())
            completed_epochs += 1
            if stop:
                break
        if any(not torch.equal(saved, row["old_logp"]) for saved, row in zip(old, self.pending)):
            raise RuntimeError("old log probabilities mutated")
        self.optimizer_steps += steps
        self.optimization_epochs += completed_epochs
        self.last_update = {"optimizer_steps": steps, "epochs": completed_epochs, "approx_kl": kl,
                            "old_logp_reconstruction_max_error": discrepancy, "old_logp_fixed": True, "loss": loss_value}
        self.pending = []
        return self.last_update

    def status(self):
        return {"format": self.format, "seed": self.seed, "parameter_count": sum(p.numel() for p in self.policy.parameters()),
                "training_episodes": self.training_episodes, "training_interactions": self.training_interactions,
                "evaluation_episodes": self.evaluation_episodes, "evaluation_interactions": self.evaluation_interactions,
                "optimizer_steps": self.optimizer_steps, "optimization_epochs": self.optimization_epochs,
                "delayed_reward_events": self.delayed_reward_events, "pending_episodes": len(self.pending)}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": self.format, "seed": self.seed, "config": asdict(self.config),
                    "policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "training_rng": self.training_generator.get_state(), "update_rng": self.update_generator.get_state(),
                    "evaluation_rng": self.evaluation_generator.get_state(), "pending": self.pending, "status": self.status()}, path)

    @classmethod
    def load(cls, path, *, device="cpu"):
        state = torch.load(path, weights_only=True, map_location="cpu")
        if state["format"] != cls.format:
            raise ValueError("unsupported PPO checkpoint")
        result = cls(state["seed"], config=PPOConfig(**state["config"]), device=device)
        result.policy.load_state_dict(state["policy"])
        result.optimizer.load_state_dict(state["optimizer"])
        result.training_generator.set_state(state["training_rng"])
        result.update_generator.set_state(state["update_rng"])
        result.evaluation_generator.set_state(state["evaluation_rng"])
        result.pending = state["pending"]
        for name in ("training_episodes", "training_interactions", "evaluation_episodes", "evaluation_interactions",
                     "optimizer_steps", "optimization_epochs", "delayed_reward_events"):
            setattr(result, name, state["status"][name])
        return result


class ElapsedRewardGUMLearner(RecurrentCausalLearner):
    """Original loss and representation; correct elapsed-time tail accounting."""
    def begin(self, spec, observation, *, training):
        super().begin(spec, observation, training=training)
        self.inactive_ticks = 0

    def credit_delayed_reward(self, reward, *, training):
        self.inactive_ticks += 1
        super().credit_delayed_reward(self.config.discount**self.inactive_ticks * float(reward), training=training)
