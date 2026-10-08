"""Explicit method settings. Unreported details are documented in FIDELITY.md."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

@dataclass
class Config:
    name: str = 'asteroids-local'
    encoder_id: str = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
    encoder_revision: str = 'main'
    encoder_path: str = ''
    image_height: int = 256
    image_width: int = 512
    encoder_dim: int = 1024
    patch_size: int = 16
    projector_hidden: int = 512
    projected_dim: int = 256
    projector_kernel: int = 2
    predictor_dim: int = 1024
    predictor_depth: int = 12
    predictor_heads: int = 16
    mlp_ratio: int = 4
    adaln_init_scale: float = 10.0
    history: int = 4
    horizon: int = 8
    action_dim: int = 5
    sigreg_directions: int = 1024
    sigreg_knots: int = 17
    sigreg_weight: float = 0.09
    sigreg_site_chunk: int = 32
    sigreg_direction_chunk: int = 128
    rollout_training: bool = True
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 0.000025
    warmup_epochs: int = 1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    activation_checkpointing: bool = True
    projection_frame_chunk: int = 8
    encoder_batch: int = 2
    feature_dtype: str = 'float16'
    candidates: int = 256
    candidate_chunk: int = 16
    train_episodes: int = 24
    validation_episodes: int = 4
    test_episodes: int = 8
    episode_steps: int = 128
    difficulty: int = 1
    frames_per_action: int = 6
    seed: int = 7
    cpu_threads: int = 2
    device: str = 'auto'
    max_process_ram_gib: float = 12.0
    max_system_ram_gib: float = 28.0
    max_gpu_gib: float = 10.0
    max_gpu_temperature_c: float = 78.0
    max_cache_gib: float = 12.0
    step_pause_seconds: float = 0.02
    # Continual-learning extension. These settings do not change the paper method
    # signature; they control how experience is gathered and replayed around it.
    online_updates_per_episode: int = 4
    online_min_windows: int = 32
    online_replay_episodes: int = 48
    online_recent_fraction: float = 0.5
    online_epsilon_start: float = 1.0
    online_epsilon_end: float = 0.1
    online_epsilon_decay_steps: int = 4096
    online_goal_library_size: int = 128
    online_milestone_updates: int = 25
    # Default general self-learning system. It starts from random weights and
    # learns its visual encoder, dynamics and outcome heads from pixels.
    self_observation_height: int = 64
    self_observation_width: int = 128
    self_frame_stack: int = 4
    self_latent_dim: int = 128
    self_ensemble: int = 3
    self_batch_size: int = 32
    self_learning_rate: float = 0.0003
    self_updates_per_episode: int = 16
    self_warmup_transitions: int = 128
    self_replay_episodes: int = 64
    self_recent_fraction: float = 0.5
    self_planner_candidates: int = 128
    self_planner_horizon: int = 8
    self_discount: float = 0.99
    self_curiosity_weight: float = 0.05
    self_epsilon_start: float = 0.4
    self_epsilon_end: float = 0.05
    self_epsilon_decay_steps: int = 5000
    self_milestone_updates: int = 100
    # Stable pixel-control learner. Unlike the experimental world-model planner,
    # this policy is trained only against returns observed in the real simulator.
    stable_latent_dim: int = 256
    stable_batch_size: int = 64
    stable_learning_rate: float = 0.0001
    stable_updates_per_episode: int = 16
    stable_warmup_transitions: int = 256
    stable_replay_episodes: int = 512
    stable_recent_fraction: float = 0.5
    stable_discount: float = 0.99
    stable_n_step: int = 10
    stable_target_interval: int = 500
    stable_epsilon_start: float = 1.0
    stable_epsilon_end: float = 0.05
    stable_epsilon_decay_steps: int = 20000
    stable_random_shift: int = 4
    stable_grad_clip: float = 10.0
    stable_milestone_updates: int = 500
    stable_jepa_weight: float = 0.2
    stable_jepa_tau: float = 0.995
    stable_outcome_fraction: float = 0.25

    def validate(self, *, allow_test: bool = False) -> 'Config':
        if self.history < 1 or self.horizon < 1 or self.batch_size < 2:
            raise ValueError('history/horizon must be positive; SIGReg needs a batch of at least two.')
        if self.image_height < 64 or self.image_width < 64 or self.image_height % 64 or self.image_width % 64:
            raise ValueError('Image dimensions must be multiples of 64 (16-pixel patches, two stride-2 layers).')
        if self.predictor_heads < 1 or self.predictor_depth < 1 or self.predictor_dim % self.predictor_heads or self.predictor_dim // self.predictor_heads < 6:
            raise ValueError('Invalid attention head dimensions for three-axis RoPE.')
        if self.projector_kernel not in (2, 3):
            raise ValueError('Supported explicitly documented projector kernels: 2 or 3.')
        if self.feature_dtype not in ('float16', 'float32'):
            raise ValueError('feature_dtype must be float16 or float32.')
        if self.candidates < 5 or self.candidates > self.action_dim ** self.horizon:
            raise ValueError('Invalid candidate vocabulary size.')
        if self.episode_steps < self.history + self.horizon or self.difficulty not in (1, 2, 3):
            raise ValueError('Episode too short, or invalid difficulty.')
        for field in ('sigreg_site_chunk', 'sigreg_direction_chunk', 'projection_frame_chunk',
                      'encoder_batch', 'candidate_chunk', 'cpu_threads', 'epochs'):
            if getattr(self, field) <= 0:
                raise ValueError(f'{field} must be positive.')
        for field in ('learning_rate','max_process_ram_gib','max_system_ram_gib','max_gpu_gib','max_gpu_temperature_c','max_cache_gib'):
            if getattr(self, field) <= 0:raise ValueError(f'{field} must be positive.')
        if self.step_pause_seconds < 0:raise ValueError('step_pause_seconds cannot be negative.')
        for field in ('online_updates_per_episode','online_min_windows','online_replay_episodes',
                      'online_epsilon_decay_steps','online_goal_library_size','online_milestone_updates'):
            if getattr(self,field)<=0:raise ValueError(f'{field} must be positive.')
        if not 0<=self.online_epsilon_end<=self.online_epsilon_start<=1:
            raise ValueError('Online exploration probabilities must satisfy 0 <= end <= start <= 1.')
        if not 0<=self.online_recent_fraction<=1:
            raise ValueError('online_recent_fraction must be between zero and one.')
        for field in ('self_observation_height','self_observation_width','self_frame_stack',
                      'self_latent_dim','self_ensemble','self_batch_size','self_updates_per_episode',
                      'self_warmup_transitions','self_replay_episodes','self_planner_candidates',
                      'self_planner_horizon','self_epsilon_decay_steps','self_milestone_updates'):
            if getattr(self,field)<=0:raise ValueError(f'{field} must be positive.')
        if self.self_observation_height%16 or self.self_observation_width%16:
            raise ValueError('Self-learning observation dimensions must be multiples of 16.')
        if not 0<=self.self_recent_fraction<=1 or not 0<self.self_discount<=1:
            raise ValueError('Invalid self-learning replay fraction or discount.')
        if not 0<=self.self_epsilon_end<=self.self_epsilon_start<=1:
            raise ValueError('Invalid self-learning exploration probabilities.')
        if self.self_learning_rate<=0 or self.self_curiosity_weight<0:
            raise ValueError('Invalid self-learning rate or curiosity weight.')
        for field in ('stable_latent_dim','stable_batch_size','stable_updates_per_episode',
                      'stable_warmup_transitions','stable_replay_episodes','stable_n_step',
                      'stable_target_interval','stable_epsilon_decay_steps','stable_milestone_updates'):
            if getattr(self,field)<=0:raise ValueError(f'{field} must be positive.')
        if self.stable_learning_rate<=0 or self.stable_grad_clip<=0 or self.stable_random_shift<0 or self.stable_jepa_weight<0:
            raise ValueError('Invalid stable learner optimization or augmentation setting.')
        if not 0<=self.stable_recent_fraction<=1 or not 0<self.stable_discount<=1 or not 0<=self.stable_outcome_fraction<=1:
            raise ValueError('Invalid stable learner replay fraction or discount.')
        if not 0<self.stable_jepa_tau<1:
            raise ValueError('stable_jepa_tau must be between zero and one.')
        if not 0<=self.stable_epsilon_end<=self.stable_epsilon_start<=1:
            raise ValueError('Invalid stable learner exploration probabilities.')
        if not allow_test:
            exact = {'encoder_dim':1024, 'projected_dim':256, 'patch_size':16,
                     'history':4, 'horizon':8, 'sigreg_directions':1024, 'sigreg_knots':17,
                     'action_dim':5}
            for key, expected in exact.items():
                if getattr(self, key) != expected:
                    raise ValueError(f'This build preserves {key}={expected}; no silent reduced-model fallback.')
            if self.encoder_id != 'facebook/dinov3-vitl16-pretrain-lvd1689m':
                raise ValueError('This runtime requires the specified DINOv3 ViT-L encoder.')
        return self

    @property
    def grid(self) -> tuple[int,int]:
        return self.image_height // 16, self.image_width // 16

    @property
    def projected_grid(self) -> tuple[int,int]:
        return self.image_height // 64, self.image_width // 64

    @property
    def sequence_frames(self) -> int:
        return self.history + (self.horizon if self.rollout_training else 1)

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> 'Config':
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(**{k:v for k,v in data.items() if not k.startswith('_')}).validate()
