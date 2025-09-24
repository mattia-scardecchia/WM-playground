"""
Hydra configuration schemas for the representation learning project.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    """Data generation and loading configuration"""

    signal_dim: int = 2
    noise_dim: int = 100
    num_actions: int = 4
    traj_len: int = 200
    step_size: float = 0.1
    n_train: int = 300
    n_val: int = 30
    n_test: int = 30
    seed: int = 0
    out_dir: str = "data/"

    def __post_init__(self):
        """Validate data configuration"""
        if self.signal_dim != 2:
            raise ValueError(f"signal_dim must be {2} for this toy environment")
        if self.num_actions != 2 * self.signal_dim:
            raise ValueError(
                f"num_actions must be {2 * self.signal_dim} for this toy environment"
            )


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    z_dim_vae: int = 2
    z_dim_contrastive: int = 2
    enc_widths: List[int] = field(default_factory=lambda: [256, 256])
    dec_widths: List[int] = field(default_factory=lambda: [256, 256])
    proj_widths: List[int] = field(default_factory=lambda: [256, 256])
    dyn_widths: List[int] = field(default_factory=lambda: [256, 256])
    probe_widths: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"


@dataclass
class VAETrainingConfig:
    """VAE-specific training configuration"""

    lr: float = 0.002
    weight_decay: float = 0.0
    beta: float = 0.001


@dataclass
class ContrastiveTrainingConfig:
    """Contrastive learning training configuration"""

    lr: float = 0.001
    weight_decay: float = 0.0
    temperature: float = 0.1


@dataclass
class DynamicsTrainingConfig:
    """Dynamics model training configuration"""

    lr: float = 0.001
    weight_decay: float = 0.0


@dataclass
class ProbeTrainingConfig:
    """Probe training configuration"""

    lr: float = 0.001
    weight_decay: float = 0.0


@dataclass
class TrainingConfig:
    """General training configuration"""

    batch_size: int = 256
    epochs_phase1: int = 10
    epochs_phase2: int = 10
    epochs_probe: int = 10
    num_workers: int = 0
    ckpt_dir: str = "ckpts/"
    eval_batch_size: int = 512
    wandb_log_freq: int = 200
    device: str = "mps"

    # Sub-configurations
    vae: VAETrainingConfig = field(default_factory=VAETrainingConfig)
    contrastive: ContrastiveTrainingConfig = field(
        default_factory=ContrastiveTrainingConfig
    )
    dynamics: DynamicsTrainingConfig = field(default_factory=DynamicsTrainingConfig)
    probe: ProbeTrainingConfig = field(default_factory=ProbeTrainingConfig)


@dataclass
class WandbConfig:
    """Weights & Biases configuration"""

    enabled: bool = True
    project: str = "repr-world"
    entity: Optional[str] = None
    group: Optional[str] = None
    mode: str = "online"
    dir: str = "logs"
    tags: List[str] = field(default_factory=lambda: ["toy", "repr"])


@dataclass
class Config:
    """Main configuration class"""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def register_configs():
    """Register configurations with Hydra ConfigStore"""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="data", name="base", node=DataConfig)
    cs.store(group="model", name="base", node=ModelConfig)
    cs.store(group="train", name="base", node=TrainingConfig)
    cs.store(group="wandb", name="base", node=WandbConfig)
