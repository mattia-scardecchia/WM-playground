from typing import Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import logging
from pathlib import Path
from typing import Tuple, Any
from datetime import datetime
import hashlib
from omegaconf import DictConfig, OmegaConf
import yaml
from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """Abstract environment interface for trajectory generation and config serialization.

    Concrete envs must implement state initialization, stepping, sampling trajectories,
    and returning a serialisable config dict describing how data was generated.
    """

    signal_dim: int
    step_size: float
    rng: Any
    seed: int
    static_noise: bool

    @abstractmethod
    def init_state(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Advance state given actions.

        For compatibility the method may return either:
          - next_observed_state (np.ndarray)
        or
          - (next_latent, next_observed_state) as a tuple of np.ndarrays
        Concrete envs should document what they return; DataManager and the
        trajectory generation helper will handle both cases.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Any) -> "BaseEnv":
        """Create an env instance from a config object (dict/OmegaConf).

        This allows DataManager to instantiate envs without knowing their
        constructor signature.
        """
        pass


class LegacyVectorWorld(BaseEnv):
    """State s = [signal(2), noise(>=1)]. Actions move only the 2-D signal; noise is resampled i.i.d.
    Actions: 0:+x, 1:-x, 2:+y, 3:-y (scaled by step).
    """

    def __init__(
        self,
        signal_dim: int = 2,
        noise_dim: int = 100,
        step: float = 1.0,
        seed: int = 0,
        static_noise: bool = False,
    ):
        assert signal_dim == 2
        self.seed = seed
        self.D = signal_dim + noise_dim
        self.signal_dim = signal_dim
        self.noise_dim = noise_dim
        self.step_size = step
        self.static_noise = static_noise
        self.rng = np.random.RandomState(seed)

    @classmethod
    def from_config(cls, cfg: Any) -> "LegacyVectorWorld":
        return cls(
            signal_dim=cfg.signal_dim,
            noise_dim=cfg.noise_dim,
            step=cfg.step_size,
            seed=cfg.seed,
            static_noise=cfg.static_noise,
        )

    def init_state(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        signal = self.rng.randn(N, self.signal_dim)
        noise = self.rng.randn(N, self.noise_dim)
        return np.concatenate([signal, noise], axis=1), signal

    def step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sig = s[:, : self.signal_dim].copy()

        sig_next = sig + a
        if self.static_noise:
            noise_next = s[:, self.signal_dim :].copy()
        else:
            noise_next = self.rng.randn(s.shape[0], self.noise_dim)
        sp = np.concatenate([sig_next, noise_next], axis=1)
        return sp, sig_next


class LatentVectorWorld(BaseEnv):
    """New environmental variant with a fixed random MLP projection.

    The env maintains a low-dimensional latent composed of three parts:
      - signal (learned / acted upon)
      - static (fixed per-episode)
      - memoryless (resampled each step)

    The observed state is the normalized projection of the latent via a fixed 1-hidden-layer
    MLP (tanh hidden).
    """

    def __init__(
        self,
        signal_dim: int,
        static_noise_dim: int,
        memoryless_noise_dim: int,
        proj_dim: int,
        mlp_hidden_dim: Optional[int],
        step: float,
        seed: int,
    ):
        self.signal_dim = signal_dim
        self.static_noise_dim = static_noise_dim
        self.memoryless_noise_dim = memoryless_noise_dim
        self.step_size = step
        self.proj_dim = proj_dim
        self.mlp_hidden_dim = (
            mlp_hidden_dim if mlp_hidden_dim is not None else self.proj_dim
        )
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        input_dim = self.signal_dim + self.static_noise_dim + self.memoryless_noise_dim
        h = self.mlp_hidden_dim
        o = self.proj_dim
        self.W1 = self.rng.randn(input_dim, h).astype(np.float32)
        self.b1 = self.rng.randn(h).astype(np.float32)
        self.W2 = self.rng.randn(h, o).astype(np.float32)
        self.b2 = self.rng.randn(o).astype(np.float32)

    @classmethod
    def from_config(cls, cfg: Any) -> "LatentVectorWorld":
        return cls(
            signal_dim=cfg.signal_dim,
            static_noise_dim=cfg.static_noise_dim,
            memoryless_noise_dim=cfg.memoryless_noise_dim,
            proj_dim=cfg.state_dim,
            mlp_hidden_dim=cfg.mlp_hidden_dim,
            step=cfg.step_size,
            seed=cfg.seed,
        )

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project low-dim concatenated state into higher-dim latent using fixed random MLP.

        Args:
            x: shape (N, input_dim)
        Returns:
            z: shape (N, proj_dim)
        """
        h = np.tanh(x.dot(self.W1) + self.b1)
        z = h.dot(self.W2) + self.b2
        z = z.astype(np.float32)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        z = z / norms
        return z

    def init_state(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        signal = self.rng.randn(N, self.signal_dim)
        static = self.rng.randn(N, self.static_noise_dim)
        memoryless = self.rng.randn(N, self.memoryless_noise_dim)
        latent_state = np.concatenate([signal, static, memoryless], axis=1)
        observed_state = self.project(latent_state)
        return observed_state, signal

    def step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = s.shape[0]
        sig = s[:, : self.signal_dim].copy()
        static = s[:, self.signal_dim : self.signal_dim + self.static_noise_dim].copy()

        sig_next = sig + a
        static_next = static
        mem_next = self.rng.randn(N, self.memoryless_noise_dim)
        sp_latent = np.concatenate([sig_next, static_next, mem_next], axis=1)
        sp_observed = self.project(sp_latent)

        return sp_observed, sig_next


def sample_trajectories(
    env: BaseEnv,
    n_traj: int,
    traj_len: int,
    policy: str = "random-discrete",
) -> Dict[str, np.ndarray]:
    """Generic trajectory generator using env.init_state and env.step. The env.step must return (next_state, signal_next)."""
    N = n_traj
    T = traj_len
    s_list, a_list, sp_list = [], [], []
    sig_list, sig_next_list = [], []

    s, sig = env.init_state(N)

    for _ in range(T):
        rng = env.rng
        sig_dim = env.signal_dim

        if policy == "random-discrete":
            num_actions = 2 * sig_dim
            a_discrete = rng.randint(0, num_actions, size=(N,))
            a_cont = np.zeros((N, sig_dim), dtype=np.float32)
            coords = a_discrete % sig_dim
            signs = np.where(a_discrete < sig_dim, 1.0, -1.0)
            a_cont[np.arange(N), coords] = signs * env.step_size
            a_to_store = a_discrete
        else:
            raise ValueError(f"Unknown policy: {policy}")

        sp, sig_next = env.step(s, a_cont)

        s_list.append(s.copy())
        a_list.append(a_to_store.copy())
        sp_list.append(sp.copy())
        sig_list.append(sig)
        sig_next_list.append(sig_next)

        s = sp
        sig = sig_next

    s_arr = np.concatenate(s_list, axis=0).astype(np.float32)
    a_arr = np.concatenate(a_list, axis=0).astype(np.float32)
    sp_arr = np.concatenate(sp_list, axis=0).astype(np.float32)
    sig_arr = np.concatenate(sig_list, axis=0).astype(np.float32)
    sig_next_arr = np.concatenate(sig_next_list, axis=0).astype(np.float32)

    out = {
        "s": s_arr,
        "a": a_arr,
        "sp": sp_arr,
        "sig": sig_arr,
        "sig_next": sig_next_arr,
    }
    return out


class TriplesNPZ(Dataset):
    def __init__(self, path: str):
        data = np.load(path)
        self.s = torch.from_numpy(data["s"])
        self.a = torch.from_numpy(data["a"])
        self.sp = torch.from_numpy(data["sp"])
        self.sig = torch.from_numpy(data["sig"])
        self.sig_next = torch.from_numpy(data["sig_next"])

    def __len__(self) -> int:
        return self.s.shape[0]

    def __getitem__(self, i: int):
        return {
            "s": self.s[i],
            "a": self.a[i],
            "sp": self.sp[i],
            "sig": self.sig[i],
            "sig_next": self.sig_next[i],
        }


class DataManager:
    """
    Manages data generation, loading, and config validation.

    Ensures that loaded data always matches the current configuration,
    automatically regenerating when needed.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.data_dir = Path(cfg.data.out_dir)
        self.config_path = self.data_dir / "data_config.yaml"
        self.metadata_path = self.data_dir / "metadata.json"

    def get_data_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """
        Main entry point - returns train/val loaders, generating data if needed.

        Returns:
            Tuple of (train_loader, val_loader)
        """
        if self._should_regenerate_data():
            logging.info("Data config mismatch or missing - regenerating data...")
            self._generate_data_with_config()
        else:
            logging.info("Data config matches - using existing data")

        return self._create_loaders(batch_size)

    def _should_regenerate_data(self) -> bool:
        """
        Check if data should be regenerated based on config comparison.

        Returns:
            True if data should be regenerated, False otherwise
        """
        # Check if data files exist
        required_files = ["train.npz", "val.npz", "test.npz"]
        if not all((self.data_dir / f).exists() for f in required_files):
            logging.debug("Missing data files")
            return True

        # Check if config file exists
        if not self.config_path.exists():
            logging.debug("Missing data config file")
            return True

        # Load and compare configs
        try:
            with open(self.config_path, "r") as f:
                stored_config = yaml.safe_load(f)
            current_config = OmegaConf.to_container(self.cfg.data, resolve=True)

            if stored_config != current_config:
                logging.info("Data config differs from current config")
                self._print_config_diff(stored_config, current_config)
                return True

        except Exception as e:
            logging.warning(f"Error reading stored config: {e}")
            return True

        return False

    def _print_config_diff(self, stored_config: Dict[str, Any], current_config: Any):
        """Print differences between stored and current config for debugging."""
        logging.info("Config differences:")
        if not isinstance(current_config, dict):
            logging.info(f"  Current config type mismatch: {type(current_config)}")
            return

        all_keys = set(stored_config.keys()) | set(current_config.keys())

        for key in sorted(all_keys):
            stored_val = stored_config.get(key, "<MISSING>")
            current_val = current_config.get(key, "<MISSING>")

            if stored_val != current_val:
                logging.info(f"  {key}: {stored_val} → {current_val}")

    def _generate_data_with_config(self):
        """Generate data and save both data and config."""
        # Create output directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        env_type = self.cfg.data.type
        if env_type not in ["legacy", "new"]:
            raise ValueError()
        env_class = LegacyVectorWorld if env_type == "legacy" else LatentVectorWorld
        env = env_class.from_config(self.cfg.data)

        splits = [
            ("train", self.cfg.data.n_train),
            ("val", self.cfg.data.n_val),
            ("test", self.cfg.data.n_test),
        ]
        for split_name, n_traj in splits:
            triples = sample_trajectories(
                env,
                n_traj=n_traj,
                traj_len=self.cfg.data.traj_len,
                policy=self.cfg.data.policy,
            )
            path = self.data_dir / f"{split_name}.npz"
            np.savez_compressed(
                str(path),
                s=triples["s"],
                a=triples["a"],
                sp=triples["sp"],
                sig=triples["sig"],
                sig_next=triples["sig_next"],
            )

            logging.info(
                f"Generated {split_name} data: {path} with shapes s{triples['s'].shape}, "
                f"a{triples['a'].shape}, sp{triples['sp'].shape}, sig{triples['sig'].shape}, "
                f"sig_next{triples['sig_next'].shape}"
            )
        self._save_config_and_metadata()

    def _save_config_and_metadata(self):
        """Save the data config and generation metadata."""
        # Save data config as YAML
        config_dict = OmegaConf.to_container(self.cfg.data, resolve=True)
        with open(self.config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
        logging.info(f"Saved data config to {self.config_path}")

        # Save metadata as JSON
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "config_hash": self._get_config_hash(config_dict),
        }

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved metadata to {self.metadata_path}")

    def _get_config_hash(self, config_dict: Any) -> str:
        """Generate a hash of the config for quick comparison."""
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _create_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders from existing data."""
        num_workers = self.cfg.train.num_workers

        logging.info(
            f"Creating data loaders (batch_size={batch_size}, num_workers={num_workers})"
        )

        train_loader = DataLoader(
            TriplesNPZ(str(self.data_dir / "train.npz")),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            TriplesNPZ(str(self.data_dir / "val.npz")),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def get_test_loader(self) -> DataLoader:
        """Get test data loader (separate method since it's used less frequently)."""
        batch_size = self.cfg.train.batch_size
        num_workers = self.cfg.train.num_workers

        return DataLoader(
            TriplesNPZ(str(self.data_dir / "test.npz")),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def force_regenerate(self):
        """Force regeneration of data regardless of config match."""
        logging.info("Force regenerating data...")
        self._generate_data_with_config()

    def get_data_info(self) -> dict:
        """Get information about the current data."""
        info = {
            "data_dir": str(self.data_dir),
            "config_exists": self.config_path.exists(),
            "metadata_exists": self.metadata_path.exists(),
        }

        # Add file sizes if they exist
        for split in ["train", "val", "test"]:
            path = self.data_dir / f"{split}.npz"
            if path.exists():
                info[f"{split}_size_mb"] = path.stat().st_size / (1024 * 1024)

        # Add metadata if available
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    metadata = json.load(f)
                info["metadata"] = metadata
            except Exception:
                pass

        return info
