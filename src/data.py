from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Tuple, Any
from datetime import datetime
import hashlib
from omegaconf import DictConfig, OmegaConf
import yaml


class NoisyVectorWorld:
    """State s = [signal(2), noise(>=1)]. Actions move only the 2-D signal; noise is resampled i.i.d.
    Actions: 0:+x, 1:-x, 2:+y, 3:-y (scaled by step).
    """

    def __init__(
        self,
        signal_dim: int = 2,
        noise_dim: int = 100,
        num_actions: int = 4,
        step: float = 1.0,
        seed: int = 0,
        static_noise: bool = False,
    ):
        assert signal_dim == 2

        self.D = signal_dim + noise_dim
        self.signal_dim = signal_dim
        self.noise_dim = noise_dim
        self.num_actions = num_actions
        self.step_size = float(step)
        self.static_noise = static_noise
        self.rng = np.random.RandomState(seed)

    def init_state(self, N: int) -> np.ndarray:
        signal = self.rng.randn(N, self.signal_dim) * self.step_size
        noise = self.rng.randn(N, self.noise_dim)
        return np.concatenate([signal, noise], axis=1)

    def step(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        pos = s[:, : self.signal_dim].copy()
        dx = np.zeros_like(pos)
        dx[:, 0] += (a == 0) * self.step_size
        dx[:, 0] += (a == 1) * (-self.step_size)
        dx[:, 1] += (a == 2) * self.step_size
        dx[:, 1] += (a == 3) * (-self.step_size)
        pos_next = pos + dx
        if self.static_noise:
            noise_next = s[:, self.signal_dim :].copy()
        else:
            noise_next = self.rng.randn(s.shape[0], self.noise_dim)
        sp = np.concatenate([pos_next, noise_next], axis=1)
        return sp


def sample_trajectories(
    env: NoisyVectorWorld,
    n_traj: int,
    traj_len: int,
    policy: str = "random",
) -> Dict[str, np.ndarray]:
    N = n_traj
    T = traj_len
    s_list, a_list, sp_list = [], [], []
    s = env.init_state(N)
    for _ in range(T):
        if policy == "random":
            a = env.rng.randint(0, env.num_actions, size=(N,))
        else:
            raise ValueError(f"Unknown policy: {policy}")
        sp = env.step(s, a)
        s_list.append(s.copy())
        a_list.append(a.copy())
        sp_list.append(sp.copy())
        s = sp
    s_arr = np.concatenate(s_list, axis=0)
    a_arr = np.concatenate(a_list, axis=0)
    sp_arr = np.concatenate(sp_list, axis=0)
    return {
        "s": s_arr.astype(np.float32),
        "a": a_arr.astype(np.int64),
        "sp": sp_arr.astype(np.float32),
    }


class TriplesNPZ(Dataset):
    def __init__(self, path: str):
        data = np.load(path)
        self.s = torch.from_numpy(data["s"])
        self.a = torch.from_numpy(data["a"])
        self.sp = torch.from_numpy(data["sp"])

    def __len__(self) -> int:
        return self.s.shape[0]

    def __getitem__(self, i: int):
        return {"s": self.s[i], "a": self.a[i], "sp": self.sp[i]}


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

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Main entry point - returns train/val loaders, generating data if needed.

        Returns:
            Tuple of (train_loader, val_loader)
        """
        if self._should_regenerate_data():
            print("🔄 Data config mismatch or missing - regenerating data...")
            self._generate_data_with_config()
        else:
            print("✅ Data config matches - using existing data")

        return self._create_loaders()

    def _should_regenerate_data(self) -> bool:
        """
        Check if data should be regenerated based on config comparison.

        Returns:
            True if data should be regenerated, False otherwise
        """
        # Check if data files exist
        required_files = ["train.npz", "val.npz", "test.npz"]
        if not all((self.data_dir / f).exists() for f in required_files):
            print("📁 Missing data files")
            return True

        # Check if config file exists
        if not self.config_path.exists():
            print("📁 Missing data config file")
            return True

        # Load and compare configs
        try:
            with open(self.config_path, "r") as f:
                stored_config = yaml.safe_load(f)
            current_config = OmegaConf.to_container(self.cfg.data, resolve=True)

            if stored_config != current_config:
                print("⚠️  Data config differs from current config")
                self._print_config_diff(stored_config, current_config)
                return True

        except Exception as e:
            print(f"❌ Error reading stored config: {e}")
            return True

        return False

    def _print_config_diff(self, stored_config: Dict[str, Any], current_config: Any):
        """Print differences between stored and current config for debugging."""
        print("Config differences:")
        if not isinstance(current_config, dict):
            print(f"  Current config type mismatch: {type(current_config)}")
            return

        all_keys = set(stored_config.keys()) | set(current_config.keys())

        for key in sorted(all_keys):
            stored_val = stored_config.get(key, "<MISSING>")
            current_val = current_config.get(key, "<MISSING>")

            if stored_val != current_val:
                print(f"  {key}: {stored_val} → {current_val}")

    def _generate_data_with_config(self):
        """Generate data and save both data and config."""
        # Create output directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create environment
        env = NoisyVectorWorld(
            signal_dim=self.cfg.data.signal_dim,
            noise_dim=self.cfg.data.noise_dim,
            num_actions=self.cfg.data.num_actions,
            step=self.cfg.data.step_size,
            seed=self.cfg.data.seed,
            static_noise=self.cfg.data.static_noise,
        )

        # Generate data splits
        splits = [
            ("train", self.cfg.data.n_train),
            ("val", self.cfg.data.n_val),
            ("test", self.cfg.data.n_test),
        ]

        for split_name, n_traj in splits:
            triples = sample_trajectories(
                env, n_traj=n_traj, traj_len=self.cfg.data.traj_len, policy="random"
            )

            # Save data
            path = self.data_dir / f"{split_name}.npz"
            np.savez(str(path), s=triples["s"], a=triples["a"], sp=triples["sp"])

            print(
                f"📝 Wrote {path} with shapes s{triples['s'].shape}, "
                f"a{triples['a'].shape}, sp{triples['sp'].shape}"
            )

        # Save config and metadata
        self._save_config_and_metadata()

    def _save_config_and_metadata(self):
        """Save the data config and generation metadata."""
        # Save data config as YAML
        config_dict = OmegaConf.to_container(self.cfg.data, resolve=True)
        with open(self.config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
        print(f"💾 Saved data config to {self.config_path}")

        # Save metadata as JSON
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "config_hash": self._get_config_hash(config_dict),
        }

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Saved metadata to {self.metadata_path}")

    def _get_config_hash(self, config_dict: Any) -> str:
        """Generate a hash of the config for quick comparison."""
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders from existing data."""
        batch_size = self.cfg.train.batch_size
        num_workers = self.cfg.train.num_workers

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
        print("🔄 Force regenerating data...")
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
