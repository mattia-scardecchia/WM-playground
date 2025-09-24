from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset


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
    ):
        assert signal_dim == 2

        self.D = signal_dim + noise_dim
        self.signal_dim = signal_dim
        self.noise_dim = noise_dim
        self.num_actions = num_actions
        self.step_size = float(step)
        self.rng = np.random.RandomState(seed)

    def init_state(self, N: int) -> np.ndarray:
        signal = self.rng.randn(N, self.signal_dim)
        noise = self.rng.randn(N, self.noise_dim)
        return np.concatenate([signal, noise], axis=1)

    def step(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        s = s.copy()
        pos = s[:, : self.signal_dim].copy()
        dx = np.zeros_like(pos)
        dx[:, 0] += (a == 0) * self.step_size
        dx[:, 0] += (a == 1) * (-self.step_size)
        dx[:, 1] += (a == 2) * self.step_size
        dx[:, 1] += (a == 3) * (-self.step_size)
        pos_next = pos + dx
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
