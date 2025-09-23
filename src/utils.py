import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Sequence
from collections import defaultdict


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_onehot(a: torch.Tensor, num_actions: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(a.long(), num_classes=num_actions).float()


def make_mlp(
    sizes: Sequence[int], activation: str = "relu", out_act: bool = False
) -> nn.Sequential:
    acts = {"relu": nn.ReLU, "tanh": nn.Tanh}
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers.append(acts[activation]())
    return nn.Sequential(*layers)


class AverageMeter:
    def __init__(self):
        self.reset()

        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def reset(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, vals: Dict[str, float], ks: Dict[str, int] = None):
        if ks is None:
            ks = {k: 1 for k in vals}
        for k in vals:
            self.sums[k] += vals[k] * ks.get(k, 1)
            self.counts[k] += ks.get(k, 1)

    @property
    def avg(self) -> Dict[str, float]:
        return {k: self.sums[k] / max(1, self.counts[k]) for k in self.sums}
