from typing import Callable, List, Sequence, Dict, Any, Tuple, Optional, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from utils import make_mlp


def _unpack_transition(
    batch: Dict[str, torch.Tensor], device: torch.device, num_actions: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper function to prepare batch data. Works with data from TransitionDataset."""
    observation = batch["observation"].to(device)
    B, _ = observation.shape
    action_onehot = (
        torch.nn.functional.one_hot(
            batch["action"].reshape((B,)).long(), num_classes=num_actions
        )
        .float()
        .to(device)
    )
    observation_next = batch["observation_next"].to(device)
    return observation, observation_next, action_onehot


def _unpack_trajectory(
    batch: Dict[str, torch.Tensor], device: torch.device, num_actions: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper function to prepare batch data. Works with data from TrajectoryDataset."""
    observation = batch["observation"].to(device)  # B, T, D
    B, T, _ = observation.shape
    action_onehot = (
        torch.nn.functional.one_hot(
            batch["action"].reshape((B, T)).long(), num_classes=num_actions
        )
        .float()
        .to(device)
    )
    observation_next = batch["observation_next"].to(device)
    return observation, observation_next, action_onehot


class TrainableModel(nn.Module, ABC):
    """Base class for models that can be trained with the generic trainer"""

    @abstractmethod
    def training_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform one training step
        Returns: (loss, metrics_dict)
        """
        pass

    @abstractmethod
    def validation_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform one validation step
        Returns: (loss, metrics_dict)
        """
        pass

    @abstractmethod
    def get_optimizer_config(self, cfg: Any) -> Dict[str, Any]:
        """Return optimizer configuration for this model"""
        pass

    def save_checkpoint(self, ckpt_dir: str, cfg: Any) -> Dict[str, Any]:
        """
        Save model checkpoint(s) and return info for wandb artifacts
        Returns: {"paths": list, "artifact_name": str, "artifact_type": str}
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        filename = f"{self.id}.pt"
        path = os.path.join(ckpt_dir, filename)
        torch.save({"state_dict": self.state_dict(), "cfg": cfg}, path)

        return {
            "paths": [path],
            "artifact_name": f"{self.id}-model",
            "artifact_type": "model",
        }

    @property
    @abstractmethod
    def id(self) -> str:
        """Return the model identifier for logging and file naming"""
        pass


def beta_vae_loss(x_hat, x, mu, logvar, beta: float):
    """Beta-VAE loss function"""
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, {"recon": recon.item(), "kl": kl.item()}


class VAE(TrainableModel):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        enc_widths,
        dec_widths,
        beta: float = 1.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.beta = beta
        self.enc = make_mlp([x_dim] + list(enc_widths), activation)
        hid = enc_widths[-1]
        self.mu = nn.Linear(hid, z_dim)
        self.logvar = nn.Linear(hid, z_dim)
        self.dec = make_mlp([z_dim] + list(dec_widths) + [x_dim], activation)

    def encode(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return mu, logvar, z

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar, z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def training_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = batch["observation"].to(device)
        x_hat, mu, logvar, z = self(x)
        loss, parts = beta_vae_loss(x_hat, x, mu, logvar, self.beta)

        metrics = {
            "train_loss": loss.item(),
            "train_recon": parts["recon"],
            "train_kl": parts["kl"],
        }
        return loss, metrics

    def validation_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = batch["observation"].to(device)
        x_hat, mu, logvar, z = self(x)
        loss, parts = beta_vae_loss(x_hat, x, mu, logvar, self.beta)

        metrics = {
            "val_loss": loss.item(),
            "val_recon": parts["recon"],
            "val_kl": parts["kl"],
        }
        return loss, metrics

    def get_optimizer_config(self, cfg: Any) -> Dict[str, Any]:
        return {
            "type": "AdamW",
            "lr": cfg.train.vae.lr,
            "weight_decay": cfg.train.vae.weight_decay,
            "gradient_clip": cfg.train.vae.gradient_clip,
        }

    @property
    def id(self) -> str:
        return "vae"

    def get_encoder_fn(self):
        """Return a function that encodes input states to latent representations"""

        def encode_fn(x):
            mu, logvar, z = self.encode(x)
            return mu

        return encode_fn


class EncoderMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        embed_dim: int,
        activation: str,
        use_layer_norm: bool,
        eps: float,
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        out_dim = embed_dim - 1 if use_layer_norm else embed_dim
        self.net = make_mlp([input_dim] + hidden_dims + [out_dim], activation)
        self.layer_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=eps)

    def forward(self, x):
        """Following https://github.com/google-research/google-research/blob/master/rl_repr/batch_rl/embed.py#L135"""
        x = self.net(x)
        if self.use_layer_norm:
            x = torch.cat([x, torch.ones_like(x[:, :1])], dim=-1)  # B, D-1 -> B, D
            x = self.layer_norm(x)
        return x


class NachumConstrastive(TrainableModel):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        num_actions: int,
        enc_widths,
        proj_widths,
        temperature: float,
        use_layer_norm: bool,
        eps: float,  # ignored if not use_layer_norm
        activation: str,
        k: int,
        alpha: float,  # ignored if k == 1
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.num_actions = num_actions
        self.temperature = temperature
        self.k = k
        self.alpha = alpha
        self.phi = EncoderMLP(
            input_dim=x_dim,
            hidden_dims=list(enc_widths),
            embed_dim=z_dim,
            activation=activation,
            eps=eps,
            use_layer_norm=use_layer_norm,
        )
        self.g = EncoderMLP(
            input_dim=x_dim + num_actions,
            hidden_dims=list(proj_widths),
            embed_dim=z_dim,
            activation=activation,
            eps=eps,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, s):
        return self.phi(s)

    def contrastive_loss(self, obs, obs_next, action, device):
        # Implementation 1: s uses own (s', a) as positive, other (s', a) in batch as negatives. no clear EBM interpretation.
        if self.k == 1:
            query = self.phi(obs)  # B, D
            keys = self.g(torch.cat([obs_next, action], dim=-1))  # B, D
            diff = query[:, None, :] - keys[None, :, :]  # B, B, D
            sim = -torch.sum(diff**2, dim=-1) / self.temperature  # B, B
            target = torch.arange(query.size(0), device=device)
            loss = F.cross_entropy(sim, target)
        else:
            B, T, _ = obs.shape
            if self.alpha is None:
                # Version 1: treat all negatives equally
                idx = torch.randint(0, T, (B,), device=device)
                query = self.phi(obs[torch.arange(B), idx])  # B, D
                g_input = torch.cat([obs_next, action], dim=-1).reshape(B * T, -1)
                keys = self.g(g_input).reshape(B, T, -1)  # B, T, D
                diff = query[:, None, :] - keys  # B, T, D
                logits = -torch.sum(diff**2, dim=-1) / self.temperature  # B, T
                target = idx
                loss = F.cross_entropy(logits, target)
            else:
                # Version 2: weigh negatives from same trajectory differently than those from other trajectories
                idx = torch.randint(0, T, (B,), device=device)
                query = self.phi(obs[torch.arange(B), idx])  # B, D
                keys = self.g(torch.cat([obs_next, action], dim=-1).reshape(B * T, -1))
                sim = -(torch.cdist(query, keys, 2) ** 2) / self.temperature  # B, B*T
                sim = sim.reshape(B, B, T)

                # traj_eq[b, b', t] is set to (b == b')
                traj_eq = torch.zeros(B, B, dtype=torch.bool, device=device)
                traj_eq[torch.arange(B), torch.arange(B)] = True
                traj_eq = traj_eq.unsqueeze(-1).expand(B, B, T)

                # time_neg[b, b', t] is set to (t != idx[b])
                time_neg = ~F.one_hot(idx, num_classes=T).bool()
                time_neg = time_neg.unsqueeze(1).expand(B, B, T)

                def masked_logsumexp(a, mask):
                    B, _, _ = a.shape
                    a_masked = a.masked_fill(~mask, float("-inf"))
                    return torch.logsumexp(a_masked.reshape(B, -1), dim=-1)

                same_traj_neg = traj_eq & time_neg  # B, B, T
                other_traj_neg = ~traj_eq  # B, B, T
                loss_positive = -sim[torch.arange(B), torch.arange(B), idx]  # B,
                loss_same_traj_neg = masked_logsumexp(sim, same_traj_neg)  # B,
                loss_other_traj_neg = masked_logsumexp(sim, other_traj_neg)  # B,
                loss_neg = (
                    self.alpha * loss_same_traj_neg
                    + (1 - self.alpha) * loss_other_traj_neg
                )  # B,
                loss = (loss_positive + loss_neg).mean()

        return loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        _unpack_batch = _unpack_transition if self.k == 1 else _unpack_trajectory
        obs, obs_next, action = _unpack_batch(batch, device, self.num_actions)
        loss = self.contrastive_loss(obs, obs_next, action, device)
        metrics = {"train_loss": loss.item()}
        return loss, metrics

    def validation_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        _unpack_batch = _unpack_transition if self.k == 1 else _unpack_trajectory
        obs, obs_next, action = _unpack_batch(batch, device, self.num_actions)
        loss = self.contrastive_loss(obs, obs_next, action, device)
        metrics = {"val_loss": loss.item()}
        return loss, metrics

    def get_optimizer_config(self, cfg: Any) -> Dict[str, Any]:
        return {
            "type": "AdamW",
            "lr": cfg.train.contrastive.lr,
            "weight_decay": cfg.train.contrastive.weight_decay,
            "gradient_clip": cfg.train.contrastive.gradient_clip,
        }

    @property
    def id(self) -> str:
        return "contrastive"

    def get_encoder_fn(self):
        """Return a function that encodes input states to latent representations"""

        def encode_fn(x):
            return self.phi(x)

        return encode_fn


class DynamicsModel(TrainableModel):
    def __init__(
        self,
        z_dim: int,
        num_actions: int,
        widths,
        repr_method: str,
        activation: str = "relu",
    ):
        super().__init__()
        self.repr_method = repr_method
        self.z_dim = z_dim
        self.num_actions = num_actions
        self.net = make_mlp([z_dim + num_actions] + list(widths) + [z_dim], activation)
        self.mse = nn.MSELoss()
        self.encoder_fn = None  # Will be set during training setup

    def forward(self, z, a_onehot):
        return self.net(torch.cat([z, a_onehot], dim=-1))

    def set_encoder(self, encoder_fn):
        """Set the encoder function to use for getting latent representations"""
        self.encoder_fn = encoder_fn

    def training_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs, obs_next, action = _unpack_transition(batch, device, self.num_actions)

        with torch.no_grad():
            z = self.encoder_fn(obs)  # type: ignore
            z_next_true = self.encoder_fn(obs_next)  # type: ignore

        z_next_pred = self(z, action)
        loss = self.mse(z_next_pred, z_next_true)

        metrics = {"train_mse": loss.item()}
        return loss, metrics

    def validation_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs, obs_next, action = _unpack_transition(batch, device, self.num_actions)

        with torch.no_grad():
            z = self.encoder_fn(obs)  # type: ignore
            z_next_true = self.encoder_fn(obs_next)  # type: ignore

        z_next_pred = self(z, action)
        loss = self.mse(z_next_pred, z_next_true)

        metrics = {"val_mse": loss.item()}
        return loss, metrics

    def get_optimizer_config(self, cfg: Any) -> Dict[str, Any]:
        return {
            "type": "AdamW",
            "lr": cfg.train.dynamics.lr,
            "weight_decay": cfg.train.dynamics.weight_decay,
            "gradient_clip": cfg.train.dynamics.gradient_clip,
        }

    def save_checkpoint(self, ckpt_dir: str, cfg: Any) -> Dict[str, Any]:
        """Save dynamics checkpoint and return info for wandb artifacts"""
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "dynamics.pt")
        torch.save({"state_dict": self.state_dict(), "cfg": cfg}, ckpt_path)

        return {
            "paths": [ckpt_path],
            "artifact_name": "dynamics",
            "artifact_type": "model",
        }

    @property
    def id(self) -> str:
        return "dynamics"


class Probe(TrainableModel):
    def __init__(
        self,
        z_dim: int,
        signal_dim: int,
        repr_method: str,
        widths=(64, 64),
        activation: str = "relu",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.signal_dim = signal_dim
        self.repr_method = repr_method
        self.net = make_mlp([z_dim] + list(widths) + [signal_dim], activation)
        self.mse = nn.MSELoss()
        self.encoder_fn = None  # Will be set during training setup

    def forward(self, z):
        return self.net(z)

    def set_encoder(self, encoder_fn):
        """Set the encoder function to use for getting latent representations"""
        self.encoder_fn = encoder_fn

    def training_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs = batch["observation"].to(device)
        signal_true = batch["signal"].to(device)

        with torch.no_grad():
            z = self.encoder_fn(obs)  # type: ignore

        signal_pred = self(z)
        loss = self.mse(signal_pred, signal_true)

        metrics = {"train_mse": loss.item()}
        return loss, metrics

    def validation_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        obs = batch["observation"].to(device)
        signal_true = batch["signal"].to(device)

        with torch.no_grad():
            z = self.encoder_fn(obs)  # type: ignore

        signal_pred = self(z)
        loss = self.mse(signal_pred, signal_true)

        metrics = {"val_mse": loss.item()}
        return loss, metrics

    def get_optimizer_config(self, cfg: Any) -> Dict[str, Any]:
        return {
            "type": "AdamW",
            "lr": cfg.train.probe.lr,
            "weight_decay": cfg.train.probe.weight_decay,
            "gradient_clip": cfg.train.probe.gradient_clip,
        }

    def save_checkpoint(self, ckpt_dir: str, cfg: Any) -> Dict[str, Any]:
        """Save probe checkpoint and return info for wandb artifacts"""
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "probe.pt")
        torch.save({"state_dict": self.state_dict(), "cfg": cfg}, ckpt_path)

        return {
            "paths": [ckpt_path],
            "artifact_name": "probe",
            "artifact_type": "model",
        }

    @property
    def id(self) -> str:
        return "probe"
