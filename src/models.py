from typing import Sequence, Dict, Any, Tuple, Optional, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from utils import make_mlp, to_onehot


def _unpack_batch(
    batch: Dict[str, torch.Tensor], device: torch.device, num_actions: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper function to prepare batch data for dynamics and contrastive models"""
    s = batch["s"].to(device)
    sp = batch["sp"].to(device)
    a = batch["a"].to(device)
    a1h = to_onehot(a, num_actions).to(device)
    return s, sp, a, a1h


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
    def get_optimizer_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Return optimizer configuration for this model"""
        pass

    @abstractmethod
    def save_checkpoint(self, ckpt_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save model checkpoint(s) and return info for wandb artifacts
        Returns: {"paths": list, "artifact_name": str, "artifact_type": str}
        """
        pass

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
        x = batch["s"].to(device)
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
        x = batch["s"].to(device)
        x_hat, mu, logvar, z = self(x)
        loss, parts = beta_vae_loss(x_hat, x, mu, logvar, self.beta)

        metrics = {
            "val_loss": loss.item(),
            "val_recon": parts["recon"],
            "val_kl": parts["kl"],
        }
        return loss, metrics

    def get_optimizer_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "AdamW",
            "lr": cfg["train"]["vae"]["lr"],
            "weight_decay": cfg["train"]["vae"]["weight_decay"],
        }

    def save_checkpoint(self, ckpt_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Save VAE checkpoint and return info for wandb artifacts"""
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
    def id(self) -> str:
        return "vae"


class NachumModel(TrainableModel):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        enc_widths,
        proj_widths,
        temperature: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.temperature = temperature
        self.phi = nn.Sequential(
            make_mlp([x_dim] + list(enc_widths) + [z_dim], activation)
        )
        self.g = nn.Sequential(
            make_mlp([x_dim + 4] + list(proj_widths) + [z_dim], activation)  # 4 = a_dim
        )

    def forward(self, s):
        return self.phi(s)

    def project_next(self, sp, a_onehot):
        return self.g(torch.cat([sp, a_onehot], dim=-1))

    def training_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        s, sp, a, a1h = _unpack_batch(batch, device)

        z = self.phi(s)  # B, D
        zpos = self.g(torch.cat([sp, a1h], dim=-1))  # B, D
        diff = z[:, None, :] - zpos[None, :, :]  # B, B, D
        logits = -torch.sum(diff**2, dim=-1) / self.temperature
        target = torch.arange(z.size(0), device=device)
        loss = F.cross_entropy(logits, target)

        metrics = {"train_loss": loss.item()}
        return loss, metrics

    def validation_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        s, sp, a, a1h = _unpack_batch(batch, device)

        z = self.phi(s)  # B, D
        zpos = self.g(torch.cat([sp, a1h], dim=-1))  # B, D
        diff = z[:, None, :] - zpos[None, :, :]  # B, B, D
        logits = -torch.sum(diff**2, dim=-1) / self.temperature
        target = torch.arange(z.size(0), device=device)
        loss = F.cross_entropy(logits, target)

        metrics = {"val_loss": loss.item()}
        return loss, metrics

    def get_optimizer_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "AdamW",
            "lr": cfg["train"]["contrastive"]["lr"],
            "weight_decay": cfg["train"]["contrastive"]["weight_decay"],
        }

    def save_checkpoint(self, ckpt_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Save contrastive model checkpoint and return info for wandb artifacts"""
        os.makedirs(ckpt_dir, exist_ok=True)

        phi_path = os.path.join(ckpt_dir, "contrastive_phi.pt")
        g_path = os.path.join(ckpt_dir, "contrastive_g.pt")

        torch.save({"state_dict": self.phi.state_dict(), "cfg": cfg}, phi_path)
        torch.save({"state_dict": self.g.state_dict(), "cfg": cfg}, g_path)

        return {
            "paths": [phi_path, g_path],
            "artifact_name": "contrastive-encoders",
            "artifact_type": "model",
        }

    @property
    def id(self) -> str:
        return "contrastive"


class DynamicsModel(TrainableModel):
    def __init__(
        self, z_dim: int, a_dim: int, widths, z_space: str, activation: str = "relu"
    ):
        super().__init__()
        self.z_space = z_space
        self.net = make_mlp([z_dim + a_dim] + list(widths) + [z_dim], activation)
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
        s, sp, a, a1h = _unpack_batch(batch, device)

        with torch.no_grad():
            z = self.encoder_fn(s)
            z_next_true = self.encoder_fn(sp)

        z_next_pred = self(z, a1h)
        loss = self.mse(z_next_pred, z_next_true)

        metrics = {"train_mse": loss.item()}
        return loss, metrics

    def validation_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        s, sp, a, a1h = _unpack_batch(batch, device)

        with torch.no_grad():
            z = self.encoder_fn(s)
            z_next_true = self.encoder_fn(sp)

        z_next_pred = self(z, a1h)
        loss = self.mse(z_next_pred, z_next_true)

        metrics = {"val_mse": loss.item()}
        return loss, metrics

    def get_optimizer_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "AdamW",
            "lr": cfg["train"]["dynamics"]["lr"],
            "weight_decay": cfg["train"]["dynamics"]["weight_decay"],
        }

    def save_checkpoint(self, ckpt_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Save dynamics checkpoint and return info for wandb artifacts"""
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"dyn_{self.z_space}.pt")
        torch.save({"state_dict": self.state_dict(), "cfg": cfg}, ckpt_path)

        return {
            "paths": [ckpt_path],
            "artifact_name": f"dyn-{self.z_space}",
            "artifact_type": "model",
        }

    @property
    def id(self) -> str:
        return f"dyn-{self.z_space}"


class Probe(TrainableModel):
    def __init__(
        self, z_dim: int, z_space: str, widths=(64, 64), activation: str = "relu"
    ):
        super().__init__()
        self.z_space = z_space
        self.net = make_mlp([z_dim] + list(widths) + [2], activation)
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
        s = batch["s"].to(device)
        pos_true = s[:, :2]

        with torch.no_grad():
            z = self.encoder_fn(s)

        pos_pred = self(z)
        loss = self.mse(pos_pred, pos_true)

        metrics = {"train_mse": loss.item()}
        return loss, metrics

    def validation_step(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        s = batch["s"].to(device)
        pos_true = s[:, :2]

        with torch.no_grad():
            z = self.encoder_fn(s)

        pos_pred = self(z)
        loss = self.mse(pos_pred, pos_true)

        metrics = {"val_mse": loss.item()}
        return loss, metrics

    def get_optimizer_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "AdamW",
            "lr": cfg["train"]["probe"]["lr"],
            "weight_decay": cfg["train"]["probe"]["weight_decay"],
        }

    def save_checkpoint(self, ckpt_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Save probe checkpoint and return info for wandb artifacts"""
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"probe_{self.z_space}.pt")
        torch.save({"state_dict": self.state_dict(), "cfg": cfg}, ckpt_path)

        return {
            "paths": [ckpt_path],
            "artifact_name": f"probe-{self.z_space}",
            "artifact_type": "model",
        }

    @property
    def id(self) -> str:
        return f"probe-{self.z_space}"
