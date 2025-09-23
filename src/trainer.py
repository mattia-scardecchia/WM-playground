import os
import torch
import wandb
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union
from utils import AverageMeter


class GenericTrainer:
    """Generic trainer that works with any TrainableModel"""

    def __init__(
        self, cfg: Dict[str, Any], device: torch.device, wb: Optional[Any] = None
    ):
        self.cfg = cfg
        self.device = device
        self.wb = wb

    def setup_optimizer(
        self, model: torch.nn.Module, opt_config: Dict[str, Any]
    ) -> torch.optim.Optimizer:
        """Create optimizer based on configuration"""
        assert opt_config["type"] == "AdamW"
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_config["lr"],
            weight_decay=opt_config["weight_decay"],
        )

    def train_epoch(
        self, model, train_loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Train for one epoch and return averaged metrics"""
        model.train()
        meter = AverageMeter()

        for batch in train_loader:
            loss, metrics = model.training_step(batch, self.device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics with batch size for proper averaging
            batch_size = batch["s"].size(0)
            meter.update(metrics, {key: batch_size for key in metrics.keys()})

        return meter.avg

    def log_epoch(self, model, epoch: int, metrics: Dict[str, float]):
        """Log epoch results to console and wandb"""
        phase_name = model.id
        metrics_str = model.get_metrics_format(metrics)

        # Console logging
        print(f"[{phase_name.upper()}] Epoch {epoch + 1}: {metrics_str}")

        # Wandb logging
        if self.wb is not None:
            wandb_metrics = {f"{phase_name}/{k}": v for k, v in metrics.items()}
            wandb_metrics.update({f"{phase_name}/epoch": epoch + 1})
            wandb.log(wandb_metrics)

    def save_checkpoint(self, model) -> list:
        """Save model checkpoint and return path(s)"""
        # Let the model handle its own saving logic and return info
        checkpoint_info = model.save_checkpoint(self.cfg["train"]["ckpt_dir"], self.cfg)

        # Extract paths and metadata
        checkpoint_paths = checkpoint_info["paths"]  # Always a list now
        artifact_name = checkpoint_info["artifact_name"]
        artifact_type = checkpoint_info["artifact_type"]

        # Print confirmation
        for path in checkpoint_paths:
            print(f"Saved {path}")

        # Handle wandb artifacts
        if self.wb is not None:
            art = wandb.Artifact(artifact_name, type=artifact_type)
            for path in checkpoint_paths:
                art.add_file(path)
            wandb.log_artifact(art)

        return checkpoint_paths

    def train(
        self,
        model,
        train_loader: DataLoader,
        num_epochs: int,
        initial_message: Optional[str] = None,
    ) -> Dict[str, float]:
        """Full training loop"""
        # Setup
        opt_config = model.get_optimizer_config(self.cfg)
        optimizer = self.setup_optimizer(model, opt_config)
        if self.wb is not None:
            wandb.watch(model, log="all", log_freq=200)

        # Print initial message
        if initial_message:
            print(initial_message)

        final_metrics = {}

        # Training loop
        for epoch in range(num_epochs):
            epoch_metrics = self.train_epoch(model, train_loader, optimizer)
            self.log_epoch(model, epoch, epoch_metrics)

            # Store final epoch metrics
            if epoch == num_epochs - 1:
                final_metrics = {
                    f"final_{model.id}_{k}": v for k, v in epoch_metrics.items()
                }

        # Save checkpoint
        self.save_checkpoint(model)

        return final_metrics
