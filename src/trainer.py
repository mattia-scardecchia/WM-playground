import os
import logging
from models import TrainableModel
import torch
import wandb
from torch.utils.data import DataLoader
from typing import Dict, Any, Union, Optional
from utils import AverageMeter


class GenericTrainer:
    """Generic trainer that works with any TrainableModel"""

    def __init__(self, cfg: Any, device: torch.device, wb: Optional[Any] = None):
        self.cfg = cfg
        self.device = device
        self.wb = wb

    def setup_optimizer(
        self, model: TrainableModel, opt_config: Dict[str, Any]
    ) -> torch.optim.Optimizer:
        """Create optimizer based on configuration"""
        assert opt_config["type"] == "AdamW"
        self.gradient_clip_config = opt_config["gradient_clip"]
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_config["lr"],
            weight_decay=opt_config["weight_decay"],
        )

    def train_epoch(
        self,
        model: TrainableModel,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Train for one epoch and return averaged metrics"""
        model.train()
        meter = AverageMeter()

        for batch in train_loader:
            loss, metrics = model.training_step(batch, self.device)

            optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip_config.enabled:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=self.gradient_clip_config.max_norm,
                    norm_type=self.gradient_clip_config.norm_type,
                )
            optimizer.step()

            batch_size = batch["s"].size(0)
            meter.update(metrics, {key: batch_size for key in metrics.keys()})

        return meter.avg

    @torch.inference_mode()
    def eval_epoch(
        self, model: TrainableModel, val_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate for one epoch and return averaged metrics"""
        model.eval()
        meter = AverageMeter()

        for batch in val_loader:
            loss, metrics = model.validation_step(batch, self.device)

            batch_size = batch["s"].size(0)
            meter.update(metrics, {key: batch_size for key in metrics.keys()})

        return meter.avg

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for console logging"""
        metric_strs = [f"{k}={v:.4f}" for k, v in metrics.items()]
        return (
            f"{', '.join(metric_strs)}"
            if len(metric_strs) > 1
            else f"{metric_strs[0]}"
            if metric_strs
            else ""
        )

    def log_epoch(
        self, model, epoch: int, metrics: Dict[str, float], phase: str = "train"
    ):
        """Log epoch results to console and wandb"""
        phase_name = model.id
        metrics_str = self._format_metrics(metrics)

        # Console logging
        if phase == "val":
            logging.info(f"[{phase_name.upper()}-VAL] Epoch {epoch + 1}: {metrics_str}")
        else:
            logging.info(f"[{phase_name.upper()}] Epoch {epoch + 1}: {metrics_str}")

        # Wandb logging
        if self.wb is not None:
            wandb_metrics = {f"{phase_name}/{k}": v for k, v in metrics.items()}
            wandb_metrics.update({f"{phase_name}/epoch": epoch + 1})
            commit = phase == "val"
            wandb.log(wandb_metrics, commit=commit)

    def save_checkpoint(self, model) -> list:
        """Save model checkpoint and return path(s)"""
        # Save checkpoint locally
        checkpoint_info = model.save_checkpoint(self.cfg.train.ckpt_dir, self.cfg)
        checkpoint_paths = checkpoint_info["paths"]
        artifact_name = checkpoint_info["artifact_name"]
        artifact_type = checkpoint_info["artifact_type"]
        for path in checkpoint_paths:
            logging.info(f"Saved checkpoint: {path}")

        # Handle wandb artifacts
        if self.wb is not None:
            art = wandb.Artifact(artifact_name, type=artifact_type)
            for path in checkpoint_paths:
                art.add_file(path)
            wandb.log_artifact(art)

        return checkpoint_paths

    def train(
        self,
        model: TrainableModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> Dict[str, float]:
        """Full training loop"""
        logging.info(f"Starting training for {model.id} model ({num_epochs} epochs)")
        opt_config = model.get_optimizer_config(self.cfg)
        optimizer = self.setup_optimizer(model, opt_config)
        if self.wb is not None:
            log_freq = self.cfg.wandb.log_freq
            if model.id in ["vae", "contrastive"]:
                wandb.watch(model, log="all", log_freq=log_freq)
        final_metrics, train_metrics, val_metrics = {}, {}, {}

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(model, train_loader, optimizer)
            self.log_epoch(model, epoch, train_metrics, phase="train")
            val_metrics = self.eval_epoch(model, val_loader)
            self.log_epoch(model, epoch, val_metrics, phase="val")

        final_metrics = {
            f"final_{model.id}_train_{k}": v for k, v in train_metrics.items()
        }
        final_metrics.update(
            {f"final_{model.id}_val_{k}": v for k, v in val_metrics.items()}
        )
        self.save_checkpoint(model)
        logging.info(f"Completed training for {model.id} model")
        return final_metrics
