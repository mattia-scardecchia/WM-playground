import logging
from math import log
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import omegaconf
import wandb
import torch
import torch.nn.functional as F
from typing import Optional

from data import DataManager
from models import VAE, NachumModel, DynamicsModel, Probe
from utils import seed_all
from trainer import GenericTrainer
from eval import main as eval_main


def create_vae(cfg: DictConfig, device: torch.device) -> VAE:
    """Create VAE model from config"""
    D = cfg.data.signal_dim + cfg.data.noise_dim
    vae_cfg = cfg.model.vae
    return VAE(
        D,
        vae_cfg.z_dim,
        vae_cfg.enc_widths,
        vae_cfg.dec_widths,
        beta=cfg.train.vae.beta,
        activation=vae_cfg.activation,
    ).to(device)


def create_contrastive_trainer(cfg: DictConfig, device: torch.device) -> NachumModel:
    """Create ContrastiveTrainer model from config"""
    D = cfg.data.signal_dim + cfg.data.noise_dim
    contrastive_cfg = cfg.model.contrastive
    return NachumModel(
        D,
        contrastive_cfg.z_dim,
        cfg.data.num_actions,
        contrastive_cfg.enc_widths,
        contrastive_cfg.proj_widths,
        temperature=cfg.train.contrastive.temperature,
        activation=contrastive_cfg.activation,
    ).to(device)


def create_dynamics(
    cfg: DictConfig, device: torch.device, z_space: str
) -> DynamicsModel:
    """Create Dynamics model from config"""
    z_dim = cfg.model.repr.z_dim
    dynamics_cfg = cfg.model.dynamics
    return DynamicsModel(
        z_dim,
        cfg.data.num_actions,
        dynamics_cfg.dyn_widths,
        z_space=z_space,
        activation=dynamics_cfg.activation,
    ).to(device)


def create_probe(cfg: DictConfig, device: torch.device, z_space: str) -> Probe:
    """Create Probe model from config"""
    z_dim = cfg.model.repr.z_dim
    probe_cfg = cfg.model.probe
    return Probe(
        z_dim,
        cfg.data.signal_dim,
        z_space=z_space,
        widths=probe_cfg.probe_widths,
        activation=probe_cfg.activation,
    ).to(device)


def load_encoder(
    cfg: DictConfig,
    device: torch.device,
    repr_method: str,
    freeze: bool,
    ckpt_path: str,
):
    """Load pre-trained encoder and return model + encoding function"""
    factory_fn = create_vae if repr_method == "vae" else create_contrastive_trainer
    model = factory_fn(cfg, device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    )
    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    return model


def _maybe_init_wandb(cfg: DictConfig):
    """Initialize wandb if enabled"""
    if not cfg.wandb.enabled:
        logging.info("Wandb logging disabled")
        return None

    logging.info(f"Initializing wandb project: {cfg.wandb.project}")
    config_dict = cfg
    if isinstance(cfg, omegaconf.dictconfig.DictConfig):
        config_dict = OmegaConf.to_container(cfg, resolve=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        dir=cfg.wandb.dir,
        tags=cfg.wandb.tags,
        config=config_dict,
        name=os.environ.get("SLURM_JOB_NAME", None) or None,
        resume="allow",
    )
    logging.info(f"Wandb run initialized: {run.name}")
    return run


def train_phase1_vae(cfg: DictConfig, device: torch.device, wb):
    """VAE training using GenericTrainer"""
    train_loader, val_loader = DataManager(cfg).get_data_loaders()
    model = create_vae(cfg, device)
    trainer = GenericTrainer(cfg, device, wb)

    final_metrics = trainer.train(
        model,
        train_loader,
        val_loader,
        cfg.train.epochs_phase1,
    )

    return os.path.join(cfg.train.ckpt_dir, "enc_vae.pt"), final_metrics


def train_phase1_contrastive(cfg: DictConfig, device: torch.device, wb):
    """Contrastive training using GenericTrainer"""
    train_loader, val_loader = DataManager(cfg).get_data_loaders()
    model = create_contrastive_trainer(cfg, device)
    trainer = GenericTrainer(cfg, device, wb)

    final_metrics = trainer.train(
        model,
        train_loader,
        val_loader,
        cfg.train.epochs_phase1,
    )

    phi_path = os.path.join(cfg.train.ckpt_dir, "contrastive_phi.pt")
    g_path = os.path.join(cfg.train.ckpt_dir, "contrastive_g.pt")

    return phi_path, g_path, final_metrics


def train_phase2_dynamics(
    cfg: DictConfig, device: torch.device, repr_method: str, wb, ckpt_dir: str
):
    """Dynamics training using GenericTrainer"""
    train_loader, val_loader = DataManager(cfg).get_data_loaders()
    if ckpt_dir is None:
        ckpt_dir = cfg.train.ckpt_dir
    ckpt_path = os.path.join(
        ckpt_dir,
        f"{repr_method}.pt",
    )
    logging.info(f"Loading encoder from {ckpt_path} for dynamics training")
    encoder_model = load_encoder(
        cfg,
        device,
        repr_method,
        freeze=True,
        ckpt_path=ckpt_path,
    )
    encoder_fn = encoder_model.get_encoder_fn()

    model = create_dynamics(cfg, device, repr_method)
    model.set_encoder(encoder_fn)
    trainer = GenericTrainer(cfg, device, wb)
    final_metrics = trainer.train(
        model, train_loader, val_loader, cfg.train.epochs_phase2
    )

    out_path = os.path.join(cfg.train.ckpt_dir, f"dyn_{repr_method}.pt")
    return out_path, final_metrics


def train_probes(
    cfg: DictConfig, device: torch.device, repr_method: str, wb, ckpt_dir: str
):
    """Refactored probe training using GenericTrainer"""
    train_loader, val_loader = DataManager(cfg).get_data_loaders()

    if ckpt_dir is None:
        ckpt_dir = cfg.train.ckpt_dir
    ckpt_path = os.path.join(
        ckpt_dir,
        f"{repr_method}.pt",
    )
    logging.info(f"Loading encoder from {ckpt_path} for probe training")
    encoder_model = load_encoder(
        cfg,
        device,
        repr_method,
        freeze=True,
        ckpt_path=ckpt_path,
    )
    encoder_fn = encoder_model.get_encoder_fn()

    model = create_probe(cfg, device, repr_method)
    model.set_encoder(encoder_fn)
    trainer = GenericTrainer(cfg, device, wb)
    final_metrics = trainer.train(
        model, train_loader, val_loader, cfg.train.epochs_probe
    )

    out_path = os.path.join(cfg.train.ckpt_dir, f"probe_{repr_method}.pt")
    return out_path, final_metrics


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Set checkpoint directory to hydra output directory
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir
    cfg.train.ckpt_dir = os.path.join(hydra_output_dir, "ckpts")

    logging.info("=== Configuration ===")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    logging.info("====================")

    device = torch.device(cfg.train.device)
    logging.info(f"Using device: {device}")
    seed_all(cfg.data.seed)
    logging.info(f"Set random seed to {cfg.data.seed}")
    wb = _maybe_init_wandb(cfg)

    # Phase 1: Representation learning
    logging.info(f"Starting Phase 1: {cfg.method} representation learning")
    if cfg.method == "vae":
        train_phase1_vae(cfg, device, wb)
    elif cfg.method == "contrastive":
        train_phase1_contrastive(cfg, device, wb)
    else:
        raise ValueError("method must be 'vae' or 'contrastive'")
    logging.info(f"Completed Phase 1: {cfg.method} representation learning")

    # Phase 2: Dynamics learning
    logging.info("Starting Phase 2: Dynamics learning")
    train_phase2_dynamics(
        cfg, device, repr_method=cfg.method, wb=wb, ckpt_dir=cfg.train.ckpt_dir
    )
    logging.info("Completed Phase 2: Dynamics learning")

    # Phase 3: Probe training
    logging.info("Starting Phase 3: Probe training")
    train_probes(
        cfg, device, repr_method=cfg.method, wb=wb, ckpt_dir=cfg.train.ckpt_dir
    )
    logging.info("Completed Phase 3: Probe training")

    # Evaluation
    logging.info("Starting evaluation")
    eval_main(cfg, cfg.method, wb)
    logging.info("Completed evaluation")

    if wb is not None:
        logging.info("Finishing wandb run")
        wb.finish()

    logging.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
