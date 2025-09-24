import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import omegaconf
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data import TriplesNPZ
from models import VAE, NachumModel, DynamicsModel, Probe
from utils import seed_all
from trainer import GenericTrainer
from config_schemas import Config, register_configs
from generate_data import main as gen_data_main
from eval import main as eval_main


def create_vae(cfg: DictConfig, device: torch.device) -> VAE:
    """Create VAE model from config"""
    D = cfg["data"]["signal_dim"] + cfg["data"]["noise_dim"]
    return VAE(
        D,
        cfg["model"]["z_dim_vae"],
        cfg["model"]["enc_widths"],
        cfg["model"]["dec_widths"],
        beta=cfg["train"]["vae"]["beta"],
        activation=cfg["model"].get("activation", "relu"),
    ).to(device)


def create_contrastive_trainer(cfg: DictConfig, device: torch.device) -> NachumModel:
    """Create ContrastiveTrainer model from config"""
    D = cfg["data"]["signal_dim"] + cfg["data"]["noise_dim"]
    return NachumModel(
        D,
        cfg["model"]["z_dim_contrastive"],
        cfg["data"]["num_actions"],
        cfg["model"]["enc_widths"],
        cfg["model"]["proj_widths"],
        temperature=cfg["train"]["contrastive"]["temperature"],
        activation=cfg["model"].get("activation", "relu"),
    ).to(device)


def create_dynamics(
    cfg: DictConfig, device: torch.device, z_space: str
) -> DynamicsModel:
    """Create Dynamics model from config"""
    z_dim = cfg["model"][f"z_dim_{z_space}"]
    return DynamicsModel(
        z_dim,
        cfg["data"]["num_actions"],
        cfg["model"]["dyn_widths"],
        z_space=z_space,
        activation=cfg["model"].get("activation", "relu"),
    ).to(device)


def create_probe(cfg: DictConfig, device: torch.device, z_space: str) -> Probe:
    """Create Probe model from config"""
    z_dim = cfg["model"][f"z_dim_{z_space}"]
    return Probe(
        z_dim,
        cfg["data"]["signal_dim"],
        z_space=z_space,
        widths=cfg["model"]["probe_widths"],
        activation=cfg["model"].get("activation", "relu"),
    ).to(device)


def load_encoder(
    cfg: DictConfig, device: torch.device, z_space: str, freeze: bool = True
):
    """Load pre-trained encoder and return model + encoding function"""
    if z_space == "vae":
        model = create_vae(cfg, device)
        ckpt_path = os.path.join(cfg["train"]["ckpt_dir"], "vae.pt")
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
        )

        def encode_fn(s):
            mu, logvar, z = model.encode(s)
            return mu

    elif z_space == "contrastive":
        model = create_contrastive_trainer(cfg, device)
        ckpt_path = os.path.join(cfg["train"]["ckpt_dir"], "contrastive_phi.pt")
        model.phi.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
        )

        def encode_fn(s):
            return model.phi(s)

    else:
        raise ValueError("z_space must be 'vae' or 'contrastive'")

    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    return model, encode_fn


def _maybe_init_wandb(cfg: DictConfig):
    """Initialize wandb if enabled"""
    if not cfg["wandb"]["enabled"]:
        return None

    config_dict = cfg
    if isinstance(cfg, omegaconf.dictconfig.DictConfig):
        config_dict = OmegaConf.to_container(cfg, resolve=True)

    run = wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        group=cfg["wandb"]["group"],
        mode=cfg["wandb"]["mode"],
        dir=cfg["wandb"]["dir"],
        tags=cfg["wandb"]["tags"],
        config=config_dict,
        name=os.environ.get("SLURM_JOB_NAME", None) or None,
        resume="allow",
    )
    return run


def make_loaders(cfg):
    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]
    dd = cfg["data"]["out_dir"]
    train = DataLoader(
        TriplesNPZ(os.path.join(dd, "train.npz")),
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
    )
    val = DataLoader(
        TriplesNPZ(os.path.join(dd, "val.npz")),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )
    return train, val


def train_phase1_vae(cfg: DictConfig, device: torch.device, wb):
    """Refactored VAE training using GenericTrainer"""
    train_loader, val_loader = make_loaders(cfg)

    # Create model and trainer
    model = create_vae(cfg, device)
    trainer = GenericTrainer(cfg, device, wb)

    # Training message
    initial_message = (
        f"Training VAE. Epochs: {cfg['train']['epochs_phase1']}; "
        f"Batch size: {cfg['train']['batch_size']}; "
        f"Steps per epoch: {len(train_loader)}."
    )
    print(initial_message)

    # Train
    final_metrics = trainer.train(
        model,
        train_loader,
        val_loader,
        cfg["train"]["epochs_phase1"],
    )

    return os.path.join(cfg["train"]["ckpt_dir"], "enc_vae.pt"), final_metrics


def train_phase1_contrastive(cfg: DictConfig, device: torch.device, wb):
    """Refactored contrastive training using GenericTrainer"""
    train_loader, val_loader = make_loaders(cfg)

    # Create model and trainer
    model = create_contrastive_trainer(cfg, device)
    trainer = GenericTrainer(cfg, device, wb)

    # Train
    final_metrics = trainer.train(
        model, train_loader, val_loader, cfg["train"]["epochs_phase1"]
    )

    phi_path = os.path.join(cfg["train"]["ckpt_dir"], "contrastive_phi.pt")
    g_path = os.path.join(cfg["train"]["ckpt_dir"], "contrastive_g.pt")

    return phi_path, g_path, final_metrics


def train_phase2_dynamics(cfg: DictConfig, device: torch.device, z_space: str, wb):
    """Refactored dynamics training using GenericTrainer"""
    train_loader, val_loader = make_loaders(cfg)

    # Load pre-trained encoder
    encoder_model, z_of = load_encoder(cfg, device, z_space, freeze=True)

    # Create dynamics model and set encoder
    model = create_dynamics(cfg, device, z_space)
    model.set_encoder(z_of)

    # Create trainer and train
    trainer = GenericTrainer(cfg, device, wb)
    final_metrics = trainer.train(
        model, train_loader, val_loader, cfg["train"]["epochs_phase2"]
    )

    out_path = os.path.join(cfg["train"]["ckpt_dir"], f"dyn_{z_space}.pt")
    return out_path, final_metrics


def train_probes(cfg: DictConfig, device: torch.device, z_space: str, wb):
    """Refactored probe training using GenericTrainer"""
    train_loader, val_loader = make_loaders(cfg)

    # Load pre-trained encoder
    encoder_model, z_of = load_encoder(cfg, device, z_space, freeze=True)

    # Create probe model and set encoder
    model = create_probe(cfg, device, z_space)
    model.set_encoder(z_of)

    # Create trainer and train
    trainer = GenericTrainer(cfg, device, wb)
    final_metrics = trainer.train(
        model, train_loader, val_loader, cfg["train"]["epochs_probe"]
    )

    out_path = os.path.join(cfg["train"]["ckpt_dir"], f"probe_{z_space}.pt")
    return out_path, final_metrics


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    register_configs()

    print("=== Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    print("====================")

    device = torch.device(cfg["train"]["device"])
    seed_all(cfg.data.seed)
    wb = _maybe_init_wandb(cfg)

    # This allows commands like: python train_hydra.py +gen_data=true +phase1=vae +phase2=vae +train_probes=vae
    gen_data = cfg.get("gen_data", default_value=False)
    phase1 = cfg.get("phase1", None)
    phase2 = cfg.get("phase2", None)
    train_probes_arg = cfg.get("train_probes", None)
    eval_arg = cfg.get("eval", None)

    if gen_data:
        gen_data_main(cfg)

    if phase1:
        if phase1 == "vae":
            train_phase1_vae(cfg, device, wb)
        elif phase1 == "contrastive":
            train_phase1_contrastive(cfg, device, wb)
        else:
            raise ValueError("phase1 must be 'vae' or 'contrastive'")

    if phase2:
        train_phase2_dynamics(cfg, device, z_space=phase2, wb=wb)

    if train_probes_arg:
        train_probes(cfg, device, z_space=train_probes_arg, wb=wb)

    if eval_arg:
        eval_main(cfg, eval_arg, wb)

    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
