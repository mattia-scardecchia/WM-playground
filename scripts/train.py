import os
import argparse
import yaml
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data import TriplesNPZ
from models import VAE, NachumModel, DynamicsModel, Probe
from utils import seed_all
from trainer import GenericTrainer


def create_vae(cfg, device):
    """Create VAE model from config"""
    return VAE(
        cfg["data"]["D"],
        cfg["model"]["z_dim_vae"],
        cfg["model"]["enc_widths"],
        cfg["model"]["dec_widths"],
        beta=cfg["train"]["vae"]["beta"],
        activation=cfg["model"].get("activation", "relu"),
    ).to(device)


def create_contrastive_trainer(cfg, device):
    """Create ContrastiveTrainer model from config"""
    return NachumModel(
        cfg["data"]["D"],
        cfg["model"]["z_dim_contrastive"],
        cfg["model"]["enc_widths"],
        cfg["model"]["proj_widths"],
        temperature=cfg["train"]["contrastive"]["temperature"],
        activation=cfg["model"].get("activation", "relu"),
    ).to(device)


def create_dynamics(cfg, device, z_space):
    """Create Dynamics model from config"""
    z_dim = cfg["model"][f"z_dim_{z_space}"]
    return DynamicsModel(
        z_dim,
        4,  # a_dim
        cfg["model"]["dyn_widths"],
        z_space=z_space,
        activation=cfg["model"].get("activation", "relu"),
    ).to(device)


def create_probe(cfg, device, z_space):
    """Create Probe model from config"""
    z_dim = cfg["model"][f"z_dim_{z_space}"]
    return Probe(
        z_dim,
        z_space=z_space,
        widths=cfg["model"]["probe_widths"],
        activation=cfg["model"].get("activation", "relu"),
    ).to(device)


def load_encoder(cfg, device, z_space, freeze=True):
    """Load pre-trained encoder and return model + encoding function"""
    if z_space == "vae":
        model = create_vae(cfg, device)
        ckpt_path = os.path.join(cfg["train"]["ckpt_dir"], "vae.pt")
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])

        def encode_fn(s):
            mu, logvar, z = model.encode(s)
            return mu

    elif z_space == "contrastive":
        model = create_contrastive_trainer(cfg, device)
        ckpt_path = os.path.join(cfg["train"]["ckpt_dir"], "contrastive_phi.pt")
        model.phi.load_state_dict(
            torch.load(ckpt_path, map_location=device)["state_dict"]
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


def _maybe_init_wandb(cfg):
    wcfg = cfg.get("wandb", {})
    if not wcfg or not wcfg.get("enabled", False):
        return None
    run = wandb.init(
        project=wcfg.get("project", "repr-world"),
        entity=wcfg.get("entity"),
        group=wcfg.get("group"),
        mode=wcfg.get("mode", "online"),
        dir=wcfg.get("dir", None),
        tags=wcfg.get("tags", []),
        config=cfg,
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


def train_phase1_vae(cfg, device, wb):
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


def train_phase1_contrastive(cfg, device, wb):
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


def train_phase2_dynamics(cfg, device, z_space: str, wb):
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


def train_probes(cfg, device, z_space: str, wb):
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


def main(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg["data"]["seed"])
    wb = _maybe_init_wandb(cfg)

    if args.phase1:
        if args.phase1 == "vae":
            train_phase1_vae(cfg, device, wb)
        elif args.phase1 == "contrastive":
            train_phase1_contrastive(cfg, device, wb)
        else:
            raise ValueError("--phase1 in {vae, contrastive}")

    if args.phase2:
        train_phase2_dynamics(cfg, device, z_space=args.phase2, wb=wb)

    if args.train_probes:
        train_probes(cfg, device, z_space=args.train_probes, wb=wb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--phase1", type=str, choices=["vae", "contrastive"], default=None)
    ap.add_argument("--phase2", type=str, choices=["vae", "contrastive"], default=None)
    ap.add_argument(
        "--train-probes",
        dest="train_probes",
        type=str,
        choices=["vae", "contrastive"],
        default=None,
    )
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg, args)
