import os
import argparse
import yaml
import wandb
import torch
from torch.utils.data import DataLoader

from data import TriplesNPZ
from models import VAE, NachumModel, DynamicsModel, Probe
from utils import seed_all


def load_components(cfg, zdim: str, device, ckpt_dir: str):
    act = cfg["model"].get("activation", "relu")
    if zdim == "vae":
        z_dim = cfg["model"]["z_dim_vae"]
        D = cfg["data"]["signal_dim"] + cfg["data"]["noise_dim"]
        vae = VAE(
            D,
            z_dim,
            cfg["model"]["enc_widths"],
            cfg["model"]["dec_widths"],
            beta=cfg["train"]["vae"]["beta"],
            activation=act,
        )
        vae.load_state_dict(
            torch.load(
                os.path.join(ckpt_dir, "vae.pt"),
                map_location=device,
                weights_only=False,
            )["state_dict"]
        )
        vae.to(device).eval()
        dyn = DynamicsModel(
            z_dim,
            cfg["data"]["num_actions"],
            cfg["model"]["dyn_widths"],
            z_space="vae",
            activation=act,
        )
        dyn.load_state_dict(
            torch.load(
                os.path.join(ckpt_dir, "dyn_vae.pt"),
                map_location=device,
                weights_only=False,
            )["state_dict"]
        )
        dyn.to(device).eval()
        probe = Probe(
            z_dim,
            cfg["data"]["signal_dim"],
            z_space="vae",
            widths=cfg["model"]["probe_widths"],
            activation=act,
        )
        probe.load_state_dict(
            torch.load(
                os.path.join(ckpt_dir, "probe_vae.pt"),
                map_location=device,
                weights_only=False,
            )["state_dict"]
        )
        probe.to(device).eval()

        def encode(x):
            with torch.no_grad():
                mu, logvar, z = vae.encode(x)
                return mu
    elif zdim == "contrastive":
        z_dim = cfg["model"]["z_dim_contrastive"]
        D = cfg["data"]["signal_dim"] + cfg["data"]["noise_dim"]
        contrastive_model = NachumModel(
            D,
            z_dim,
            cfg["data"]["num_actions"],
            cfg["model"]["enc_widths"],
            cfg["model"]["proj_widths"],
            temperature=cfg["train"]["contrastive"]["temperature"],
            activation=act,
        )
        contrastive_model.phi.load_state_dict(
            torch.load(
                os.path.join(ckpt_dir, "contrastive_phi.pt"),
                map_location=device,
                weights_only=False,
            )["state_dict"]
        )
        contrastive_model.to(device).eval()
        dyn = DynamicsModel(
            z_dim,
            cfg["data"]["num_actions"],
            cfg["model"]["dyn_widths"],
            z_space="contrastive",
            activation=act,
        )
        dyn.load_state_dict(
            torch.load(
                os.path.join(ckpt_dir, "dyn_contrastive.pt"),
                map_location=device,
                weights_only=False,
            )["state_dict"]
        )
        dyn.to(device).eval()
        probe = Probe(
            z_dim,
            cfg["data"]["signal_dim"],
            z_space="contrastive",
            widths=cfg["model"]["probe_widths"],
            activation=act,
        )
        probe.load_state_dict(
            torch.load(
                os.path.join(ckpt_dir, "probe_contrastive.pt"),
                map_location=device,
                weights_only=False,
            )["state_dict"]
        )
        probe.to(device).eval()

        def encode(x):
            with torch.no_grad():
                return contrastive_model.phi(x)
    else:
        raise ValueError("--path in {vae, contrastive}")
    return encode, dyn, probe


def evaluate(cfg, zdim: str, device, ckpt_dir: str):
    dd = cfg["data"]["out_dir"]
    eval_batch_size = cfg["train"]["eval_batch_size"]
    num_workers = cfg["train"]["num_workers"]

    loader = DataLoader(
        TriplesNPZ(os.path.join(dd, "test.npz")),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    encode, dyn, probe = load_components(cfg, zdim, device, ckpt_dir)
    mse_sum = 0.0
    n = 0
    for batch in loader:
        s = batch["s"].to(device)
        a = batch["a"].to(device)
        sp = batch["sp"].to(device)
        a1h = torch.nn.functional.one_hot(
            a.long(), num_classes=cfg["data"]["num_actions"]
        ).float()
        with torch.no_grad():
            z = encode(s)
            z_next = dyn(z, a1h)
            pos_pred = probe(z_next)
            pos_true = sp[:, : cfg["data"]["signal_dim"]]
            mse = torch.mean((pos_pred - pos_true) ** 2).item()
        mse_sum += mse * s.size(0)
        n += s.size(0)
    final = mse_sum / n
    print(f"[EVAL-{zdim}] next-signal MSE = {final:.6f}")
    wcfg = cfg.get("wandb", {})
    if wcfg.get("enabled", False):
        wandb.log({f"eval/{zdim}_next_signal_mse": final})
    return {"final_eval_loss": final}


def main(cfg, zdim, wb=None, ckpt_dir=None):
    if ckpt_dir is None:
        ckpt_dir = cfg["train"]["ckpt_dir"]
    wcfg = cfg["wandb"]
    if wcfg["enabled"] and wb is None:
        wandb.init(
            project=wcfg["project"],
            entity=wcfg["entity"],
            group="eval" if wcfg["group"] is None else wcfg["group"],
            mode=wcfg["mode"],
            dir=wcfg["dir"],
            tags=wcfg["tags"] + ["eval"],
            config=cfg,
            name="eval",
            resume="allow",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg["data"]["seed"])
    evaluate(
        cfg,
        zdim,
        device,
        ckpt_dir,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate trained models on test data",
    )
    ap.add_argument(
        "--zdim",
        type=str,
        choices=["vae", "contrastive"],
        required=True,
        help="Model representation space to evaluate",
    )
    ap.add_argument(
        "--hydra-output-dir",
        type=str,
        required=True,
        help="Path to Hydra output directory containing .hydra/config.yaml",
    )
    args = ap.parse_args()

    config_path = os.path.join(args.hydra_output_dir, ".hydra", "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["wandb"]["enabled"] = False  # no wandb for standalone eval
    print(f"Loaded config from: {config_path}")

    main(
        cfg=cfg,
        zdim=args.zdim,
        wb=None,
        ckpt_dir=os.path.join(args.hydra_output_dir, "ckpts"),
    )
