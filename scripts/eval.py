import os
import argparse
import yaml
import wandb
import torch
from torch.utils.data import DataLoader

from data import TriplesNPZ
from models import VAE, NachumModel, DynamicsModel, Probe
from utils import seed_all


def load_components(cfg, path: str, device):
    act = cfg["model"].get("activation", "relu")
    if path == "vae":
        z_dim = cfg["model"]["z_dim_vae"]
        vae = VAE(
            cfg["data"]["D"],
            z_dim,
            cfg["model"]["enc_widths"],
            cfg["model"]["dec_widths"],
            beta=cfg["train"]["vae"]["beta"],
            activation=act,
        )
        vae.load_state_dict(
            torch.load(
                os.path.join(cfg["train"]["ckpt_dir"], "vae.pt"), map_location=device
            )["state_dict"]
        )
        vae.to(device).eval()
        dyn = DynamicsModel(
            z_dim, 4, cfg["model"]["dyn_widths"], z_space="vae", activation=act
        )
        dyn.load_state_dict(
            torch.load(
                os.path.join(cfg["train"]["ckpt_dir"], "dyn_vae.pt"),
                map_location=device,
            )["state_dict"]
        )
        dyn.to(device).eval()
        probe = Probe(
            z_dim, z_space="vae", widths=cfg["model"]["probe_widths"], activation=act
        )
        probe.load_state_dict(
            torch.load(
                os.path.join(cfg["train"]["ckpt_dir"], "probe_vae.pt"),
                map_location=device,
            )["state_dict"]
        )
        probe.to(device).eval()

        def encode(x):
            with torch.no_grad():
                mu, logvar, z = vae.encode(x)
                return mu
    elif path == "contrastive":
        z_dim = cfg["model"]["z_dim_contrastive"]
        contrastive_model = NachumModel(
            cfg["data"]["D"],
            z_dim,
            cfg["model"]["enc_widths"],
            cfg["model"]["proj_widths"],
            temperature=cfg["train"]["contrastive"]["temperature"],
            activation=act,
        )
        contrastive_model.phi.load_state_dict(
            torch.load(
                os.path.join(cfg["train"]["ckpt_dir"], "contrastive_phi.pt"),
                map_location=device,
            )["state_dict"]
        )
        contrastive_model.to(device).eval()
        dyn = DynamicsModel(
            z_dim, 4, cfg["model"]["dyn_widths"], z_space="contrastive", activation=act
        )
        dyn.load_state_dict(
            torch.load(
                os.path.join(cfg["train"]["ckpt_dir"], "dyn_contrastive.pt"),
                map_location=device,
            )["state_dict"]
        )
        dyn.to(device).eval()
        probe = Probe(
            z_dim,
            z_space="contrastive",
            widths=cfg["model"]["probe_widths"],
            activation=act,
        )
        probe.load_state_dict(
            torch.load(
                os.path.join(cfg["train"]["ckpt_dir"], "probe_contrastive.pt"),
                map_location=device,
            )["state_dict"]
        )
        probe.to(device).eval()

        def encode(x):
            with torch.no_grad():
                return contrastive_model.phi(x)
    else:
        raise ValueError("--path in {vae, contrastive}")
    return encode, dyn, probe


def evaluate(cfg, path: str, device):
    dd = cfg["data"]["out_dir"]
    loader = DataLoader(
        TriplesNPZ(os.path.join(dd, "test.npz")),
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )
    encode, dyn, probe = load_components(cfg, path, device)
    mse_sum = 0.0
    n = 0
    for batch in loader:
        s = batch["s"].to(device)
        a = batch["a"].to(device)
        sp = batch["sp"].to(device)
        a1h = torch.nn.functional.one_hot(a.long(), num_classes=4).float()
        with torch.no_grad():
            z = encode(s)
            z_next = dyn(z, a1h)
            pos_pred = probe(z_next)
            pos_true = sp[:, :2]
            mse = torch.mean((pos_pred - pos_true) ** 2).item()
        mse_sum += mse * s.size(0)
        n += s.size(0)
    final = mse_sum / n
    print(f"[EVAL-{path}] next-signal MSE = {final:.6f}")
    wcfg = cfg.get("wandb", {})
    if wcfg.get("enabled", False):
        wandb.log({f"eval/{path}_next_signal_mse": final})
    return {"final_eval_loss": final}


def main(cfg, args):
    wcfg = cfg.get("wandb", {})
    wb = None
    if wcfg.get("enabled", False):
        wb = wandb.init(
            project=wcfg.get("project", "repr-world"),
            entity=wcfg.get("entity"),
            group=f"eval-{args.path}"
            if wcfg.get("group") is None
            else wcfg.get("group"),
            mode=wcfg.get("mode", "online"),
            dir=wcfg.get("dir", None),
            tags=(wcfg.get("tags", []) + ["eval"]),
            config=cfg,
            name=f"eval-{args.path}",
            resume="allow",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg["data"]["seed"])
    evaluate(cfg, args.path, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--path", type=str, choices=["vae", "contrastive"], required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg, args)
