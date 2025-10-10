import os
import argparse
import signal
import yaml
import wandb
import torch
import logging
from torch.utils.data import DataLoader

from data import TriplesNPZ
from models import VAE, NachumConstrastive, DynamicsModel, Probe, _unpack_batch
from utils import seed_all, AverageMeter


def load_components(cfg, repr_method: str, device, ckpt_dir: str):
    logging.info(f"Loading {repr_method} components from {ckpt_dir}")
    D = cfg.data.state_dim

    if repr_method == "vae":
        vae_cfg = cfg.model.vae
        z_dim = vae_cfg.z_dim
        vae = VAE(
            D,
            z_dim,
            vae_cfg.enc_widths,
            vae_cfg.dec_widths,
            beta=cfg.train.vae.beta,
            activation=vae_cfg.activation,
        )
        vae.load_state_dict(
            torch.load(
                os.path.join(ckpt_dir, f"{repr_method}.pt"),
                map_location=device,
                weights_only=False,
            )["state_dict"]
        )
        vae.to(device).eval()
        encode_fn = vae.get_encoder_fn()

    elif repr_method == "contrastive":
        contrastive_cfg = cfg.model.contrastive
        z_dim = contrastive_cfg.z_dim
        contrastive_model = NachumConstrastive(
            D,
            z_dim,
            cfg.data.num_actions,
            contrastive_cfg.enc_widths,
            contrastive_cfg.proj_widths,
            temperature=cfg.train.contrastive.temperature,
            eps=contrastive_cfg.eps,
            use_layer_norm=contrastive_cfg.use_layer_norm,
            activation=contrastive_cfg.activation,
        )
        contrastive_model.load_state_dict(
            torch.load(
                os.path.join(ckpt_dir, f"{repr_method}.pt"),
                map_location=device,
                weights_only=False,
            )["state_dict"]
        )
        contrastive_model.to(device).eval()
        encode_fn = contrastive_model.get_encoder_fn()
    else:
        raise ValueError(
            f"Unknown repr_method: {repr_method}. Must be 'vae' or 'contrastive'"
        )

    dyn = DynamicsModel(
        z_dim,
        cfg.data.num_actions,
        cfg.model.dynamics.dyn_widths,
        z_space=repr_method,
        activation=cfg.model.dynamics.activation,
    )
    dyn.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, "dynamics.pt"),
            map_location=device,
            weights_only=False,
        )["state_dict"]
    )
    dyn.to(device).eval()

    probe = Probe(
        z_dim,
        cfg.data.signal_dim,
        z_space=repr_method,
        widths=cfg.model.probe.probe_widths,
        activation=cfg.model.probe.activation,
    )
    probe.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, "probe.pt"),
            map_location=device,
            weights_only=False,
        )["state_dict"]
    )
    probe.to(device).eval()

    return encode_fn, dyn, probe


@torch.inference_mode()
def evaluate(cfg, repr_method: str, device, ckpt_dir: str, split: str = "test"):
    dd = cfg.data.out_dir
    eval_batch_size = cfg.train.eval_batch_size
    num_workers = cfg.train.num_workers

    loader = DataLoader(
        TriplesNPZ(os.path.join(dd, f"{split}.npz")),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    encode, dyn, probe = load_components(cfg, repr_method, device, ckpt_dir)

    meter = AverageMeter()
    mse = torch.nn.MSELoss()

    for batch in loader:
        s, sp, a, a1h, sig, sig_next = _unpack_batch(
            batch, device, cfg.data.num_actions
        )
        batch_size = s.size(0)

        with torch.no_grad():
            z = encode(s)
            z_next = encode(sp)
            z_next_pred = dyn(z, a1h)

            signal_pred = probe(z)
            signal_true = s[:, : cfg.data.signal_dim]
            signal_next_pred = probe(z_next_pred)
            signal_next_true = sp[:, : cfg.data.signal_dim]

            mse_signal_next = mse(signal_next_pred, signal_next_true).item()
            mse_signal = mse(signal_pred, signal_true).item()
            mse_znext_znextpred = mse(z_next, z_next_pred).item()
            mse_z_znext = mse(z, z_next).item()

            z_expanded1 = z.unsqueeze(1)  # [B, 1, D]
            z_expanded2 = z.unsqueeze(0)  # [1, B, D]
            pairwise_diffs = z_expanded1 - z_expanded2  # [B, B, D]
            pairwise_mse = torch.mean(pairwise_diffs**2, dim=2)  # [B, B]
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            inter_batch_mse = torch.mean(pairwise_mse[mask]).item()

            metrics = {
                "mse_signal_reconstruction": mse_signal,
                "mse_next_signal_prediction": mse_signal_next,
                "mse_consecutive_embeddings": mse_z_znext,
                "mse_unrelated_embeddings": inter_batch_mse,
                "mse_dynamics_prediction": mse_znext_znextpred,
            }
            meter.update(metrics, {k: batch_size for k in metrics})

    final_metrics = meter.avg
    final_metrics.update(
        {k.replace("mse", "rmse"): v**0.5 for k, v in final_metrics.items()}
    )
    for metric_name, value in final_metrics.items():
        logging.info(f"[EVAL-{repr_method}] {metric_name} = {value:.4f}")
    wcfg = cfg.get("wandb", {})
    if wcfg.get("enabled", False):
        prefix = {"test": "eval", "train": "train", "val": "valid"}[
            split
        ]  # test -> eval for backwards compatibility
        wandb_metrics = {f"{prefix}/{k}": v for k, v in final_metrics.items()}
        wandb.log(wandb_metrics)

    return final_metrics


def main(cfg, repr_method, wb=None, ckpt_dir=None):
    logging.info(f"Starting evaluation for {repr_method} method")
    if ckpt_dir is None:
        ckpt_dir = cfg.train.ckpt_dir
    wcfg = cfg.wandb
    if wcfg.enabled and wb is None:
        logging.info("Initializing wandb for evaluation")
        wandb.init(
            project=wcfg.project,
            entity=wcfg.entity,
            group="eval" if wcfg.group is None else wcfg.group,
            mode=wcfg.mode,
            dir=wcfg.dir,
            tags=wcfg.tags + ["eval"],
            config=cfg,
            name="eval",
            resume="allow",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    seed_all(cfg.data.seed)
    for split in ["train", "val", "test"]:
        evaluate(
            cfg,
            repr_method,
            device,
            ckpt_dir,
            split=split,
        )
    logging.info("Evaluation completed")


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

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config_path = os.path.join(args.hydra_output_dir, ".hydra", "config.yaml")
    logging.info(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["wandb"]["enabled"] = False  # no wandb for standalone eval

    main(
        cfg=cfg,
        repr_method=args.zdim,
        wb=None,
        ckpt_dir=os.path.join(args.hydra_output_dir, "ckpts"),
    )
