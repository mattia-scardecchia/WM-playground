import os
import argparse
import yaml
import numpy as np

from data import NoisyVectorWorld, sample_trajectories


def main(cfg):
    """Generate trajectory data using configuration"""
    out_dir = cfg["data"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    env = NoisyVectorWorld(
        signal_dim=cfg["data"]["signal_dim"],
        noise_dim=cfg["data"]["noise_dim"],
        num_actions=cfg["data"]["num_actions"],
        step=cfg["data"]["step_size"],
        seed=cfg["data"]["seed"],
    )

    def write_split(name: str, n_traj: int):
        """Write a data split (train/val/test) to disk"""
        triples = sample_trajectories(
            env, n_traj=n_traj, traj_len=cfg["data"]["traj_len"], policy="random"
        )
        path = os.path.join(out_dir, f"{name}.npz")
        np.savez(path, **triples)
        print(
            f"Wrote {path} with shapes s{triples['s'].shape}, a{triples['a'].shape}, sp{triples['sp'].shape}"
        )

    write_split("train", cfg["data"]["n_train"])
    write_split("val", cfg["data"]["n_val"])
    write_split("test", cfg["data"]["n_test"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate trajectory data")
    ap.add_argument(
        "--config", type=str, default="config.yaml", help="Config file path"
    )
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
