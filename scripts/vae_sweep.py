import os
import copy
import json
import itertools
import numpy as np
import torch
import yaml
from pathlib import Path
from datetime import datetime

from generate_data import main as gen_main
from train import train_phase1_vae, train_phase2_dynamics, train_probes
from scripts.eval import evaluate


def run_single_experiment(base_cfg, beta, z_dim_vae, step_size, device, exp_id):
    """Run a single experiment with given hyperparameters"""
    print(
        f"\n=== Experiment {exp_id}: beta={beta}, z_dim_vae={z_dim_vae}, step_size={step_size} ==="
    )

    # Create modified config
    cfg = copy.deepcopy(base_cfg)
    cfg["train"]["vae"]["beta"] = beta
    cfg["model"]["z_dim_vae"] = z_dim_vae
    cfg["data"]["step_size"] = step_size

    # Create unique checkpoint directory for this experiment
    cfg["train"]["ckpt_dir"] = f"ckpts_exp_{exp_id}/"
    cfg["data"]["out_dir"] = f"data_exp_{exp_id}/"

    # Generate data with new step_size
    print("Generating data...")
    gen_main(cfg)

    # Train VAE (modified function returns metrics)
    print("Training VAE...")
    _, vae_metrics = train_phase1_vae(cfg, device, wb=None)

    # Train dynamics (modified function returns metrics)
    print("Training dynamics...")
    _, dynamics_metrics = train_phase2_dynamics(cfg, device, z_space="vae", wb=None)

    # Train probe (modified function returns metrics)
    print("Training probe...")
    _, probe_metrics = train_probes(cfg, device, z_space="vae", wb=None)

    # Evaluate (modified function returns metrics)
    print("Evaluating...")
    eval_metrics = evaluate(cfg, "vae", device)

    # Combine all metrics
    all_metrics = {
        "beta": beta,
        "z_dim_vae": z_dim_vae,
        "step_size": step_size,
        "exp_id": exp_id,
        **vae_metrics,
        **dynamics_metrics,
        **probe_metrics,
        **eval_metrics,
    }

    print(f"Metrics: {all_metrics}")
    return all_metrics


def main():
    # Load base configuration
    with open("config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    # Disable wandb for sweep
    base_cfg["wandb"]["enabled"] = False

    # Define hyperparameter grid
    betas = [0.001, 0.005, 0.01]
    z_dims = [2, 30, 102]
    step_sizes = [0.1, 1.0]

    # Setup device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Run all combinations
    all_results = []
    exp_id = 0

    for beta, z_dim_vae, step_size in itertools.product(betas, z_dims, step_sizes):
        try:
            result = run_single_experiment(
                base_cfg, beta, z_dim_vae, step_size, device, exp_id
            )
            all_results.append(result)
            exp_id += 1
        except Exception as e:
            print(f"Error in experiment {exp_id}: {e}")
            exp_id += 1
            continue

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"hyperparameter_sweep_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== SWEEP COMPLETE ===")
    print(f"Results saved to: {results_file}")
    print(f"Total experiments: {len(all_results)}")

    # Print summary
    print("\n=== SUMMARY ===")
    for result in all_results:
        print(
            f"Exp {result['exp_id']}: beta={result['beta']}, z_dim={result['z_dim_vae']}, step_size={result['step_size']}"
        )
        print(
            f"  VAE: recon={result['final_vae_recon']:.4f}, kl={result['final_vae_kl']:.4f}"
        )
        print(f"  Dynamics: {result['final_dynamics_loss']:.4f}")
        print(f"  Probe: {result['final_probe_loss']:.4f}")
        print(f"  Eval: {result['final_eval_loss']:.4f}")


if __name__ == "__main__":
    main()
