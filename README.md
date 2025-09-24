# Representation Learning in Grid World

Note: AI-generated readme (for now).

A PyTorch implementation for learning representations in a noisy vector environment using VAE and contrastive learning approaches.

## Overview

This project compares representation learning methods on a simple 2D grid world with high-dimensional noisy observations:

- **VAE**: Variational autoencoder with β-VAE regularization
- **Contrastive**: Nachum et al. style contrastive learning
- **Dynamics**: Forward dynamics prediction models
- **Probing**: Linear probes to evaluate learned representations

## Quick Start

### Installation

```bash
pip install -e .
```

### Generate Data

```bash
python scripts/generate_data.py --config config.yaml
```

### Train Models

```bash
python scripts/train.py --config config.yaml --mode vae
python scripts/train.py --config config.yaml --mode contrastive
```

### Evaluate

```bash
python scripts/eval.py --config config.yaml
```

## Configuration

Key parameters in `config.yaml`:

- `data.signal_dim`: Dimensionality of true state (default: 2)
- `data.noise_dim`: Dimensionality of noise (default: 100)
- `model.z_dim_*`: Latent representation dimensions
- `train.*`: Training hyperparameters

## Notebooks

- `notebooks/vae.ipynb`: VAE experiments and analysis
- `notebooks/contrastive.ipynb`: Contrastive learning experiments

## Structure

- `src/`: Core implementation (models, data, training)
- `scripts/`: Training and evaluation scripts
- `notebooks/`: Jupyter notebooks for analysis
- `data/`: Generated trajectory data
