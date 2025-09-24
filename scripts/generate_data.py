import os
import numpy as np
import hydra
from omegaconf import DictConfig

from data import NoisyVectorWorld, sample_trajectories
from src.data import DataManager
from config_schemas import register_configs


def main(cfg: DictConfig):
    """Generate trajectory data using DataManager (includes config saving)"""
    print("🔄 Generating data using DataManager...")
    data_manager = DataManager(cfg)
    data_manager.force_regenerate()
    print("✅ Data generation complete!")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main_wrapper(cfg: DictConfig) -> None:
    """Hydra entry point for standalone execution"""
    register_configs()
    main(cfg)


if __name__ == "__main__":
    main_wrapper()
