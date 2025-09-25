import logging
import os
import numpy as np
import hydra
from omegaconf import DictConfig

from src.data import DataManager


def main(cfg: DictConfig):
    """Generate trajectory data using DataManager (includes config saving)"""
    logging.info("🔄 Generating data using DataManager...")
    data_manager = DataManager(cfg)
    data_manager.force_regenerate()
    logging.info("✅ Data generation complete!")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main_wrapper(cfg: DictConfig) -> None:
    """Hydra entry point for standalone execution"""
    main(cfg)


if __name__ == "__main__":
    main_wrapper()
