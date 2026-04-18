from pathlib import Path
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
import os

load_dotenv(Path(__file__).parents[2] / ".env")


def load_config(path: str | Path = "configs/default.yaml") -> DictConfig:
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return cfg


def get_device(cfg: DictConfig):
    import torch
    if cfg.project.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
