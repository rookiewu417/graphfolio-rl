from pathlib import Path
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
import os

load_dotenv(Path(__file__).parents[2] / ".env")


def load_config(path: str | Path = "configs/default.yaml") -> DictConfig:
    path = Path(path)
    cfg = OmegaConf.load(path)
    # Handle simple 'defaults' list (mimics Hydra base-config merging)
    if "defaults" in cfg:
        base_cfgs = []
        for name in cfg.defaults:
            base_path = path.parent / f"{name}.yaml"
            base_cfgs.append(OmegaConf.load(base_path))
        base = OmegaConf.merge(*base_cfgs) if len(base_cfgs) > 1 else base_cfgs[0]
        cfg = OmegaConf.merge(base, OmegaConf.masked_copy(cfg, [k for k in cfg if k != "defaults"]))
    OmegaConf.resolve(cfg)
    return cfg


def get_device(cfg: DictConfig):
    import torch
    if cfg.project.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
