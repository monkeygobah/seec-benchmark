from __future__ import annotations

from pathlib import Path
import yaml
from src.run_utils import make_run_dir, save_config
from src.seed import seed_everything
import torch.distributed as dist
from datetime import datetime

def load_yaml(path):
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_config_bundle(args):
    cfg = load_yaml(args.cfg)
    return cfg


def init_run(cfg, is_main):
    seed_everything(cfg["run"]["seed"])

    runs_root = Path(cfg["run"]["runs_root"])
    run_name = cfg["run"]["name"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if is_main else None
    if dist.is_initialized():
        obj = [run_id]
        dist.broadcast_object_list(obj, src=0)
        run_id = obj[0]

    rp = make_run_dir(runs_root, run_name, run_id=run_id, mkdir=is_main)

    if is_main:
        save_config(cfg, rp.run_dir)

    if dist.is_initialized():
        dist.barrier()

    return rp
