from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Any
from projectors import MLPProjector,ProjectorCfg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from embedding_extract._old.objectives.embed_lejepa import LeJEPABackbonePlusProjector
from embedding_extract._old.load_backbones import *
from typing import Optional

def _as_state_dict(x: Any) -> Optional[Dict[str, torch.Tensor]]:
    return x if isinstance(x, dict) and any(torch.is_tensor(v) for v in x.values()) else None



def _strip_module(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def load_model_from_ckpt(
    ckpt_path: str | Path,
    init: str,
    seg_ckpt: Optional[str],
    proj_hidden: int,
    proj_layers: int,
    device: torch.device,
    proj_dim: int = 128,
) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    encoder = load_encoder_backbone(init=init, seg_ckpt=seg_ckpt)
    encoder.load_state_dict(_strip_module(ckpt["encoder"]), strict=True)

    projector = MLPProjector(
        ProjectorCfg(
            in_dim=2048,
            proj_dim=proj_dim,
            hidden_dim=proj_hidden,
            layers=proj_layers,
        )
    )

    obj_sd = ckpt["objective"]
    if not isinstance(obj_sd, dict):
        raise TypeError(f"ckpt['objective'] is not a dict; got {type(obj_sd)}")

    proj_sd = {k[len("projector."):]: v for k, v in obj_sd.items() if k.startswith("projector.")}
    if not proj_sd:
        proj_sd = {k[len("proj."):]: v for k, v in obj_sd.items() if k.startswith("proj.")}

    if not proj_sd:
        raise KeyError(
            "No projector weights found in ckpt['objective']. "
            "Expected keys like 'projector....'. "
            f"First keys: {list(obj_sd.keys())[:20]}"
        )

    projector.load_state_dict(_strip_module(proj_sd), strict=True)

    model = LeJEPABackbonePlusProjector(encoder, projector).to(device)
    return model



from pathlib import Path
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TEMP test values — pick ONE InfoNCE checkpoint
ckpt_path = Path(r"C:\local_storage\eccv\embedding_probe\models\infonce-2V-random.pth")

init = "random"          # or "imagenet", "seg_init"
seg_ckpt = None          # only needed for seg_init
PROJ_HIDDEN = 2048
PROJ_LAYERS = 3
ckpt = torch.load(ckpt_path, map_location="cpu")
print("Top-level keys:", ckpt.keys())
print("Objective key sample:", list(ckpt["objective"].keys())[:30])


model = load_model_from_ckpt(
    ckpt_path=str(ckpt_path),
    init=init,
    seg_ckpt=seg_ckpt,
    proj_hidden=PROJ_HIDDEN,
    proj_layers=PROJ_LAYERS,
    device=device,
    proj_dim=128,
)


x = torch.randn(2, 3, 128, 128, device=device)
with torch.no_grad():
    z = model(x)

print("Output shape:", z.shape)
print("Mean norm:", z.norm(dim=1).mean().item())
