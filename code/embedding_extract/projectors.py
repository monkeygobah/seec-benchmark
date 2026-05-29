import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from typing import Iterable, Tuple, List, Dict, Any


def get_feat_out(y):
    return y["out"] if isinstance(y, dict) else y


class ProjectorCfg:
    def __init__(self, in_dim: int, proj_dim: int, hidden_dim: int, layers: int):
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.layers = layers

class MLPProjector(nn.Module):
    def __init__(self, cfg: ProjectorCfg):
        super().__init__()
        if cfg.layers not in (2, 3):
            raise ValueError("proj_layers must be 2 or 3")

        dims = [cfg.in_dim]
        if cfg.layers == 3:
            dims += [cfg.hidden_dim, cfg.hidden_dim, cfg.proj_dim]
        else:
            dims += [cfg.hidden_dim, cfg.proj_dim]

        blocks: List[nn.Module] = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < len(dims) - 2:
                blocks.append(nn.BatchNorm1d(dims[i + 1]))
                blocks.append(nn.ReLU(inplace=True))
            else:
                blocks.append(nn.BatchNorm1d(dims[i + 1]))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)