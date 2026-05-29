from __future__ import annotations

import numpy as np
import torch


def mean_l2_per_landmark(yhat: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    diffs = (yhat - y).view(y.shape[0], k, 2)
    return torch.sqrt((diffs**2).sum(dim=-1) + 1e-9).mean()


def per_landmark_stats(yhat: torch.Tensor, y: torch.Tensor, landmark_keys: tuple[str, ...]) -> list[dict[str, float | str]]:
    k = len(landmark_keys)
    diffs = (yhat - y).view(y.shape[0], k, 2)
    distances = torch.sqrt((diffs**2).sum(dim=-1) + 1e-9).cpu().numpy()
    rows = []
    for idx, key in enumerate(landmark_keys):
        values = distances[:, idx]
        rows.append(
            {
                "landmark": key,
                "mean_l2": float(values.mean()),
                "median_l2": float(np.median(values)),
                "std_l2": float(values.std(ddof=0)),
                "p90_l2": float(np.percentile(values, 90)),
            }
        )
    return rows


def per_sample_mean_l2(yhat: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    diffs = (yhat - y).view(y.shape[0], k, 2)
    return torch.sqrt((diffs**2).sum(dim=-1) + 1e-9).mean(dim=1)
