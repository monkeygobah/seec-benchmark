from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, Tuple, List

import torch


# ----------------------------
# Core math utilities
# ----------------------------

def _load_embeddings(pt_path: Path) -> torch.Tensor:
    """Load a [N,D] tensor from a .pt file (tensor or dict payload)."""
    obj = torch.load(pt_path, map_location="cpu")

    if torch.is_tensor(obj):
        x = obj
    elif isinstance(obj, dict):
        for k in ["proj", "z", "emb", "embeddings", "X", "features"]:
            if k in obj and torch.is_tensor(obj[k]):
                x = obj[k]
                break
        else:
            raise ValueError(f"No tensor found in dict keys: {list(obj.keys())}")
    else:
        raise TypeError(f"Unsupported .pt payload type: {type(obj)}")

    if x.ndim != 2:
        raise ValueError(f"Expected [N,D], got {tuple(x.shape)}")

    return x


def _center(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return centered X and mean vector mu."""
    mu = x.mean(dim=0, keepdim=True)
    return x - mu, mu.squeeze(0)


def _l2_normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 normalization."""
    n = x.norm(dim=1, keepdim=True).clamp_min(eps)
    return x / n


def _cov_eigs(x_centered: torch.Tensor) -> torch.Tensor:
    """
    Eigenvalues of covariance via singular values:
      if X = U S V^T then cov = (S^2)/(n-1) in the V-basis.
    Returns descending eigenvalues of length D (pads zeros if N < D).
    """
    n, d = x_centered.shape
    xc = x_centered.to(torch.float64)
    s = torch.linalg.svdvals(xc)                 # length = min(n,d)
    eigs = (s * s) / max(n - 1, 1)              # sample covariance scaling

    if eigs.numel() < d:
        eigs = torch.cat([eigs, torch.zeros(d - eigs.numel(), dtype=eigs.dtype)])

    eigs, _ = torch.sort(eigs, descending=True)
    return eigs


def _explained_variance(eigs: torch.Tensor, k: int) -> float:
    """Sum of top-k eigenvalues divided by total variance (trace)."""
    tot = eigs.sum().clamp_min(1e-30)
    return float((eigs[:k].sum() / tot).item())


def _effective_rank_from_eigs(eigs: torch.Tensor) -> float:
    """Participation ratio: (sum λ)^2 / sum(λ^2)."""
    s1 = eigs.sum()
    s2 = (eigs * eigs).sum().clamp_min(1e-30)
    return float((s1 * s1 / s2).item())


@torch.no_grad()
def _sample_pairwise_cosines(x_unit: torch.Tensor, num_pairs: int, seed: int = 0) -> torch.Tensor:
    """Monte Carlo estimate of cosine distribution from random pairs."""
    n = x_unit.shape[0]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    i = torch.randint(0, n, (num_pairs,), generator=g)
    j = torch.randint(0, n, (num_pairs,), generator=g)

    mask = i == j
    if mask.any():
        j[mask] = (j[mask] + 1) % n

    cos = (x_unit[i] * x_unit[j]).sum(dim=1)
    return cos.to(torch.float32)


def compute_isotropy_metrics(x: torch.Tensor, num_pairs: int = 200_000, seed: int = 0) -> Dict[str, Any]:
    """
    Covariance-spectrum isotropy + angular isotropy (pairwise cosines),
    computed on centered embeddings.
    """
    n, d = x.shape
    out: Dict[str, Any] = {"N": int(n), "D": int(d)}

    xc, mu = _center(x)
    out["mean_norm"] = float(mu.norm().item())

    eigs = _cov_eigs(xc)
    out["ev1"] = _explained_variance(eigs, 1)
    out["ev5"] = _explained_variance(eigs, min(5, d))
    out["ev10"] = _explained_variance(eigs, min(10, d))
    out["ev20"] = _explained_variance(eigs, min(20, d))

    lam1 = eigs[0].clamp_min(1e-30)
    lam_med = eigs[d // 2].clamp_min(1e-30)
    out["cond_1_med"] = float((lam1 / lam_med).item())

    er = _effective_rank_from_eigs(eigs)
    out["erank"] = er
    out["erank_over_d"] = er / float(d)

    # Angular stats on centered + row-normalized embeddings
    x_unit = _l2_normalize_rows(xc.to(torch.float32))

    # Avoid absurd oversampling for small N
    pairs_used = min(int(num_pairs), max(10_000, n * 50))
    cos = _sample_pairwise_cosines(x_unit.cpu(), num_pairs=pairs_used, seed=seed)

    out["num_pairs_used"] = int(pairs_used)
    out["cos_mean"] = float(cos.mean().item())
    out["cos_std"] = float(cos.std(unbiased=True).item())
    for thr in [0.2, 0.3, 0.4]:
        out[f"cos_frac_abs_gt_{thr}"] = float((cos.abs() > thr).float().mean().item())

    out["cos_std_expected_sphere"] = float(1.0 / (d ** 0.5))
    return out


# ----------------------------
# Project-specific runner
# ----------------------------

# def main() -> None:
#     # EDIT THESE PATHS
#     outputs_root = Path(r"C:\local_storage\eccv\embedding_probe\outputs\lejepa-multiview")
#     out_dir = Path(r"C:\local_storage\eccv\embedding_probe\isotropy_out_cfc\lejepa-multiview")

#     # Settings
#     dataset_name = "cfc"        # start with training data
#     num_pairs = 200_000
#     seed = 0

#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Collect files like: outputs/<run_name>/cfc_proj.pt
#     pt_files = sorted(outputs_root.glob(f"*/{dataset_name}_proj.pt"))
#     if not pt_files:
#         raise FileNotFoundError(f"No files found at: {outputs_root}\\*\\{dataset_name}_proj.pt")

#     rows: List[Dict[str, Any]] = []
#     for pt_path in pt_files:
#         run_name = pt_path.parent.name  # e.g., "lejepa-2V-imagenet"

#         x = _load_embeddings(pt_path)
#         m = compute_isotropy_metrics(x, num_pairs=num_pairs, seed=seed)

#         m["run"] = run_name
#         m["dataset"] = dataset_name
#         m["pt"] = str(pt_path)

#         # Per-run JSON
#         out_json = out_dir / f"{run_name}_{dataset_name}_isotropy.json"
#         with open(out_json, "w", encoding="utf-8") as f:
#             json.dump(m, f, indent=2)

#         rows.append(m)

#     # Summary CSV (easy to sort in Excel)
#     summary_csv = out_dir / f"isotropy_summary_{dataset_name}.csv"
#     fieldnames = [
#         "run", "dataset", "pt",
#         "N", "D",
#         "mean_norm",
#         "erank", "erank_over_d",
#         "ev1", "ev5", "ev10", "ev20",
#         "cond_1_med",
#         "cos_mean", "cos_std", "cos_std_expected_sphere",
#         "cos_frac_abs_gt_0.2", "cos_frac_abs_gt_0.3", "cos_frac_abs_gt_0.4",
#         "num_pairs_used",
#     ]
#     with open(summary_csv, "w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=fieldnames)
#         w.writeheader()
#         for r in rows:
#             w.writerow({k: r.get(k, None) for k in fieldnames})

#     # Console: quick ranked view (most isotropic by erank_over_d)
#     rows_sorted = sorted(rows, key=lambda r: r["erank_over_d"], reverse=True)
#     print(f"\n[SUMMARY] {dataset_name} (sorted by erank_over_d desc)\n")
#     for r in rows_sorted:
#         print(
#             f"{r['run']:>20s} | "
#             f"erank/D={r['erank_over_d']:.3f} | "
#             f"ev1={r['ev1']:.3f} | "
#             f"cond_1_med={r['cond_1_med']:.2f} | "
#             f"cos_std={r['cos_std']:.4f} (sphere~{r['cos_std_expected_sphere']:.4f})"
#         )

#     print(f"\nWrote: {summary_csv}")


# if __name__ == "__main__":
#     main()
