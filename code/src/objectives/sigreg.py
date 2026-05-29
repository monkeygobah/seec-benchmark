from __future__ import annotations

import math

import torch
import torch.nn as nn


class SIGReg(nn.Module):

    def __init__(self, knots: int = 17, t_max: float = 3.0, num_slices: int = 256):
        super().__init__()
        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        window = torch.exp(-t.square() / 2.0)  # phi for N(0,1) at t
        self.register_buffer("t", t)                 # (knots,)
        self.register_buffer("phi", window)          # (knots,)
        self.register_buffer("weights", weights * window)  # (knots,)

        self.num_slices = int(num_slices)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # Canonicalize to (N, K)
        if proj.ndim == 3:
            # (bs,V,K) or (V,bs,K) -> flatten samples
            N = proj.shape[0] * proj.shape[1]
            K = proj.shape[2]
            x = proj.reshape(N, K)
        elif proj.ndim == 2:
            x = proj
            N, K = x.shape
        else:
            raise ValueError("proj must have shape (N,K) or (A,B,K)")

        device = x.device
        dtype = x.dtype

        # Sample random unit directions A: (K, S)
        A = torch.randn(K, self.num_slices, device=device, dtype=dtype)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-12)

        # (N, S)
        xA = x @ A

        # ECF terms over knots: compute for each slice independently, then average slices
        # x_t: (N, S, knots)
        x_t = xA.unsqueeze(-1) * self.t.to(device=device, dtype=dtype)

        # mean over N (samples): (S, knots)
        cos_m = x_t.cos().mean(dim=0)
        sin_m = x_t.sin().mean(dim=0)

        phi = self.phi.to(device=device, dtype=dtype)              # (knots,)
        weights = self.weights.to(device=device, dtype=dtype)      # (knots,)

        err = (cos_m - phi).square() + sin_m.square()              # (S, knots)
        statistic = (err @ weights) * float(N)                     # (S,)

        return statistic.mean()


class EPPartial(nn.Module):
    """
    Three-term sliced Epps-Pulley partial statistic.

    For each random unit direction, project x in R^K to a scalar X and compute

        (E[e^-X^2/2] - 1/sqrt(2))^2
        + E[X e^-X^2/2]^2
        + 1/2 * (E[X^2 e^-X^2/2] - 1/(2sqrt(2)))^2.

    Inputs follow SIGReg: either (N, K) or (A, B, K), with the latter flattened
    across the first two axes before slicing.
    """

    def __init__(self, num_slices: int = 256, scale_by_n: bool = False):
        super().__init__()
        self.num_slices = int(num_slices)
        self.scale_by_n = bool(scale_by_n)

        inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        self.register_buffer("d0", torch.tensor(inv_sqrt_2, dtype=torch.float32))
        self.register_buffer("d2", torch.tensor(0.5 * inv_sqrt_2, dtype=torch.float32))

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim == 3:
            N = proj.shape[0] * proj.shape[1]
            K = proj.shape[2]
            x = proj.reshape(N, K)
        elif proj.ndim == 2:
            x = proj
            N, K = x.shape
        else:
            raise ValueError("proj must have shape (N,K) or (A,B,K)")

        device = x.device
        dtype = x.dtype

        A = torch.randn(K, self.num_slices, device=device, dtype=dtype)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-12)

        xA = x @ A                                                  # (N, S)
        envelope = torch.exp(-0.5 * xA.square())                    # E

        c0 = envelope.mean(dim=0)                                   # (S,)
        c1 = (envelope * xA).mean(dim=0)                            # (S,)
        c2 = (envelope * xA.square()).mean(dim=0)                   # (S,)

        d0 = self.d0.to(device=device, dtype=dtype)
        d2 = self.d2.to(device=device, dtype=dtype)

        statistic = (c0 - d0).square() + c1.square() + 0.5 * (c2 - d2).square()
        if self.scale_by_n:
            statistic = statistic * float(N)

        return statistic.mean()


class BHEP(nn.Module):
    """
    Direct multivariate Baringhaus-Henze-Epps-Pulley statistic.

    This computes the closed-form Gaussian-kernel discrepancy between empirical
    samples x in R^K and a standard normal target:

        mean_ij exp(-||x_i - x_j||^2 / (2 beta^2))
        - 2 (1 + beta^2)^(-K/2) mean_i exp(-||x_i||^2 / (2(1 + beta^2)))
        + (1 + 2 beta^2)^(-K/2).

    Inputs follow SIGReg: either (N, K) or (A, B, K), with the latter flattened
    across the first two axes.
    """

    def __init__(self, beta: float = 1.0, scale_by_n: bool = False):
        super().__init__()
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        self.beta = float(beta)
        self.scale_by_n = bool(scale_by_n)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim == 3:
            N = proj.shape[0] * proj.shape[1]
            K = proj.shape[2]
            x = proj.reshape(N, K)
        elif proj.ndim == 2:
            x = proj
            N, K = x.shape
        else:
            raise ValueError("proj must have shape (N,K) or (A,B,K)")

        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        beta2 = self.beta * self.beta

        squared_norm = x.square().sum(dim=1, keepdim=True)          # (N, 1)
        dist2 = squared_norm - 2.0 * (x @ x.T) + squared_norm.T     # (N, N)
        dist2 = dist2.clamp_min(0.0)

        term1 = torch.exp(-dist2 / (2.0 * beta2)).mean()

        cross_scale = x.new_tensor(math.exp(-0.5 * K * math.log1p(beta2)))
        cross_kernel = torch.exp(
            -squared_norm.squeeze(1) / (2.0 * (1.0 + beta2))
        ).mean()
        term2 = 2.0 * cross_scale * cross_kernel

        target_term = x.new_tensor(math.exp(-0.5 * K * math.log1p(2.0 * beta2)))

        statistic = term1 - term2 + target_term
        if self.scale_by_n:
            statistic = statistic * float(N)

        return statistic
