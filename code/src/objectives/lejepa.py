from collections.abc import Mapping

import torch
import torch.nn as nn

from src.projectors import MLPProjector, ProjectorCfg, gap_pool
from .sigreg import BHEP, EPPartial, SIGReg


def lejepa_sim_loss(proj_bvk):
    center = proj_bvk.mean(dim=1, keepdim=True)
    return (center - proj_bvk).square().mean()


def get_feat_out(y):
    return y["out"] if isinstance(y, Mapping) else y


class LeJEPAObjective(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        proj_cfg = ProjectorCfg(
            in_dim=int(cfg["model"].get("feat_dim", 2048)),
            proj_dim=int(cfg["model"]["proj_dim"]),
            hidden_dim=int(cfg["model"]["proj_hidden"]),
            layers=int(cfg["model"]["proj_layers"]),
        )

        self.projector = MLPProjector(proj_cfg)

        loss_cfg = cfg["loss"]
        self.regularizer_name = str(loss_cfg.get("regularizer", "sigreg")).lower()
        if self.regularizer_name == "sigreg":
            self.regularizer = SIGReg(
                knots=int(loss_cfg["sigreg_knots"]),
                num_slices=int(loss_cfg["sigreg_num_slices"]),
            )
        elif self.regularizer_name == "ep_partial":
            self.regularizer = EPPartial(
                num_slices=int(
                    loss_cfg.get(
                        "ep_partial_num_slices",
                        loss_cfg.get("sigreg_num_slices", 256),
                    )
                ),
                scale_by_n=bool(loss_cfg.get("ep_partial_scale_by_n", False)),
            )
        elif self.regularizer_name == "bhep":
            self.regularizer = BHEP(
                beta=float(loss_cfg.get("bhep_beta", 1.0)),
                scale_by_n=bool(loss_cfg.get("bhep_scale_by_n", False)),
            )
        else:
            raise ValueError(f"Unknown LeJEPA regularizer: {self.regularizer_name}")

        self.regularizer_log_key = (
            "sigreg" if self.regularizer_name == "sigreg" else self.regularizer_name
        )
        self.lamb = float(cfg["loss"]["lamb"])

    def forward(self, encoder, vs):
        if isinstance(vs, (list, tuple)):
            return self._forward_multicrop(encoder, vs)
        else:
            return self._forward_local(encoder, vs)

    def _forward_local(self, encoder, vs):
        # vs: (bs, V, C, H, W)
        bs, V, C, H, W = vs.shape
        x = vs.view(bs * V, C, H, W)

        feat = get_feat_out(encoder(x))
        emb = gap_pool(feat)
        proj = self.projector(emb)
        K = proj.shape[1]
        proj_bvk = proj.view(bs, V, K)

        sim = lejepa_sim_loss(proj_bvk)
        reg = self.regularizer(proj_bvk)
        loss = (1.0 - self.lamb) * sim + self.lamb * reg

        return loss, {
            "loss": loss,
            "sim": sim,
            self.regularizer_log_key: reg,
            "reg": reg,
            "V": V,
        }

    def _forward_multicrop(self, encoder, vs):
        # vs: List[Tensor(bs, C, H, W)], length Vg + Vl (mixed resolutions)
        V = len(vs)
        bs = vs[0].shape[0]

        # Group views by spatial resolution and batch them together
        # to avoid redundant forward passes at the same resolution
        groups: dict = {}
        for j, v in enumerate(vs):
            key = tuple(v.shape[-2:])
            groups.setdefault(key, []).append(j)

        emb_per_view: list = [None] * V

        for (H, W), idxs in groups.items():
            # Stack all views of this resolution into one batch
            x = torch.cat([vs[j] for j in idxs], dim=0)  # (bs * len(idxs), C, H, W)
            feat = get_feat_out(encoder(x))
            emb = gap_pool(feat)  # (bs * len(idxs), 2048)
            # Split back out per-view
            chunks = emb.chunk(len(idxs), dim=0)  # len(idxs) x (bs, 2048)
            for t, j in enumerate(idxs):
                emb_per_view[j] = chunks[t]

        emb_bvk = torch.stack(emb_per_view, dim=1)  # (bs, V, 2048)
        proj_bvk = self.projector(
            emb_bvk.view(bs * V, -1)
        ).view(bs, V, -1)  # (bs, V, K)

        sim = lejepa_sim_loss(proj_bvk)
        reg = self.regularizer(proj_bvk)
        loss = (1.0 - self.lamb) * sim + self.lamb * reg

        return loss, {
            "loss": loss,
            "sim": sim,
            self.regularizer_log_key: reg,
            "reg": reg,
            "V": V,
        }
