from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import init_run, load_config_bundle
from src.dataset_utils import build_dataset, is_iterable_dataset
from src.load_backbones import load_encoder_backbone
from src.objectives.byol import BYOLObjective
from src.objectives.lejepa import LeJEPAObjective
from src.objectives.infonce import CrossViewInfoNCEObjective
from src.objectives.vicreg import VICRegObjective
from src.run_utils import load_checkpoint, save_checkpoint
from src.transforms import (
    LocalViewsCfg,
    MultiCropCfg,
    ViewAugCfg,
    build_local_views_transform,
    build_multicrop_transform,
    collate_multicrop_with_meta,
    collate_views_with_meta,
)


def should_use_ddp(cfg: dict[str, Any]) -> bool:
    runtime_cfg = cfg.get("runtime", {})
    mode = str(runtime_cfg.get("distributed", "auto")).lower()
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_runtime(cfg: dict[str, Any]):
    ddp_enabled = should_use_ddp(cfg)
    runtime_cfg = cfg.get("runtime", {})
    backend = str(runtime_cfg.get("ddp_backend", "nccl"))
    gpu = int(runtime_cfg.get("gpu", 0))

    if ddp_enabled:
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = gpu
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

    is_main = rank == 0
    return {
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "is_main": is_main,
        "ddp_enabled": ddp_enabled,
    }


def disable_running_stats(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def build_ssl_transform(cfg: dict[str, Any]):
    aug_mode = cfg["ssl"].get("aug_mode", "local_only")

    if aug_mode == "local_only":
        tcfg = LocalViewsCfg(
            V=int(cfg["ssl"]["V"]),
            crop_size=int(cfg["ssl"]["crop_size"]),
            scale_min=float(cfg["ssl"]["crop_scale_min"]),
            scale_max=float(cfg["ssl"]["crop_scale_max"]),
            normalize_imagenet=bool(cfg["ssl"]["normalize_imagenet"]),
        )
        return build_local_views_transform(tcfg), collate_views_with_meta

    if aug_mode == "multicrop":
        gcfg = ViewAugCfg(
            V=int(cfg["ssl"]["global_V"]),
            crop_size=int(cfg["ssl"]["global_crop"]),
            scale_min=float(cfg["ssl"]["global_scale_min"]),
            scale_max=float(cfg["ssl"]["global_scale_max"]),
            normalize_imagenet=bool(cfg["ssl"]["normalize_imagenet"]),
        )
        lcfg = ViewAugCfg(
            V=int(cfg["ssl"]["local_V"]),
            crop_size=int(cfg["ssl"]["local_crop"]),
            scale_min=float(cfg["ssl"]["local_scale_min"]),
            scale_max=float(cfg["ssl"]["local_scale_max"]),
            normalize_imagenet=bool(cfg["ssl"]["normalize_imagenet"]),
        )
        mcfg = MultiCropCfg(global_=gcfg, local=lcfg)
        return build_multicrop_transform(mcfg), collate_multicrop_with_meta

    raise ValueError(f"Unknown ssl.aug_mode: {aug_mode}")


def build_objective(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    method = cfg["ssl"]["method"]
    if method == "lejepa":
        return LeJEPAObjective(cfg).to(device)
    if method == "vicreg":
        return VICRegObjective(cfg).to(device)
    if method == "infonce":
        return CrossViewInfoNCEObjective(cfg).to(device)
    if method == "byol":
        return BYOLObjective(cfg).to(device)
    raise ValueError(f"Unknown ssl.method: {method}")


def main(args):
    cfg = load_config_bundle(args)
    runtime = init_runtime(cfg)
    device = runtime["device"]
    local_rank = runtime["local_rank"]
    is_main = runtime["is_main"]
    ddp_enabled = runtime["ddp_enabled"]
    run_cfg = cfg["run"]

    try:
        rp = init_run(cfg, is_main=is_main)


        transform, collate_fn = build_ssl_transform(cfg)
        ds = build_dataset(cfg=cfg, transform=transform, root_key="train_root")

        ds_is_iterable = is_iterable_dataset(ds)
        sampler = DistributedSampler(ds, shuffle=True) if ddp_enabled and not ds_is_iterable else None
        shuffle = bool(cfg["dataloader"].get("shuffle", True)) if sampler is None and not ds_is_iterable else False
        num_workers = int(cfg["dataloader"]["num_workers"])
        persistent_workers = bool(cfg["dataloader"].get("persistent_workers", False)) and num_workers > 0
        dl = DataLoader(
            ds,
            batch_size=int(cfg["dataloader"]["batch_size"]),
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=bool(cfg["dataloader"]["pin_memory"]),
            drop_last=bool(cfg["dataloader"].get("drop_last", False)),
            collate_fn=collate_fn,
        )

        encoder = load_encoder_backbone(
            backbone=cfg["model"].get("backbone", "resnet101"),
            init=cfg["model"]["init"],
            seg_ckpt=cfg["model"].get("seg_ckpt"),
        ).to(device)
        objective = build_objective(cfg, device)

        disable_running_stats(encoder)

        if ddp_enabled:
            objective = DDP(objective, device_ids=[local_rank], output_device=local_rank)
            encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)

        encoder_core = unwrap_model(encoder)
        objective_core = unwrap_model(objective)

        if cfg["ssl"]["method"] == "byol":
            objective_core.init_target_encoder(encoder_core)

        trainable = list(encoder.parameters())
        if cfg["ssl"]["method"] == "byol":
            trainable += list(objective_core.projector.parameters())
            trainable += list(objective_core.predictor.parameters())
        else:
            trainable += list(objective.parameters())

        opt = torch.optim.AdamW(
            trainable,
            lr=float(cfg["optim"]["lr"]),
            weight_decay=float(cfg["optim"]["weight_decay"]),
        )

        total_steps = int(cfg["run"]["total_steps"])
        warmup_steps = int(cfg["run"]["warmup_steps"])
        s1 = LinearLR(
            opt,
            start_factor=float(cfg["optim"]["warmup_factor"]),
            total_iters=warmup_steps,
        )
        s2 = CosineAnnealingLR(
            opt,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=float(cfg["sched"]["final_lr"]),
        )
        scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

        amp_enabled = bool(cfg["amp"]["enabled"]) and device.type == "cuda"
        amp_dtype = cfg["amp"]["dtype"].lower()
        autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        scaler = GradScaler(enabled=amp_enabled and autocast_dtype == torch.float16)

        log_every = int(cfg["run"]["log_every"])
        ckpt_every = int(cfg["run"]["ckpt_every"])
        metrics_path = rp.run_dir / "train_metrics.jsonl"

        step = 0
        epoch = 0

        resume = bool(run_cfg.get("resume", False))
        resume_ckpt = run_cfg.get("resume_ckpt")
        if resume:
            if not resume_ckpt:
                raise ValueError("run.resume=true requires run.resume_ckpt")

            step, epoch = load_checkpoint(
                resume_ckpt,
                encoder=encoder,
                objective=objective,
                opt=opt,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
            )
            if is_main:
                print(f"Resumed from {resume_ckpt} at step={step}, epoch={epoch}")

        while step < total_steps:
            encoder.train()
            objective.train()
            if sampler is not None:
                sampler.set_epoch(epoch)

            try:
                epoch_total = len(dl)
            except TypeError:
                epoch_total = None
            iterator = tqdm(dl, total=epoch_total, desc=f"epoch {epoch}") if is_main else dl

            for vs, _ in iterator:
                if step >= total_steps:
                    break

                if cfg["ssl"].get("aug_mode", "local_only") == "multicrop":
                    vs = [v.to(device, non_blocking=True) for v in vs]
                else:
                    vs = vs.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                with autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                    loss, logs = objective(encoder, vs)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                scheduler.step()

                if cfg["ssl"]["method"] == "byol":
                    objective_core.update_target(
                        encoder_core,
                        step=step + 1,
                        total_steps=total_steps,
                    )

                if is_main and step % log_every == 0:
                    rec = {
                        "step": step,
                        "epoch": epoch,
                        "lr": float(opt.param_groups[0]["lr"]),
                        "world_size": int(runtime["world_size"]),
                        "bs": int(vs[0].shape[0]) if isinstance(vs, (list, tuple)) else int(vs.shape[0]),
                    }
                    for k, v in logs.items():
                        rec[k] = float(v.detach().item()) if torch.is_tensor(v) else v
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")

                completed_step = step + 1

                if is_main and ckpt_every > 0 and completed_step % ckpt_every == 0:
                    save_checkpoint(
                        ckpt_dir=rp.ckpt_dir,
                        step=completed_step,
                        encoder=encoder,
                        objective=objective,
                        opt=opt,
                        epoch=epoch,
                        scheduler=scheduler,
                        scaler=scaler,
                    )

                step += 1

            epoch += 1

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)

    main(ap.parse_args())
