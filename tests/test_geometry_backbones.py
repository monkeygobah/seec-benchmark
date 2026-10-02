from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from embedding_extract import inference
from src.backbones.vit import TimmViTPatchMap
from src.projectors import MLPProjector, ProjectorCfg


@pytest.mark.parametrize("backbone", ["resnet101", "vit_base_patch16_224", None])
def test_bundle_forwards_run_backbone_and_loads_states(monkeypatch, backbone):
    model = {"init": "random", "feat_dim": 4, "proj_dim": 2,
             "proj_hidden": 8, "proj_layers": 2}
    if backbone is not None:
        model["backbone"] = backbone
    cfg = {"model": model, "ssl": {"method": "vicreg"}}
    encoder = nn.Conv2d(3, 4, 1)
    projector = MLPProjector(ProjectorCfg(in_dim=4, proj_dim=2, hidden_dim=8, layers=2))
    calls = []
    def select(**kwargs):
        calls.append(kwargs)
        return encoder
    monkeypatch.setattr(inference, "load_encoder_backbone", select)
    checkpoint = {"encoder": encoder.state_dict(),
                  "objective": {"projector." + key: value for key, value in projector.state_dict().items()}}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    bundle = inference.build_inference_bundle(cfg, "unused.pth")
    assert calls == [{"init": "random", "seg_ckpt": None,
                      "backbone": backbone or "resnet101"}]
    assert bundle.encoder is encoder
    assert not encoder.training
    assert all(not p.requires_grad for p in bundle.encoder.parameters())
    result = inference.create_embedding_model(bundle, torch.device("cpu"))(torch.zeros(1, 3, 16, 16))
    assert result["emb"].shape == (1, 4)
    assert result["proj"].shape == (1, 2)


@pytest.mark.parametrize("backbone", ["resnet101", "vit_base_patch16_224"])
def test_actual_architecture_without_checkpoints_or_downloads(backbone):
    # Meta tensors instantiate the real architecture without allocating large weights.
    with torch.device("meta"):
        encoder = inference.build_geometry_encoder({"model": {"backbone": backbone, "init": "random"}})
    if backbone == "resnet101":
        assert isinstance(encoder, nn.Sequential)
        assert len(encoder[6]) == 23  # ResNet-101 layer3, unlike ResNet-50.
        assert encoder[7][-1].conv3.out_channels == 2048
    else:
        assert isinstance(encoder, TimmViTPatchMap)
        assert encoder.patch_size == 16
        assert encoder.model.embed_dim == 768
        assert len(encoder.model.blocks) == 12
        assert encoder.model.patch_embed.patch_size == (16, 16)


def test_unknown_backbone_is_rejected_before_checkpoint_load(monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Unsupported backbone must fail before loading weights")
    monkeypatch.setattr(torch, "load", unexpected_load)
    with pytest.raises(ValueError, match="Unknown encoder backbone"):
        inference.build_inference_bundle({"model": {"backbone": "not-a-backbone", "init": "random"}}, "unused.pth")


def test_loaded_run_config_overrides_select_backbone(tmp_path):
    import yaml
    from embedding_extract.pipeline_config import RunSpec
    from embedding_extract.runtime import load_training_config_for_run
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"model": {"backbone": "resnet101", "init": "random"}}),
        encoding="utf-8",
    )
    run = RunSpec(run_name="test", run_dir=tmp_path, checkpoint_step=0,
                  config_overrides={"model": {"backbone": "vit_base_patch16_224"}})
    loaded = load_training_config_for_run(run)
    with torch.device("meta"):
        encoder = inference.build_geometry_encoder(loaded)
    assert isinstance(encoder, TimmViTPatchMap)
    assert encoder.model.embed_dim == 768
