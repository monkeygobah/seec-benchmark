from __future__ import annotations

LANDMARK_KEYS: tuple[str, ...] = (
    "iris_centroid",
    "iris_superior",
    "iris_inferior",
    "sclera_superior",
    "sclera_inferior",
    "medial_canthus",
    "lateral_canthus",
    "lid_superior",
    "lid_inferior",
)

RIGHT_KEYS: tuple[str, ...] = (
    "left_iris_centroid",
    "left_iris_superior",
    "left_iris_inferior",
    "left_sclera_superior",
    "left_sclera_inferior",
    "left_medial_canthus",
    "left_lateral_canthus",
    "left_lid_superior",
    "left_lid_inferior",
)

LEFT_KEYS: tuple[str, ...] = (
    "right_iris_centroid",
    "right_iris_superior",
    "right_iris_inferior",
    "right_sclera_superior",
    "right_sclera_inferior",
    "right_medial_canthus",
    "right_lateral_canthus",
    "right_lid_superior",
    "right_lid_inferior",
)

REPRESENTATION_BACKBONE = "backbone"
REPRESENTATION_PATCH_TOKENS = "patch_tokens"
POOL_GAP = "gap"
POOL_G2 = "g2"
POOL_G4 = "g4"
VALID_POOLING = (POOL_GAP, POOL_G2, POOL_G4)
VALID_SPLITS = ("train", "val", "test")
VALID_EXTERNAL_MODELS = ("dinov2_vitb14", "mae_vitb16_in1k_pretrain")
