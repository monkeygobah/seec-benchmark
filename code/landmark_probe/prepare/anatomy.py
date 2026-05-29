from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from landmark_probe.constants import LANDMARK_KEYS, LEFT_KEYS, RIGHT_KEYS
from segmentation.build_dataset.periorbital_tools.find_anatomy_from_masks import EyeFeatureExtractor


@dataclass(frozen=True)
class EyeCropSample:
    sample_id: str
    source_id: str
    dataset_name: str
    anatomical_side: str
    image_name: str
    image_rel_path: str
    image: Image.Image
    landmarks: dict[str, float]


@dataclass(frozen=True)
class EyeCropFailure:
    source_id: str
    sample_id: str
    dataset_name: str
    anatomical_side: str
    image_path: str
    mask_path: str
    reason: str


def extract_and_split_masks(combined_prediction: np.ndarray) -> dict[str, np.ndarray]:
    categories = {
        "sclera_orig": 2,
        "iris": 3,
        "brow": 1,
        "caruncle": 4,
        "lid": 5,
    }
    midline_x = combined_prediction.shape[1] // 2
    masks_dict: dict[str, np.ndarray] = {}
    for structure, value in categories.items():
        left_mask = np.where(combined_prediction == value, 1, 0)
        right_mask = np.where(combined_prediction == value, 1, 0)
        right_mask[:, midline_x:] = 0
        left_mask[:, :midline_x] = 0
        masks_dict[f"left_{structure}"] = left_mask
        masks_dict[f"right_{structure}"] = right_mask
    return masks_dict


def crop_and_resize_pair(img: Image.Image, size: int, is_mask: bool = False) -> tuple[Image.Image, Image.Image]:
    mid = img.width // 2
    right_half_start = mid if img.width % 2 == 0 else mid + 1
    left_half = img.crop((0, 0, mid, img.height))
    right_half = img.crop((right_half_start, 0, img.width, img.height))
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return (
        left_half.resize((size, size), resample=resample),
        right_half.resize((size, size), resample=resample),
    )


def map_full_to_eye_xy(x: float, y: float, width: int, height: int, side: str, out_size: int) -> tuple[float, float]:
    mid = width // 2
    right_start = mid if width % 2 == 0 else mid + 1
    if side == "L":
        return out_size * (x / float(mid)), out_size * (y / float(height))
    if side == "R":
        w_right = width - right_start
        x_rel = x - right_start
        return out_size * (x_rel / float(w_right)), out_size * (y / float(height))
    raise ValueError(f"Unsupported side: {side}")


def canonical_key(key: str) -> str:
    for prefix, replacement in (
        ("left_", ""),
        ("right_", ""),
        ("sup_l_", "sup_"),
        ("sup_r_", "sup_"),
        ("l_", ""),
        ("r_", ""),
    ):
        if key.startswith(prefix):
            return replacement + key[len(prefix):]
    return key


def sample_prefix_from_stem(stem: str, dataset_name: str) -> str:
    suffix = f"_crop_{dataset_name}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _as_xy(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]), float(value[1])
    arr = np.asarray(value).reshape(-1)
    if arr.size == 2:
        return float(arr[0]), float(arr[1])
    return None


def _canon_eye_landmarks(raw_landmarks: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float] | None]:
    out = {key: None for key in LANDMARK_KEYS}
    for key, value in raw_landmarks.items():
        canon = canonical_key(key)
        if canon in out:
            out[canon] = value
    return out


def split_landmarks_to_eye(
    landmarks_full: dict[str, Any],
    width: int,
    height: int,
    out_size: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    left_eye: dict[str, tuple[float, float]] = {}
    right_eye: dict[str, tuple[float, float]] = {}

    for key in RIGHT_KEYS:
        xy = _as_xy(landmarks_full.get(key))
        if xy is None:
            continue
        left_eye[key] = map_full_to_eye_xy(xy[0], xy[1], width, height, side="R", out_size=out_size)

    for key in LEFT_KEYS:
        xy = _as_xy(landmarks_full.get(key))
        if xy is None:
            continue
        right_eye[key] = map_full_to_eye_xy(xy[0], xy[1], width, height, side="L", out_size=out_size)

    return left_eye, right_eye


def landmark_row(sample_id: str, dataset_name: str, landmark_values: dict[str, tuple[float, float] | None]) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
    }
    for key in LANDMARK_KEYS:
        xy = landmark_values[key]
        row[f"{key}_x"] = float(xy[0]) if xy is not None else np.nan
        row[f"{key}_y"] = float(xy[1]) if xy is not None else np.nan
    return row


def _values_from_landmark_row(row: dict[str, float | str]) -> dict[str, float]:
    return {k: float(v) for k, v in row.items() if k not in {"sample_id", "dataset_name"}}


def _invalid_landmark_columns(values: dict[str, float], out_size: int) -> list[str]:
    invalid = []
    for key, value in values.items():
        if np.isnan(value):
            invalid.append(f"{key}=nan")
        elif value < 0.0 or value > float(out_size):
            invalid.append(f"{key}={value:.6g}")
    return sorted(invalid)


def build_eye_samples_with_failures(
    dataset_name: str,
    image_path: Path,
    mask_path: Path,
    out_size: int,
) -> tuple[list[EyeCropSample], list[EyeCropFailure]]:
    source_id = sample_prefix_from_stem(image_path.stem, dataset_name)

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)
    mask_arr = np.array(mask)
    predictions = extract_and_split_masks(mask_arr)
    extractor = EyeFeatureExtractor(predictions, mask)
    landmarks_full, *_ = extractor.extract_features()
    if landmarks_full is None:
        failures = [
            EyeCropFailure(
                source_id=source_id,
                sample_id=f"{source_id}_{side}",
                dataset_name=dataset_name,
                anatomical_side=side,
                image_path=str(image_path),
                mask_path=str(mask_path),
                reason="landmark_extraction_returned_none",
            )
            for side in ("l", "r")
        ]
        return [], failures

    viewer_left_img, viewer_right_img = crop_and_resize_pair(image, size=out_size, is_mask=False)
    left_landmarks_raw, right_landmarks_raw = split_landmarks_to_eye(
        landmarks_full,
        width=image.width,
        height=image.height,
        out_size=out_size,
    )
    left_landmarks = _canon_eye_landmarks(left_landmarks_raw)
    right_landmarks = _canon_eye_landmarks(right_landmarks_raw)

    samples: list[EyeCropSample] = []
    failures: list[EyeCropFailure] = []
    for side, eye_image, landmark_values in (
        ("l", viewer_right_img, left_landmarks),
        ("r", viewer_left_img, right_landmarks),
    ):
        sample_id = f"{source_id}_{side}"
        image_name = f"{sample_id}.jpg"
        row = landmark_row(sample_id, dataset_name, landmark_values)
        values = _values_from_landmark_row(row)
        invalid = _invalid_landmark_columns(values, out_size)
        if invalid:
            failures.append(
                EyeCropFailure(
                    source_id=source_id,
                    sample_id=sample_id,
                    dataset_name=dataset_name,
                    anatomical_side=side,
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    reason="invalid_landmarks:" + ",".join(invalid),
                )
            )
            continue
        samples.append(
            EyeCropSample(
                sample_id=sample_id,
                source_id=source_id,
                dataset_name=dataset_name,
                anatomical_side=side,
                image_name=image_name,
                image_rel_path=f"{dataset_name}/images/{image_name}",
                image=eye_image,
                landmarks=values,
            )
        )
    return samples, failures


def build_eye_samples(
    dataset_name: str,
    image_path: Path,
    mask_path: Path,
    out_size: int,
) -> tuple[EyeCropSample, EyeCropSample] | None:
    samples, _ = build_eye_samples_with_failures(dataset_name, image_path, mask_path, out_size)
    if len(samples) != 2:
        return None
    return samples[0], samples[1]
