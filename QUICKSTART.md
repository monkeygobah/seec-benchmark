# Quickstart

This is the shortest path through the public release. Full paper-scale
reproduction requires authorized data and downloaded checkpoints.

## Install

```bash
cd seec-benchmark
python -m pip install -r requirements.txt
export EEB_RELEASE_ROOT="$PWD"
export EEB_DATA_ROOT=/path/to/data
export EEB_CHECKPOINT_ROOT=/path/to/checkpoints
export EEB_OUTPUT_ROOT=/path/to/outputs
```

## Arrange Data

```text
$EEB_DATA_ROOT/
  subset6/
  landmark_raw/
    celeb/images/
    celeb/masks/
    cfd/images/
    cfd/masks/
  disease_byod/
    images/
    manifest.csv
```

The paper `Clinic` and `Disease` clinical datasets are not publicly released.
The disease command is for users with their own, IRB approved dataset.

Validate the layout:

```bash
python scripts/validate_release_inputs.py --data-root "$EEB_DATA_ROOT"
```

## Download Checkpoints

```text
Pretrained model checkpoints: [Google Drive link to be added]
```

Place the downloaded checkpoint folders under `$EEB_CHECKPOINT_ROOT`.

## Prepare Landmark Data

Download the public periorbital segmentation dataset:

```text
https://zenodo.org/records/13916845
```

Arrange the raw images and masks as shown above, then run:

```bash
python scripts/prepare_landmark_dataset.py \
  --cfg configs/landmarks/prepare_celeb_cfd.yaml \
  --overwrite
```

## Run Benchmarks

Geometry:

```bash
python scripts/run_geometry_eval.py \
  --cfg configs/geometry/resnet101_50k_grid.yaml
```

Landmarks:

```bash
python scripts/run_landmark_probe.py \
  --cfg configs/landmarks/probe_within_and_transfer.yaml
```

Disease BYOD:

```bash
python scripts/run_disease_probe.py \
  --cfg configs/disease/byod_disease_classification.yaml
```

Disease manifests require:

```csv
image_path,label,group_id
class_a/example_001.png,class_a,subject_001
class_b/example_002.png,class_b,subject_002
```
