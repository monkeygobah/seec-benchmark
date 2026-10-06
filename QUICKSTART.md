# Quickstart

Choose the task you need. Clinic and Disease datasets are restricted and cannot be reproduced from the public release.

## 1. Install

Commands below use Bash. Run from the repository root:

```bash
git clone https://github.com/monkeygobah/seec-benchmark.git
cd seec-benchmark
python -m pip install -r requirements.txt
export EEB_RELEASE_ROOT="$PWD"
export EEB_DATA_ROOT=/absolute/path/to/benchmark-data
export EEB_CHECKPOINT_ROOT=/absolute/path/to/checkpoints
export EEB_OUTPUT_ROOT=/absolute/path/to/outputs
```

On Windows PowerShell, set the same variables with:

```powershell
$env:EEB_RELEASE_ROOT = (Get-Location).Path.Replace('\', '/')
$env:EEB_DATA_ROOT = 'C:/path/to/benchmark-data'
$env:EEB_CHECKPOINT_ROOT = 'C:/path/to/checkpoints'
$env:EEB_OUTPUT_ROOT = 'C:/path/to/outputs'
```

Use forward slashes in these variables so expansion inside YAML is safe.
Use `$env:EEB_DATA_ROOT` in place of `$EEB_DATA_ROOT` in the command below.

## 2. Prepare the image layout

Obtain permitted sources and reconstruct `SUBSET_6` (224) and `SUBSET_7` (512)
using the [dataset toolkit](https://github.com/monkeygobah/seec-dataset#readme).
Then create a separate benchmark view:

```bash
python scripts/prepare_benchmark_layout.py --subset6-root /path/to/SUBSET_6 --subset7-root /path/to/SUBSET_7 --out-root "$EEB_DATA_ROOT/subset6"
```

The prepared root contains both resolutions:

```text
$EEB_DATA_ROOT/subset6/
  celeb__001049_rfc_OS_224.jpg             # Pretrain / Holdout: flat 224 files
  cfd/
    CFD-AF-200-228-N_rfc_OD_512.jpg       # Open-HR: dataset-relative 512 files
```

The supplied geometry configs resize Open-HR images to 224 for encoder input.
Use the shipped fixed manifests; do not regenerate splits.

Default `--mode auto` creates hard links and falls back to copies when links are
unavailable, including across drives. Treat linked views as read-only: editing a
hard link also changes its source. Use `--mode copy` for independent files.


## 3. Download checkpoints

Download the [current checkpoint archive](https://drive.google.com/file/d/1d7z4p7s5l8GPTBHxTFs0vQ_s5DRxWWjK/view?usp=drive_link).

Arrange the geometry checkpoint folders under:

```text
$EEB_CHECKPOINT_ROOT/
  resnet101_50k/
  vit_b16_50k/
```

Keep each run's `config.yaml` and `checkpoints/` together. The task configs specify
run folders and checkpoint steps. DINOv2/MAE baselines fetch upstream weights when needed.

## 4. Run a task

Geometry (prepared image view and checkpoints required):

```bash
python scripts/run_geometry_eval.py --cfg configs/geometry/resnet101_50k_grid.yaml
python scripts/run_geometry_eval.py --cfg configs/geometry/vit_b16_50k_grid.yaml
```

Landmarks: obtain the [public image/mask pairs](https://zenodo.org/records/13916845)
and arrange them as `landmark_raw/{celeb,cfd}/{images,masks}/` under `EEB_DATA_ROOT`.
Then prepare and evaluate:

```bash
python scripts/prepare_landmark_dataset.py --cfg configs/landmarks/prepare_celeb_cfd.yaml
python scripts/run_landmark_probe.py --cfg configs/landmarks/probe_within_and_transfer.yaml
```

Disease BYOD: place your governed images under `disease_byod/images/` and create
`disease_byod/manifest.csv` under `EEB_DATA_ROOT`:

```csv
image_path,label,group_id
class_a/example_001.png,class_a,subject_001
class_b/example_002.png,class_b,subject_002
```

Paths are relative to `disease_byod/images/` or absolute. Use one `group_id` for
all images from the same subject, including both eyes. Provide at least two groups
per class to form train/test partitions. This CSV illustrates the schema only.

This BYOD config uses an example checkpoint path; set `run_dir` and
`checkpoint_step` to your chosen run before running it. No dedicated disease
checkpoint is required in the released archive.

```bash
python scripts/run_disease_probe.py --cfg configs/disease/byod_disease_classification.yaml
```

Results are written under `EEB_OUTPUT_ROOT`. For an example pretraining run:

```bash
python scripts/train_ssl.py --cfg configs/pretraining/pretrain_10k.yaml
```
