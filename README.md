# SEEC Benchmark

The SEEC benchmark is a public release of code, manifests, configs, and
protocols for evaluating self-supervised representations of standardized
external-eye crops. It supports fixed-scale pretraining, embedding geometry
evaluation, anatomical landmark probing, and bring-your-own-data disease
classification.

This repository does not include clinical images from our paper as `Clinic` and `Disease` datasets are not available for public release owing to IRB restrictions however, they are potentially available pending IRB approval and appropriate data use agreements. The public disease task is provided as a protocol for users with their own appropriately governed dataset.

For a short first-run path through the artifact, see `QUICKSTART.md`.

## Installation

```bash
cd seec-benchmark
python -m pip install -r requirements.txt
export EEB_RELEASE_ROOT="$PWD"
export EEB_DATA_ROOT=/path/to/data
export EEB_CHECKPOINT_ROOT=/path/to/checkpoints
export EEB_OUTPUT_ROOT=/path/to/outputs
```

Once all datasets have been prepared and curated, the expected local data layout is as follows:

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

Everything other than diseases_boyd should be present (unless you bring your own data.


Validate the expected layout:

```bash
python scripts/validate_release_inputs.py --data-root "$EEB_DATA_ROOT"
```

You can download all model checkpoints used in our work here:

INSERT LINK

## Release Contents

```text
configs/      benchmark and pretraining configuration files
code/         copied source code used by the release scripts
manifests/    fixed pretraining and evaluation split manifests
scripts/      public command-line entrypoints
tests/        lightweight release integrity tests
```

The released manifests contain one relative image path per line. Real image
crops are not bundled in this v0.1 release.

## Rebuilding Benchmark Subsets

This release assumes that users already have an authorized local copy of the
canonical `subset6` external-eye corpus. Full reconstruction of the external-eye
corpus from all upstream source datasets will be documented in a separate
repository.

That repository will be linked here following acceptance- it is also linked in section 2.5 of our submission.

From an authorized `subset6` corpus, this release provides fixed public-facing
manifests for pretraining and geometry evaluation:

```bash
python scripts/prepare_subset6_splits.py \
  --subset6-root "$EEB_DATA_ROOT/subset6" \
  --out-dir manifests

wc -l manifests/pretrain/pretrain_10k.txt \
      manifests/pretrain/pretrain_100k.txt \
      manifests/pretrain/pretrain_1m.txt \
      manifests/geometry/holdout.txt \
      manifests/geometry/open_hr.txt
```

Expected counts:

```text
10000    manifests/pretrain/pretrain_10k.txt
100000   manifests/pretrain/pretrain_100k.txt
1000000  manifests/pretrain/pretrain_1m.txt
421730   manifests/geometry/holdout.txt
134969   manifests/geometry/open_hr.txt
```

## Periorbital Landmark Dataset Preparation

The landmark benchmark uses public periorbital segmentation data:

- Zenodo record: https://zenodo.org/records/13916845
- DOI: `10.5281/zenodo.13916845`
- License: Creative Commons Attribution 4.0 International
- Files include `periorbital_dataset.zip` plus helper scripts.

After downloading and arranging the raw image/mask pairs, use this layout:

```text
$EEB_DATA_ROOT/landmark_raw/
  celeb/images/
  celeb/masks/
  cfd/images/
  cfd/masks/
```

Prepare unilateral `224x224` eye crops, landmark coordinates, and splits:

```bash
python scripts/prepare_landmark_dataset.py \
  --cfg configs/landmarks/prepare_celeb_cfd.yaml \
  --overwrite
```

Outputs are written under:

```text
$EEB_OUTPUT_ROOT/landmarks/periorbital_224_v2/
  metadata/dataset_manifest.csv
  metadata/landmarks.csv
  metadata/split_assignments.csv
  metadata/prep_failures.csv
  metadata/prep_summary.csv
```

## Benchmark Tasks

Publicly supported tasks:

- `Holdout` embedding geometry on source-distribution external-eye crops
- `Open-HR` embedding geometry on high native-resolution open-source crops
- `LM-Celeb` and `LM-CFD` anatomical landmark probing
- disease classification using a bring-your-own-data manifest

The restricted clinical tasks remain described by the protocol so you can run the same code on your own data if you wish.

## Pretraining Models

Example fixed-compute pretraining configs are provided in `configs/pretraining/`.
Run one with:

```bash
python scripts/train_ssl.py --cfg configs/pretraining/pretrain_10k.yaml
```

Important config fields:

- `data.train_root`: local root containing authorized `subset6` images
- `data.train_manifest`: fixed manifest used for the pretraining subset
- `model.backbone`: encoder architecture, such as `resnet101`
- `model.init`: initialization, such as `random` or `imagenet`
- `ssl.method`: self-supervised objective, such as `infonce`, `vicreg`, or `lejepa`
- `run.total_steps`: fixed training budget
- `dataloader.batch_size`: per-process batch size

## Download Pretrained Checkpoints

Pretrained model checkpoints are not committed to this repository. Download the
checkpoint bundle before running geometry, landmark, or disease benchmarks:

```text
Pretrained model checkpoints: [Google Drive link to be added]
```

Expected checkpoint layout:

```text
$EEB_CHECKPOINT_ROOT/
  resnet101_50k/
  vit_b16_50k/
  vit_b16/
```

The benchmark configs expect this layout when resolving checkpoint paths.

## Embedding Geometry Evaluation

Geometry evaluation extracts frozen embeddings, computes isotropy summaries,
and writes aggregate CSVs under `$EEB_OUTPUT_ROOT/geometry`. Download the
pretrained checkpoints into `$EEB_CHECKPOINT_ROOT` before running these
commands.

ResNet-101 grid:

```bash
python scripts/run_geometry_eval.py \
  --cfg configs/geometry/resnet101_50k_grid.yaml
```

ViT-B/16 grid:

```bash
python scripts/run_geometry_eval.py \
  --cfg configs/geometry/vit_b16_50k_grid.yaml
```

The geometry configs evaluate `Holdout` and `Open-HR` through released manifest
files against the authorized local corpus root.

## Landmark Probing

After preparing the landmark dataset and downloading checkpoints into
`$EEB_CHECKPOINT_ROOT`, run frozen feature extraction, MLP probe training, and
aggregation:

```bash
python scripts/run_landmark_probe.py \
  --cfg configs/landmarks/probe_within_and_transfer.yaml
```

The landmark config includes within-dataset tasks for `LM-Celeb` and `LM-CFD`
and a Celeb-to-CFD transfer task.

## Disease Classification BYOD Benchmark

The public disease benchmark is bring-your-own-data. Provide images and a CSV
manifest:

```csv
image_path,label,group_id
class_a/example_001.png,class_a,subject_001
class_b/example_002.png,class_b,subject_002
```

Required columns:

- `image_path`: path relative to `$EEB_DATA_ROOT/disease_byod/images`, or absolute
- `label`: disease class label
- `group_id`: grouping key used for leakage-safe train/test splitting

Optional columns include `eye`, `patient_id`, `source_image_path`, and
`disease_status`. The split logic groups by `group_id`, which is intended to
prevent left/right eye or repeated-source leakage.

Run the benchmark:

```bash
python scripts/run_disease_probe.py \
  --cfg configs/disease/byod_disease_classification.yaml
```

This command also expects any external-eye pretrained checkpoints referenced by
`configs/disease/byod_disease_classification.yaml` to be available under
`$EEB_CHECKPOINT_ROOT`.

## Licensing, Citation, and Intended Use

The benchmark code is released under the MIT License. Dataset components retain
their original source licenses and data-use restrictions. Users are responsible
for obtaining source datasets and complying with their terms.

This benchmark is intended for representation learning research on external-eye
images. It is not intended for face recognition, identity verification,
surveillance, or deployment as a medical diagnostic system.

If you use this release, cite the associated paper and the original source
datasets, including the public periorbital segmentation dataset when using the
landmark benchmark.

## Development Checks

These checks are optional, but useful after editing the release files. From the
`BENCHMARK_RELEASE/` directory, run:

```bash
python -m pytest tests -q
```

The fixed public manifests should also have the expected row counts:

```bash
wc -l manifests/pretrain/pretrain_10k.txt \
      manifests/pretrain/pretrain_100k.txt \
      manifests/pretrain/pretrain_1m.txt \
      manifests/geometry/holdout.txt \
      manifests/geometry/open_hr.txt
```

Expected output:

```text
10000    manifests/pretrain/pretrain_10k.txt
100000   manifests/pretrain/pretrain_100k.txt
1000000  manifests/pretrain/pretrain_1m.txt
421730   manifests/geometry/holdout.txt
134969   manifests/geometry/open_hr.txt
```
