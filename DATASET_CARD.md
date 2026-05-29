# Dataset Card

External Eye Benchmark is organized around standardized unilateral external-eye
crops and associated benchmark protocols.

## Components

- `Pretrain-10K`, `Pretrain-100K`, `Pretrain-1M`: fixed unlabeled pretraining
  subsets drawn from the canonical `subset6` corpus.
- `Holdout`: held-out open-source source-distribution evaluation split.
- `Open-HR`: high native-resolution open-source evaluation split.
- `LM-Celeb`, `LM-CFD`: landmark probe datasets derived from public image/mask
  pairs.
- `Disease`: a bring-your-own-data disease classification protocol. The paper
  uses restricted clinical data; those images and labels are not released.

## Public and Restricted Data

This release includes manifests and code. It does not include full upstream
source datasets, real image crops, clinical images, checkpoints, generated
embeddings, or training runs.

The `Clinic` geometry set and paper `Disease` classification set are not
available for public release. Authorized users can run the same protocols on
appropriately governed local datasets.

## Landmark Source Dataset

The landmark benchmark uses public periorbital segmentation data:

- Zenodo record: https://zenodo.org/records/13916845
- DOI: `10.5281/zenodo.13916845`
- License: Creative Commons Attribution 4.0 International

## Intended Use

The release supports representation learning research and benchmarking for
external-eye image crops. It is not intended for face recognition, identity
verification, surveillance, or deployment as a medical diagnostic system.

## Privacy and Licensing

External-eye crops reduce full-face exposure but do not remove identifiability
risks. Users are responsible for obtaining source datasets, following original
license terms, and complying with institutional data-use restrictions.
