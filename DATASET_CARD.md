# SEEC Benchmark Dataset Card

SEEC means Standardized External-Eye Corpus. This card covers the benchmark
resources and protocols; source-corpus reconstruction is documented in
[seec-dataset](https://github.com/monkeygobah/seec-dataset).

## Resources

| Resource | Images | Purpose / access |
|---|---:|---|
| Pretrain-10K | 10,000 | Fixed unlabeled pretraining; authorized source reconstruction |
| Pretrain-100K | 100,000 | Fixed unlabeled pretraining; authorized source reconstruction |
| Pretrain-1M | 1,000,000 | Fixed unlabeled pretraining; authorized source reconstruction |
| Holdout | 421,730 | Open-source geometry evaluation; authorized source reconstruction |
| Open-HR | 134,969 | High-resolution geometry evaluation; authorized source reconstruction |
| Clinic | 35,082 | Clinical geometry; restricted, not publicly released |
| LM-Celeb | 3,610 | Landmark recovery and transfer; public source image/mask pairs |
| LM-CFD | 1,596 | Landmark recovery and transfer; public source image/mask pairs |
| Disease | 633 | Five-class clinical classification; restricted, not publicly released |

Pretrain/Holdout use 224 x 224 files. Open-HR uses 512 x 512 files, resized to 224
by the supplied geometry configs. See [Quickstart](QUICKSTART.md) for input layout.

## Public reproduction

The repository includes code, configs, fixed split manifests, and small examples.
Full corpus images and clinical images are not bundled. Checkpoints are distributed
separately through the [current checkpoint archive](https://drive.google.com/file/d/1d7z4p7s5l8GPTBHxTFs0vQ_s5DRxWWjK/view?usp=drive_link); permanent archival hosting may change.

Clinic and Disease results from the paper are not publicly reproducible.
The public BYOD disease protocol supports a user's appropriately governed dataset;
it does not supply or reproduce the paper cohort. No clinical access is promised.

Landmark preparation uses [Zenodo record 13916845](https://zenodo.org/records/13916845),
DOI `10.5281/zenodo.13916845`, described by the release as CC BY 4.0.
Follow the source's terms and attribution requirements.

## Intended use and limitations

Use SEEC for external-eye/periocular representation learning, anatomical probing,
and research benchmarking. It is not intended for face recognition, identity
verification, surveillance, or medical deployment. Clinical research results are
not deployment-ready clinical validation, and source-to-clinic generalization
requires further evaluation.

Periocular cropping reduces full-face exposure but does not eliminate identifiability.
Source-specific licenses and institutional restrictions continue to apply.
The [MIT software license](LICENSE) does not grant redistribution rights for
source-derived images. Users must obtain sources under their applicable terms.
