# SEEC Benchmark

SEEC (Standardized External-Eye Corpus) is a corpus and benchmark for
self-supervised representation learning on external-eye/periocular images.
This repository provides benchmark code, example training configs, and fixed splits.

**Start with the [Quickstart](QUICKSTART.md): install, prepare data, download checkpoints, and run.**

<a id="landmark-probing"></a>

## Tasks

| Task                                                        | Resources                                                                |
| ----------------------------------------------------------- | ------------------------------------------------------------------------ |
| Embedding geometry under distribution shift                 | Holdout, Open-HR; restricted Clinic cohort                               |
| In-distribution anatomical landmark recovery                | LM-Celeb, LM-CFD                                                         |
| Cross-dataset anatomical transfer                           | LM-Celeb to LM-CFD                                                       |
| Disease classification: frozen probes and short fine-tuning | Restricted Disease cohort; public bring your own disease (BYOD) protocol |

## Data and checkpoints

- Reconstruct SEEC from this repo: [seec-dataset](https://github.com/monkeygobah/seec-dataset).
- Use [scripts/prepare_benchmark_layout.py](scripts/prepare_benchmark_layout.py) to create a separate input view matching the [fixed manifests](manifests/README.md). See the [layout instructions](QUICKSTART.md#2-prepare-the-image-layout).
- Download the [current checkpoint archive](https://drive.google.com/file/d/1d7z4p7s5l8GPTBHxTFs0vQ_s5DRxWWjK/view?usp=drive_link). 

Pretraining and Holdout use flat, dataset-prefixed **224 x 224** files. Open-HR
uses dataset-relative **512 x 512** files; the supplied geometry configs resize
these to 224 x 224 for encoder input. Both live under the prepared `subset6/` root.

**Clinic and Disease are restricted and not publicly released. Their paper results
are not publicly reproducible.** The BYOD disease command evaluates your own
appropriately governed data; it does not reproduce the paper's clinical cohort.

## Documentation

- [Quickstart](QUICKSTART.md)
- [Benchmark dataset card](DATASET_CARD.md)
- [Fixed manifests](manifests/README.md)
- [Software citation](CITATION.cff)

## Use and licensing

Benchmark code is [MIT licensed](LICENSE). Source-derived data retain their
applicable source terms. Cite this software and the source datasets you use;
include the landmark source when running landmark tasks. The manuscript citation
will be added when its bibliographic metadata are available.

SEEC supports representation-learning research and benchmarking. It is not intended
for face recognition, identity verification, or clinical deployment. Cropping does
not guarantee anonymity; clinical benchmark results are not deployment validation.
