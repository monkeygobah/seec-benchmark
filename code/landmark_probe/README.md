# Landmark Probe

Config-driven downstream landmark probe pipeline for frozen SSL backbones.

Stages:

- `prepare`: rebuild canonical `224x224` periorbital dataset with fixed splits
- `extract`: reopen training runs/checkpoints and write pooled backbone embeddings
- `probe`: train an MLP landmark regressor on frozen embeddings
- `aggregate`: summarize completed probe runs into flat CSV tables

Entry points live in `scripts/`:

- `run_landmark_prepare.py`
- `run_landmark_extract.py`
- `run_landmark_probe.py`
- `run_landmark_aggregate.py`

The active 50k-step configs are:

- dataset: `landmark_probe/configs/datasets/periorbital_224_v2.yaml`
- extraction: `landmark_probe/configs/studies/followup_50k_extract_all_poolings.yaml`
- G4 probe matrix: `landmark_probe/configs/studies/followup_50k_probe_matrix.yaml`

Useful commands:

```bash
python scripts/run_landmark_prepare.py --cfg landmark_probe/configs/datasets/periorbital_224_v2.yaml --overwrite
python scripts/run_landmark_extract.py --cfg landmark_probe/configs/studies/followup_50k_extract_all_poolings.yaml --overwrite
python scripts/run_landmark_probe.py --cfg landmark_probe/configs/studies/followup_50k_probe_matrix.yaml
python scripts/run_landmark_aggregate.py --cfg landmark_probe/configs/studies/followup_50k_probe_matrix.yaml
```

Or run the full suite with logs:

```bash
scripts/landmark_probe/run_followup_50k_matrix.sh --prepare --overwrite-extract
```
